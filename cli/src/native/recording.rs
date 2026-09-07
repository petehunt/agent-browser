use serde_json::{json, Value};
use std::collections::VecDeque;
use std::path::Path;
use std::process::Stdio;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::io::AsyncWriteExt;
use tokio::sync::{mpsc, oneshot};

use super::cdp::client::CdpClient;
use super::cdp::types::{AttachToTargetParams, AttachToTargetResult};

/// Capture rate used when the caller does not ask for one. 30 fps reads as
/// smooth motion, so scrolls, hovers, and CSS transitions survive the
/// recording instead of turning into a slideshow.
pub const DEFAULT_FPS: u32 = 30;

/// Highest capture rate the recorder accepts. 60 fps is worth asking for on
/// short, motion-heavy clips (drag interactions, animation, scroll polish
/// work) where the extra temporal detail is the point.
pub const MAX_FPS: u32 = 60;

/// Changed-pixel ratio that selects a contact-sheet frame when the caller
/// does not provide one. Five percent filters minor animation while retaining
/// meaningful UI transitions.
pub const DEFAULT_CONTACT_SHEET_THRESHOLD: f64 = 0.05;

/// Contact sheets stay reviewable and memory-bounded during long recordings.
pub const MAX_CONTACT_SHEET_FRAMES: usize = 100;

const CONTACT_SHEET_COLUMNS: u32 = 4;
const CONTACT_SHEET_CELL_WIDTH: u32 = 640;
const CONTACT_SHEET_LABEL_HEIGHT: u32 = 24;
const CONTACT_SHEET_GAP: u32 = 8;
const CONTACT_SHEET_DIFF_WIDTH: u32 = 320;
const CONTACT_SHEET_PIXEL_DELTA: u8 = 24;
const CONTACT_SHEET_TILE_SIZE: u32 = 8;
const CONTACT_SHEET_MIN_TILE_PIXELS: u32 = 4;
const CONTACT_SHEET_MIN_REGION_PIXELS: u64 = 8;
const CONTACT_SHEET_REGION_PADDING_TILES: u32 = 1;
const CONTACT_SHEET_REGION_MERGE_GAP_TILES: u32 = 4;

/// Rate above which the deferred encoder uses a second thread.
const HIGH_FPS_THRESHOLD: u32 = 30;

/// VP8 budget chosen for readable UI text and thin drawing strokes.
const WEBM_BITRATE_KBPS: u32 = 8000;

/// Upper bound on waiting for Chrome to acknowledge screencast teardown.
/// The page may already be gone by the time a recording stops.
const TEARDOWN_TIMEOUT: Duration = Duration::from_secs(2);

/// Reject frame rates the pipeline cannot honor.
pub fn validate_fps(fps: u32) -> Result<u32, String> {
    if fps == 0 || fps > MAX_FPS {
        return Err(format!(
            "Invalid fps: {} is out of range (valid range: 1-{})",
            fps, MAX_FPS
        ));
    }
    Ok(fps)
}

/// The CDP session a recording attaches to its page target for its screencast.
///
/// Chrome keeps one screencast per session and the live stream already runs
/// one on the page session, so the recorder attaches a second flattened
/// session to the same target. The daemon's event handlers use this to avoid
/// treating that attachment as a tab.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CaptureSession {
    pub target_id: String,
    /// `None` while `Target.attachToTarget` is in flight: Chrome emits
    /// `Target.attachedToTarget` before it answers, so in that window the
    /// attachment is recognised by target instead.
    pub session_id: Option<String>,
}

impl CaptureSession {
    /// Whether a `Target.attachedToTarget` event is this recorder's own
    /// attachment rather than a tab the daemon should track.
    pub fn owns_attachment(&self, target_id: &str, session_id: &str) -> bool {
        match self.session_id.as_deref() {
            Some(own) => own == session_id,
            None => self.target_id == target_id,
        }
    }
}

pub type SharedCaptureSession = Arc<Mutex<Option<CaptureSession>>>;

#[derive(Clone, Copy, Debug, Default)]
pub struct RecordingCursorState {
    pub x: f64,
    pub y: f64,
    pub buttons: i32,
    pub visible: bool,
}

#[derive(Clone, Debug, Default)]
pub struct RecordingCursorHistory {
    samples: VecDeque<(f64, RecordingCursorState)>,
}

impl RecordingCursorHistory {
    /// Record input after Chrome acknowledges it.
    pub fn record(&mut self, x: f64, y: f64, buttons: i32) {
        self.record_at(cursor_timestamp(), x, y, buttons);
    }

    pub fn record_at(&mut self, timestamp: f64, x: f64, y: f64, buttons: i32) {
        self.samples.push_back((
            timestamp,
            RecordingCursorState {
                x,
                y,
                buttons,
                visible: true,
            },
        ));
    }

    pub fn at(&self, timestamp: f64) -> RecordingCursorState {
        self.samples
            .iter()
            .rev()
            .find(|(time, _)| *time <= timestamp)
            .map(|(_, state)| *state)
            .unwrap_or_default()
    }

    fn interpolated_at(&self, timestamp: f64) -> RecordingCursorState {
        let Some(next_index) = self.samples.iter().position(|(time, _)| *time > timestamp) else {
            return self.at(timestamp);
        };
        if next_index == 0 {
            return RecordingCursorState::default();
        }
        let (before_time, before) = self.samples[next_index - 1];
        let (after_time, after) = self.samples[next_index];
        if before.buttons != 0 || after.buttons != 0 || after_time <= before_time {
            return before;
        }
        let progress = ((timestamp - before_time) / (after_time - before_time)).clamp(0.0, 1.0);
        RecordingCursorState {
            x: before.x + (after.x - before.x) * progress,
            y: before.y + (after.y - before.y) * progress,
            ..before
        }
    }

    fn click_starts(&self) -> Vec<(f64, RecordingCursorState)> {
        let mut previous_buttons = 0;
        self.samples
            .iter()
            .filter_map(|&(timestamp, state)| {
                let started = previous_buttons == 0 && state.buttons != 0;
                previous_buttons = state.buttons;
                started.then_some((timestamp, state))
            })
            .collect()
    }
}

pub fn cursor_timestamp() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64()
}

pub type SharedRecordingCursor = Arc<Mutex<RecordingCursorHistory>>;

const CURSOR_PATH: [(f64, f64); 4] = [(0.0, 0.0), (14.0, 8.5), (7.5, 10.0), (4.0, 16.0)];
const CURSOR_BASE_SCALE: f64 = 28.0 / 24.0;
const CURSOR_PRESS_SCALE: f64 = 0.8;
const CURSOR_STROKE_WIDTH: f64 = 1.5;
const CURSOR_SUPERSAMPLE: usize = 4;
const RIPPLE_DURATION_SECS: f64 = 0.4;
const RIPPLE_FILL_RADIUS: f64 = 24.0;
const RIPPLE_RING_RADIUS: f64 = 32.0;
const RIPPLE_COLOR: [u8; 3] = [96, 165, 250];

fn point_in_polygon(points: &[(f64, f64)], x: f64, y: f64) -> bool {
    let mut inside = false;
    let mut previous = points.len() - 1;
    for (current, &(cx, cy)) in points.iter().enumerate() {
        let (px, py) = points[previous];
        if (cy > y) != (py > y) && x < (px - cx) * (y - cy) / (py - cy) + cx {
            inside = !inside;
        }
        previous = current;
    }
    inside
}

fn distance_to_segment(x: f64, y: f64, start: (f64, f64), end: (f64, f64)) -> f64 {
    let dx = end.0 - start.0;
    let dy = end.1 - start.1;
    let length_squared = dx * dx + dy * dy;
    if length_squared == 0.0 {
        return (x - start.0).hypot(y - start.1);
    }
    let t = (((x - start.0) * dx + (y - start.1) * dy) / length_squared).clamp(0.0, 1.0);
    (x - (start.0 + t * dx)).hypot(y - (start.1 + t * dy))
}

fn distance_to_polygon(points: &[(f64, f64)], x: f64, y: f64) -> f64 {
    points
        .iter()
        .zip(points.iter().cycle().skip(1))
        .take(points.len())
        .map(|(&start, &end)| distance_to_segment(x, y, start, end))
        .fold(f64::INFINITY, f64::min)
}

fn blend_pixel(pixel: &mut image::Rgb<u8>, color: [u8; 3], alpha: f64) {
    let alpha = alpha.clamp(0.0, 1.0);
    for (channel, source) in pixel.0.iter_mut().zip(color) {
        *channel = ((*channel as f64 * (1.0 - alpha)) + (source as f64 * alpha)).round() as u8;
    }
}

fn composite_ripple(frame: &mut image::RgbImage, x: f64, y: f64, progress: f64) {
    if !(0.0..1.0).contains(&progress) {
        return;
    }
    let fill_radius = RIPPLE_FILL_RADIUS * progress;
    let ring_radius = RIPPLE_RING_RADIUS * progress;
    let fill_opacity = 0.8 * (1.0 - progress);
    let ring_opacity = 0.9 * (1.0 - progress);
    let bound = RIPPLE_RING_RADIUS.ceil() as i32 + 2;
    let origin_x = x.round() as i32;
    let origin_y = y.round() as i32;
    let samples = (CURSOR_SUPERSAMPLE * CURSOR_SUPERSAMPLE) as f64;

    for py in origin_y - bound..=origin_y + bound {
        for px in origin_x - bound..=origin_x + bound {
            if px < 0 || py < 0 || px >= frame.width() as i32 || py >= frame.height() as i32 {
                continue;
            }
            let mut fill_coverage = 0.0;
            let mut ring_coverage = 0.0;
            for sy in 0..CURSOR_SUPERSAMPLE {
                for sx in 0..CURSOR_SUPERSAMPLE {
                    let sample_x = px as f64 + (sx as f64 + 0.5) / CURSOR_SUPERSAMPLE as f64;
                    let sample_y = py as f64 + (sy as f64 + 0.5) / CURSOR_SUPERSAMPLE as f64;
                    let distance = (sample_x - x).hypot(sample_y - y);
                    fill_coverage += if distance <= fill_radius { 1.0 } else { 0.0 };
                    ring_coverage += if (distance - ring_radius).abs() <= 1.0 {
                        1.0
                    } else {
                        0.0
                    };
                }
            }
            let pixel = frame.get_pixel_mut(px as u32, py as u32);
            blend_pixel(pixel, RIPPLE_COLOR, fill_opacity * fill_coverage / samples);
            blend_pixel(pixel, RIPPLE_COLOR, ring_opacity * ring_coverage / samples);
        }
    }
}

/// Draw the recording pointer after the clean frame has been analyzed for the
/// contact sheet, keeping presentation pixels out of change detection.
fn composite_cursor(frame: &mut image::RgbImage, cursor: RecordingCursorState) {
    if !cursor.visible {
        return;
    }
    let pressed_scale = if cursor.buttons == 0 {
        1.0
    } else {
        CURSOR_PRESS_SCALE
    };
    let scale = CURSOR_BASE_SCALE * pressed_scale;
    let points: Vec<(f64, f64)> = CURSOR_PATH
        .iter()
        .map(|&(x, y)| (cursor.x + x * scale, cursor.y + y * scale))
        .collect();
    let max_x = points.iter().map(|point| point.0).fold(cursor.x, f64::max);
    let max_y = points.iter().map(|point| point.1).fold(cursor.y, f64::max);
    let samples = (CURSOR_SUPERSAMPLE * CURSOR_SUPERSAMPLE) as f64;

    for py in (cursor.y.floor() as i32 - 4)..=(max_y.ceil() as i32 + 5) {
        for px in (cursor.x.floor() as i32 - 4)..=(max_x.ceil() as i32 + 5) {
            if px < 0 || py < 0 || px >= frame.width() as i32 || py >= frame.height() as i32 {
                continue;
            }
            let mut fill_coverage = 0.0;
            let mut stroke_coverage = 0.0;
            let mut shadow_alpha = 0.0;
            for sy in 0..CURSOR_SUPERSAMPLE {
                for sx in 0..CURSOR_SUPERSAMPLE {
                    let sample_x = px as f64 + (sx as f64 + 0.5) / CURSOR_SUPERSAMPLE as f64;
                    let sample_y = py as f64 + (sy as f64 + 0.5) / CURSOR_SUPERSAMPLE as f64;
                    let inside = point_in_polygon(&points, sample_x, sample_y);
                    let edge_distance = distance_to_polygon(&points, sample_x, sample_y);
                    fill_coverage += if inside { 1.0 } else { 0.0 };
                    stroke_coverage += if edge_distance <= CURSOR_STROKE_WIDTH * scale / 2.0 {
                        1.0
                    } else {
                        0.0
                    };

                    let shadow_x = sample_x;
                    let shadow_y = sample_y - 1.0;
                    let shadow_inside = point_in_polygon(&points, shadow_x, shadow_y);
                    let shadow_distance = distance_to_polygon(&points, shadow_x, shadow_y);
                    let shadow_sample = if shadow_inside {
                        0.5
                    } else if shadow_distance < 3.0 {
                        0.5 * (1.0 - shadow_distance / 3.0).powi(2)
                    } else {
                        0.0
                    };
                    shadow_alpha += shadow_sample;
                }
            }
            let pixel = frame.get_pixel_mut(px as u32, py as u32);
            blend_pixel(pixel, [0, 0, 0], shadow_alpha / samples);
            blend_pixel(pixel, [255, 255, 255], fill_coverage / samples);
            blend_pixel(pixel, [0, 0, 0], stroke_coverage / samples);
        }
    }
}

/// PPM carries exact RGB pixels to FFmpeg and supports frame-size changes.
/// Only the final video encoder compresses the composited frame.
fn cursor_video_frame(source: &[u8], cursor: RecordingCursorState) -> Result<Vec<u8>, String> {
    let mut frame = image::load_from_memory(source)
        .map_err(|e| format!("Failed to decode recording frame: {}", e))?
        .to_rgb8();
    composite_cursor(&mut frame, cursor);
    let mut output = format!("P6\n{} {}\n255\n", frame.width(), frame.height()).into_bytes();
    output.extend_from_slice(frame.as_raw());
    Ok(output)
}

pub struct RecordingState {
    pub active: bool,
    pub output_path: String,
    /// Capture rate for the active (or most recent) recording.
    pub fps: u32,
    /// Frames written to the file, including frames held through gaps.
    pub frame_count: u64,
    /// Distinct frames received from the screencast.
    pub captured_count: u64,
    pub contact_sheet_frame_count: u64,
    pub capture_task: Option<tokio::task::JoinHandle<Result<(), String>>>,
    pub shared_frame_count: Option<Arc<AtomicU64>>,
    pub shared_captured_count: Option<Arc<AtomicU64>>,
    pub shared_contact_sheet_count: Option<Arc<AtomicU64>>,
    pub cancel_tx: Option<oneshot::Sender<()>>,
    /// Shared with the daemon's event handlers.
    pub capture_session: SharedCaptureSession,
    /// Whether the encoded video includes a synthetic pointer.
    pub cursor: bool,
    /// Pointer state composited onto encoded frames after contact-sheet analysis.
    pub shared_cursor: SharedRecordingCursor,
    /// Whether the capture task exports selected frames as a PNG sheet.
    pub contact_sheet: bool,
    pub contact_sheet_threshold: f64,
    pub contact_sheet_path: Option<String>,
}

impl RecordingState {
    pub fn new() -> Self {
        Self {
            active: false,
            output_path: String::new(),
            fps: DEFAULT_FPS,
            frame_count: 0,
            captured_count: 0,
            contact_sheet_frame_count: 0,
            capture_task: None,
            shared_frame_count: None,
            shared_captured_count: None,
            shared_contact_sheet_count: None,
            cancel_tx: None,
            capture_session: Arc::new(Mutex::new(None)),
            cursor: false,
            shared_cursor: Arc::new(Mutex::new(RecordingCursorHistory::default())),
            contact_sheet: false,
            contact_sheet_threshold: DEFAULT_CONTACT_SHEET_THRESHOLD,
            contact_sheet_path: None,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct RecordingOptions {
    pub fps: Option<u32>,
    pub cursor: bool,
    pub contact_sheet: bool,
    pub contact_sheet_threshold: f64,
}

impl Default for RecordingOptions {
    fn default() -> Self {
        Self {
            fps: None,
            cursor: false,
            contact_sheet: false,
            contact_sheet_threshold: DEFAULT_CONTACT_SHEET_THRESHOLD,
        }
    }
}

pub fn validate_contact_sheet_threshold(threshold: f64) -> Result<f64, String> {
    if threshold.is_finite() && (0.0..=1.0).contains(&threshold) {
        Ok(threshold)
    } else {
        Err(format!(
            "Invalid contact sheet threshold: {} is out of range (valid range: 0-1)",
            threshold
        ))
    }
}

pub fn contact_sheet_path(output_path: &str) -> String {
    let path = Path::new(output_path);
    let stem = path
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or("recording");
    let filename = format!("{}.contact-sheet.png", stem);
    path.parent()
        .unwrap_or_else(|| Path::new(""))
        .join(filename)
        .to_string_lossy()
        .to_string()
}

/// [`CaptureSession::owns_attachment`] against the shared slot.
pub fn owns_attachment(shared: &SharedCaptureSession, target_id: &str, session_id: &str) -> bool {
    shared
        .lock()
        .ok()
        .and_then(|guard| {
            guard
                .as_ref()
                .map(|c| c.owns_attachment(target_id, session_id))
        })
        .unwrap_or(false)
}

pub fn recording_start(
    state: &mut RecordingState,
    path: &str,
    options: RecordingOptions,
) -> Result<Value, String> {
    if state.active {
        return Err("Recording already active".to_string());
    }

    let fps = validate_fps(options.fps.unwrap_or(DEFAULT_FPS))?;
    let threshold = validate_contact_sheet_threshold(options.contact_sheet_threshold)?;

    state.active = true;
    state.output_path = path.to_string();
    state.fps = fps;
    state.frame_count = 0;
    state.captured_count = 0;
    state.contact_sheet_frame_count = 0;
    state.cursor = options.cursor;
    if let Ok(mut cursor) = state.shared_cursor.lock() {
        *cursor = RecordingCursorHistory::default();
    }
    state.contact_sheet = options.contact_sheet;
    state.contact_sheet_threshold = threshold;
    state.contact_sheet_path = options.contact_sheet.then(|| contact_sheet_path(path));

    let mut result = json!({
        "started": true,
        "path": path,
        "fps": fps,
        "cursor": options.cursor,
        "contactSheet": options.contact_sheet
    });
    if let Some(ref contact_path) = state.contact_sheet_path {
        result["contactSheetPath"] = json!(contact_path);
        result["contactSheetThreshold"] = json!(threshold);
    }
    Ok(result)
}

pub fn recording_stop(state: &mut RecordingState) -> Result<Value, String> {
    if !state.active {
        return Err("No recording in progress".to_string());
    }

    state.active = false;

    if state.frame_count == 0 {
        return Err("No frames captured".to_string());
    }

    let mut result = json!({
        "path": &state.output_path,
        "frames": state.frame_count,
        "capturedFrames": state.captured_count,
        "fps": state.fps,
    });
    if let Some(ref path) = state.contact_sheet_path {
        result["contactSheetPath"] = json!(path);
        result["contactSheetFrames"] = json!(state.contact_sheet_frame_count);
        result["contactSheetThreshold"] = json!(state.contact_sheet_threshold);
    }
    Ok(result)
}

fn build_ffmpeg_command(output_path: &str, fps: u32, cursor: bool) -> tokio::process::Command {
    let mut cmd = tokio::process::Command::new("ffmpeg");
    let high_fps = fps > HIGH_FPS_THRESHOLD;

    cmd.args(["-y"])
        .args(["-avioflags", "direct"])
        .args([
            "-fpsprobesize",
            "0",
            "-probesize",
            "32",
            "-analyzeduration",
            "0",
        ])
        .args([
            "-f",
            "image2pipe",
            "-c:v",
            if cursor { "ppm" } else { "png" },
            "-framerate",
            &fps.to_string(),
            "-i",
            "pipe:0",
        ])
        .args(["-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2"]);

    if output_path.ends_with(".webm") {
        cmd.args(["-c:v", "libvpx", "-crf", "18"])
            .args(["-b:v", &format!("{}k", WEBM_BITRATE_KBPS)]);
    } else {
        cmd.args(["-c:v", "libx264", "-preset", "ultrafast"]);
    }

    // One encoder thread keeps CPU away from the browser at ordinary rates;
    // above 30 fps the encoder needs a second one to drain the pipe in time.
    cmd.args(["-pix_fmt", "yuv420p"])
        .args(["-threads", if high_fps { "2" } else { "1" }])
        .arg(output_path)
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .kill_on_drop(true);

    cmd
}

#[derive(Debug)]
struct ContactSheetFrame {
    image_data: Vec<u8>,
    elapsed_ms: u64,
    cursor: RecordingCursorState,
    device_width: f64,
    device_height: f64,
}

struct ContactSheetCollector {
    threshold: f64,
    selected: Vec<ContactSheetFrame>,
    previous_selected: Option<image::RgbImage>,
    latest: Option<ContactSheetFrame>,
}

impl ContactSheetCollector {
    fn new(threshold: f64) -> Self {
        Self {
            threshold,
            selected: Vec::new(),
            previous_selected: None,
            latest: None,
        }
    }

    fn consider(
        &mut self,
        image_data: &[u8],
        elapsed: Duration,
        cursor: RecordingCursorState,
        device_width: f64,
        device_height: f64,
    ) {
        let elapsed_ms = elapsed.as_millis().min(u64::MAX as u128) as u64;
        self.latest = Some(ContactSheetFrame {
            image_data: image_data.to_vec(),
            elapsed_ms,
            cursor,
            device_width,
            device_height,
        });
        if self.selected.len() >= MAX_CONTACT_SHEET_FRAMES {
            return;
        }
        let Ok(source) = image::load_from_memory(image_data) else {
            return;
        };
        let source_width = source.width().max(1);
        let source_height = source.height().max(1);
        let diff_height = ((source_height as f64 * CONTACT_SHEET_DIFF_WIDTH as f64
            / source_width as f64)
            .round() as u32)
            .max(1);
        let thumbnail = source
            .resize_exact(
                CONTACT_SHEET_DIFF_WIDTH,
                diff_height,
                image::imageops::FilterType::Triangle,
            )
            .to_rgb8();

        let selection = match self.previous_selected.as_ref() {
            None => Some(()),
            Some(previous) => {
                let (ratio, _) = changed_pixel_regions(previous, &thumbnail);
                (ratio > 0.0 && ratio >= self.threshold).then_some(())
            }
        };
        if selection.is_some() {
            self.previous_selected = Some(thumbnail);
            self.selected.push(ContactSheetFrame {
                image_data: image_data.to_vec(),
                elapsed_ms,
                cursor,
                device_width,
                device_height,
            });
        }
    }

    fn finish(mut self) -> Vec<ContactSheetFrame> {
        if let Some(latest) = self.latest {
            let already_selected = self
                .selected
                .last()
                .is_some_and(|frame| frame.elapsed_ms == latest.elapsed_ms);
            if !already_selected {
                if self.selected.len() >= MAX_CONTACT_SHEET_FRAMES {
                    self.selected.pop();
                }
                self.selected.push(latest);
            }
        }
        self.selected
    }
}

/// Ratio and normalized bounds of visually changed tile clusters. Tile density
/// suppresses isolated noise while preserving separate areas of page activity.
fn changed_pixel_regions(
    before: &image::RgbImage,
    after: &image::RgbImage,
) -> (f64, Vec<[f32; 4]>) {
    if before.dimensions() != after.dimensions() {
        return (1.0, vec![[0.0, 0.0, 1.0, 1.0]]);
    }
    let (width, height) = after.dimensions();
    if width == 0 || height == 0 {
        return (0.0, Vec::new());
    }
    let mut changed = 0u64;
    let tiles_wide = width.div_ceil(CONTACT_SHEET_TILE_SIZE);
    let tiles_high = height.div_ceil(CONTACT_SHEET_TILE_SIZE);
    let mut tile_counts = vec![0u32; (tiles_wide * tiles_high) as usize];
    for y in 0..height {
        for x in 0..width {
            let a = before.get_pixel(x, y).0;
            let b = after.get_pixel(x, y).0;
            let delta = a
                .iter()
                .zip(b.iter())
                .map(|(left, right)| left.abs_diff(*right))
                .max()
                .unwrap_or(0);
            if delta >= CONTACT_SHEET_PIXEL_DELTA {
                changed += 1;
                let tile_x = x / CONTACT_SHEET_TILE_SIZE;
                let tile_y = y / CONTACT_SHEET_TILE_SIZE;
                tile_counts[(tile_y * tiles_wide + tile_x) as usize] += 1;
            }
        }
    }
    if changed == 0 {
        return (0.0, Vec::new());
    }
    let ratio = changed as f64 / (width as u64 * height as u64) as f64;
    let mut components: Vec<(u64, u32, u32, u32, u32)> = Vec::new();
    for start_y in 0..tiles_high {
        for start_x in 0..tiles_wide {
            let start = (start_y * tiles_wide + start_x) as usize;
            if tile_counts[start] < CONTACT_SHEET_MIN_TILE_PIXELS {
                continue;
            }
            let initial_pixels = tile_counts[start] as u64;
            tile_counts[start] = 0;
            let mut pending = std::collections::VecDeque::from([(start_x, start_y)]);
            let (mut min_x, mut min_y, mut max_x, mut max_y) = (start_x, start_y, start_x, start_y);
            let mut pixels = initial_pixels;
            while let Some((x, y)) = pending.pop_front() {
                min_x = min_x.min(x);
                min_y = min_y.min(y);
                max_x = max_x.max(x);
                max_y = max_y.max(y);
                for neighbor_y in y.saturating_sub(1)..=(y + 1).min(tiles_high - 1) {
                    for neighbor_x in x.saturating_sub(1)..=(x + 1).min(tiles_wide - 1) {
                        let neighbor = (neighbor_y * tiles_wide + neighbor_x) as usize;
                        if tile_counts[neighbor] >= CONTACT_SHEET_MIN_TILE_PIXELS {
                            pixels += tile_counts[neighbor] as u64;
                            tile_counts[neighbor] = 0;
                            pending.push_back((neighbor_x, neighbor_y));
                        }
                    }
                }
            }
            if pixels >= CONTACT_SHEET_MIN_REGION_PIXELS {
                components.push((pixels, min_x, min_y, max_x, max_y));
            }
        }
    }
    let gap = CONTACT_SHEET_REGION_MERGE_GAP_TILES;
    // Revisit all pairs after a union: the enlarged region may now reach a
    // component inspected earlier. Never discard regions to meet a box limit.
    let mut index = 0;
    while index < components.len() {
        let mut other = index + 1;
        while other < components.len() {
            let (_, ax1, ay1, ax2, ay2) = components[index];
            let (_, bx1, by1, bx2, by2) = components[other];
            let close = ax1 <= bx2.saturating_add(gap)
                && bx1 <= ax2.saturating_add(gap)
                && ay1 <= by2.saturating_add(gap)
                && by1 <= ay2.saturating_add(gap);
            if close {
                let merged = components.swap_remove(other);
                components[index].0 += merged.0;
                components[index].1 = components[index].1.min(merged.1);
                components[index].2 = components[index].2.min(merged.2);
                components[index].3 = components[index].3.max(merged.3);
                components[index].4 = components[index].4.max(merged.4);
                index = 0;
                other = 1;
            } else {
                other += 1;
            }
        }
        index += 1;
    }
    components.sort_by_key(|&(_, min_x, min_y, _, _)| (min_y, min_x));
    let regions = components
        .into_iter()
        .map(|(_, min_tile_x, min_tile_y, max_tile_x, max_tile_y)| {
            let min_x = min_tile_x.saturating_sub(CONTACT_SHEET_REGION_PADDING_TILES)
                * CONTACT_SHEET_TILE_SIZE;
            let min_y = min_tile_y.saturating_sub(CONTACT_SHEET_REGION_PADDING_TILES)
                * CONTACT_SHEET_TILE_SIZE;
            let max_x = ((max_tile_x + CONTACT_SHEET_REGION_PADDING_TILES + 1)
                * CONTACT_SHEET_TILE_SIZE)
                .min(width);
            let max_y = ((max_tile_y + CONTACT_SHEET_REGION_PADDING_TILES + 1)
                * CONTACT_SHEET_TILE_SIZE)
                .min(height);
            [
                min_x as f32 / width as f32,
                min_y as f32 / height as f32,
                (max_x - min_x) as f32 / width as f32,
                (max_y - min_y) as f32 / height as f32,
            ]
        })
        .collect();
    (ratio, regions)
}

fn format_contact_timestamp(milliseconds: u64) -> String {
    let hours = milliseconds / 3_600_000;
    let minutes = (milliseconds / 60_000) % 60;
    let seconds = (milliseconds / 1_000) % 60;
    let millis = milliseconds % 1_000;
    format!("{:02}:{:02}:{:02}.{:03}", hours, minutes, seconds, millis)
}

fn glyph_rows(character: char) -> [u8; 7] {
    match character {
        '0' => [
            0b01110, 0b10001, 0b10011, 0b10101, 0b11001, 0b10001, 0b01110,
        ],
        '1' => [
            0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110,
        ],
        '2' => [
            0b01110, 0b10001, 0b00001, 0b00010, 0b00100, 0b01000, 0b11111,
        ],
        '3' => [
            0b11110, 0b00001, 0b00001, 0b01110, 0b00001, 0b00001, 0b11110,
        ],
        '4' => [
            0b00010, 0b00110, 0b01010, 0b10010, 0b11111, 0b00010, 0b00010,
        ],
        '5' => [
            0b11111, 0b10000, 0b10000, 0b11110, 0b00001, 0b00001, 0b11110,
        ],
        '6' => [
            0b01110, 0b10000, 0b10000, 0b11110, 0b10001, 0b10001, 0b01110,
        ],
        '7' => [
            0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b01000, 0b01000,
        ],
        '8' => [
            0b01110, 0b10001, 0b10001, 0b01110, 0b10001, 0b10001, 0b01110,
        ],
        '9' => [
            0b01110, 0b10001, 0b10001, 0b01111, 0b00001, 0b00001, 0b01110,
        ],
        ':' => [0, 0b00100, 0b00100, 0, 0b00100, 0b00100, 0],
        '.' => [0, 0, 0, 0, 0, 0b00100, 0b00100],
        _ => [0; 7],
    }
}

fn draw_timestamp(image: &mut image::RgbaImage, x: u32, y: u32, value: &str) {
    const SCALE: u32 = 2;
    for (index, character) in value.chars().enumerate() {
        for (row, bits) in glyph_rows(character).iter().enumerate() {
            for column in 0..5 {
                if bits & (1 << (4 - column)) == 0 {
                    continue;
                }
                for dy in 0..SCALE {
                    for dx in 0..SCALE {
                        let px = x + index as u32 * 6 * SCALE + column * SCALE + dx;
                        let py = y + row as u32 * SCALE + dy;
                        if px < image.width() && py < image.height() {
                            image.put_pixel(px, py, image::Rgba([255, 255, 255, 255]));
                        }
                    }
                }
            }
        }
    }
}

fn draw_changed_region(image: &mut image::RgbaImage, x: u32, y: u32, width: u32, height: u32) {
    if width == 0 || height == 0 {
        return;
    }
    let right = (x + width - 1).min(image.width().saturating_sub(1));
    let bottom = (y + height - 1).min(image.height().saturating_sub(1));
    // A light tint keeps the content legible; the solid edge defines its bounds.
    for py in y..=bottom {
        for px in x..=right {
            let pixel = image.get_pixel_mut(px, py);
            for (channel, tint) in pixel.0[..3].iter_mut().zip([239u16, 68, 68]) {
                *channel = ((*channel as u16 * 7 + tint) / 8) as u8;
            }
        }
    }
    for thickness in 0..2 {
        let left = x.saturating_sub(thickness);
        let top = y.saturating_sub(thickness);
        let r = (right + thickness).min(image.width().saturating_sub(1));
        let b = (bottom + thickness).min(image.height().saturating_sub(1));
        for px in left..=r {
            image.put_pixel(px, top, image::Rgba([239, 68, 68, 255]));
            image.put_pixel(px, b, image::Rgba([239, 68, 68, 255]));
        }
        for py in top..=b {
            image.put_pixel(left, py, image::Rgba([239, 68, 68, 255]));
            image.put_pixel(r, py, image::Rgba([239, 68, 68, 255]));
        }
    }
}

fn write_contact_sheet(path: &Path, frames: &[ContactSheetFrame]) -> Result<(), String> {
    let first = frames
        .first()
        .ok_or("No frames selected for contact sheet")?;
    let first_image = image::load_from_memory(&first.image_data)
        .map_err(|e| format!("Failed to decode contact sheet frame: {}", e))?;
    let cell_height = ((CONTACT_SHEET_CELL_WIDTH as f64 * first_image.height() as f64
        / first_image.width().max(1) as f64)
        .round() as u32)
        .max(1);
    let columns = CONTACT_SHEET_COLUMNS.min(frames.len() as u32).max(1);
    let rows = (frames.len() as u32).div_ceil(columns);
    let canvas_width = CONTACT_SHEET_GAP + columns * (CONTACT_SHEET_CELL_WIDTH + CONTACT_SHEET_GAP);
    let canvas_height =
        CONTACT_SHEET_GAP + rows * (CONTACT_SHEET_LABEL_HEIGHT + cell_height + CONTACT_SHEET_GAP);
    let mut canvas =
        image::RgbaImage::from_pixel(canvas_width, canvas_height, image::Rgba([17, 24, 39, 255]));

    let mut previous_source: Option<image::RgbImage> = None;
    for (index, frame) in frames.iter().enumerate() {
        let source = image::load_from_memory(&frame.image_data)
            .map_err(|e| format!("Failed to decode contact sheet frame: {}", e))?;
        let source_rgb = source.to_rgb8();
        let changed_regions = previous_source
            .as_ref()
            .map(|previous| changed_pixel_regions(previous, &source_rgb).1)
            .unwrap_or_default();
        let mut display_rgb = source_rgb.clone();
        let cursor = scale_cursor_dimensions(
            frame.cursor,
            frame.device_width,
            frame.device_height,
            display_rgb.width(),
            display_rgb.height(),
        );
        composite_cursor(&mut display_rgb, cursor);
        let rendered = image::DynamicImage::ImageRgb8(display_rgb)
            .resize(
                CONTACT_SHEET_CELL_WIDTH,
                cell_height,
                image::imageops::FilterType::Triangle,
            )
            .to_rgba8();
        let column = index as u32 % columns;
        let row = index as u32 / columns;
        let cell_x = CONTACT_SHEET_GAP + column * (CONTACT_SHEET_CELL_WIDTH + CONTACT_SHEET_GAP);
        let label_y = CONTACT_SHEET_GAP
            + row * (CONTACT_SHEET_LABEL_HEIGHT + cell_height + CONTACT_SHEET_GAP);
        let image_y = label_y + CONTACT_SHEET_LABEL_HEIGHT;
        let image_x = cell_x + (CONTACT_SHEET_CELL_WIDTH - rendered.width()) / 2;
        image::imageops::overlay(&mut canvas, &rendered, image_x.into(), image_y.into());
        draw_timestamp(
            &mut canvas,
            cell_x + 4,
            label_y + 4,
            &format_contact_timestamp(frame.elapsed_ms),
        );

        for [rx, ry, rw, rh] in &changed_regions {
            draw_changed_region(
                &mut canvas,
                image_x + (rx * rendered.width() as f32).round() as u32,
                image_y + (ry * rendered.height() as f32).round() as u32,
                (rw * rendered.width() as f32).round().max(1.0) as u32,
                (rh * rendered.height() as f32).round().max(1.0) as u32,
            );
        }
        previous_source = Some(source_rgb);
    }

    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("Failed to create contact sheet directory: {}", e))?;
    }
    canvas
        .save(path)
        .map_err(|e| format!("Failed to save contact sheet: {}", e))
}

/// Attach the recorder's own flattened session to the target behind
/// `page_session_id`, publishing it in `shared` before the command is sent so
/// the resulting `Target.attachedToTarget` is recognised as the recorder's.
pub async fn attach_capture_session(
    client: &CdpClient,
    page_session_id: &str,
    shared: &SharedCaptureSession,
) -> Result<String, String> {
    let info = client
        .send_command_no_params("Target.getTargetInfo", Some(page_session_id))
        .await?;
    let target_id = info
        .get("targetInfo")
        .and_then(|t| t.get("targetId"))
        .and_then(Value::as_str)
        .ok_or("Failed to resolve recording target")?
        .to_string();

    if let Ok(mut guard) = shared.lock() {
        *guard = Some(CaptureSession {
            target_id: target_id.clone(),
            session_id: None,
        });
    }

    let attached: Result<AttachToTargetResult, String> = client
        .send_command_typed(
            "Target.attachToTarget",
            &AttachToTargetParams {
                target_id,
                flatten: true,
            },
            None,
        )
        .await;

    match attached {
        Ok(result) => {
            if let Ok(mut guard) = shared.lock() {
                if let Some(capture) = guard.as_mut() {
                    capture.session_id = Some(result.session_id.clone());
                }
            }
            Ok(result.session_id)
        }
        Err(e) => {
            if let Ok(mut guard) = shared.lock() {
                *guard = None;
            }
            Err(format!("Failed to attach recording session: {}", e))
        }
    }
}

#[derive(Debug)]
struct CapturedVideoFrame {
    path: std::path::PathBuf,
    elapsed: Duration,
    timestamp: f64,
    device_width: f64,
    device_height: f64,
}

#[derive(Debug)]
struct CapturedRecording {
    frames: Vec<CapturedVideoFrame>,
    duration: Duration,
}

/// Capture clean PNG frames, then build the video and contact sheet separately.
#[allow(clippy::too_many_arguments)]
pub fn spawn_recording_task(
    client: Arc<CdpClient>,
    capture_session: String,
    output_path: String,
    fps: u32,
    shared_count: Arc<AtomicU64>,
    shared_captured: Arc<AtomicU64>,
    cursor: bool,
    shared_cursor: SharedRecordingCursor,
    contact_sheet_path: Option<String>,
    contact_sheet_threshold: f64,
    shared_contact_sheet_count: Arc<AtomicU64>,
    cancel_rx: oneshot::Receiver<()>,
) -> tokio::task::JoinHandle<Result<(), String>> {
    tokio::spawn(async move {
        let fps = validate_fps(fps)?;
        let events = client.subscribe_session(&capture_session);
        let capture_dir = tempfile::tempdir()
            .map_err(|e| format!("Failed to create recording frame directory: {}", e))?;

        let started = client
            .send_command(
                "Page.startScreencast",
                Some(json!({
                    "format": "png",
                    // Always 1: Chrome skips frames by count, and a static
                    // page produces exactly one, which a higher value would
                    // drop, leaving nothing to record.
                    "everyNthFrame": 1,
                })),
                Some(&capture_session),
            )
            .await;

        let capture = match started {
            Ok(_) => {
                collect_frames(
                    &client,
                    &capture_session,
                    events,
                    capture_dir.path(),
                    &shared_captured,
                    cancel_rx,
                )
                .await
            }
            Err(e) => Err(format!("Failed to start screencast: {}", e)),
        };

        client.unsubscribe_session(&capture_session);

        // Best effort: the page may already be closed.
        let _ = tokio::time::timeout(
            TEARDOWN_TIMEOUT,
            client.send_command_no_params("Page.stopScreencast", Some(&capture_session)),
        )
        .await;
        let _ = tokio::time::timeout(
            TEARDOWN_TIMEOUT,
            client.send_command(
                "Target.detachFromTarget",
                Some(json!({ "sessionId": capture_session })),
                None,
            ),
        )
        .await;

        let capture = capture?;
        let history = shared_cursor
            .lock()
            .map(|value| value.clone())
            .unwrap_or_default();
        let written = encode_capture(&capture, &output_path, fps, cursor, &history).await?;
        shared_count.store(written, Ordering::Relaxed);

        if let Some(path) = contact_sheet_path {
            let contact_frames =
                select_contact_frames(&capture.frames, contact_sheet_threshold, cursor, &history)?;
            write_contact_sheet(Path::new(&path), &contact_frames)?;
            shared_contact_sheet_count.store(contact_frames.len() as u64, Ordering::Relaxed);
        }

        Ok(())
    })
}

async fn collect_frames(
    client: &CdpClient,
    capture_session: &str,
    mut events: mpsc::Receiver<super::cdp::types::CdpEvent>,
    directory: &Path,
    shared_captured: &AtomicU64,
    cancel_rx: oneshot::Receiver<()>,
) -> Result<CapturedRecording, String> {
    let mut cancel_rx = std::pin::pin!(cancel_rx);
    let started = tokio::time::Instant::now();
    let mut frames = Vec::new();

    loop {
        tokio::select! {
            _ = &mut cancel_rx => break,
            event = events.recv() => {
                let Some(event) = event else { break };
                if event.method == "Page.screencastFrame" {
                    if let Some(sid) = event.params.get("sessionId").and_then(Value::as_i64) {
                        let _ = client
                            .send_command_no_wait(
                                "Page.screencastFrameAck",
                                Some(json!({ "sessionId": sid })),
                                Some(capture_session),
                            )
                            .await;
                    }
                    let decoded = event
                        .params
                        .get("data")
                        .and_then(Value::as_str)
                        .and_then(|data| {
                            base64::Engine::decode(&base64::engine::general_purpose::STANDARD, data)
                                .ok()
                        });
                    if let Some(bytes) = decoded {
                        let index = frames.len();
                        let path = directory.join(format!("frame-{index:08}.png"));
                        tokio::fs::write(&path, bytes).await
                            .map_err(|e| format!("Failed to spool recording frame: {}", e))?;
                        let metadata = &event.params["metadata"];
                        frames.push(CapturedVideoFrame {
                            path,
                            elapsed: started.elapsed(),
                            timestamp: metadata["timestamp"].as_f64().unwrap_or_else(cursor_timestamp),
                            device_width: metadata["deviceWidth"].as_f64().unwrap_or(0.0),
                            device_height: metadata["deviceHeight"].as_f64().unwrap_or(0.0),
                        });
                        shared_captured.fetch_add(1, Ordering::Relaxed);
                    }
                } else if event.method == "Inspector.detached" {
                    // The recorded page was closed; finish the file.
                    break;
                }
            }
        }
    }

    if frames.is_empty() {
        return Err("No frames captured".to_string());
    }
    Ok(CapturedRecording {
        frames,
        duration: started.elapsed(),
    })
}

fn select_contact_frames(
    frames: &[CapturedVideoFrame],
    threshold: f64,
    cursor: bool,
    history: &RecordingCursorHistory,
) -> Result<Vec<ContactSheetFrame>, String> {
    let mut collector = ContactSheetCollector::new(threshold);
    for frame in frames {
        let clean = std::fs::read(&frame.path)
            .map_err(|e| format!("Failed to read contact sheet frame: {}", e))?;
        let cursor_state = if cursor {
            history.at(frame.timestamp)
        } else {
            RecordingCursorState::default()
        };
        collector.consider(
            &clean,
            frame.elapsed,
            cursor_state,
            frame.device_width,
            frame.device_height,
        );
    }
    Ok(collector.finish())
}

fn scale_cursor_dimensions(
    mut cursor: RecordingCursorState,
    device_width: f64,
    device_height: f64,
    width: u32,
    height: u32,
) -> RecordingCursorState {
    if device_width > 0.0 {
        cursor.x *= width as f64 / device_width;
    }
    if device_height > 0.0 {
        cursor.y *= height as f64 / device_height;
    }
    cursor
}

fn scaled_cursor(
    cursor: RecordingCursorState,
    frame: &CapturedVideoFrame,
    width: u32,
    height: u32,
) -> RecordingCursorState {
    scale_cursor_dimensions(
        cursor,
        frame.device_width,
        frame.device_height,
        width,
        height,
    )
}

fn cursor_for_video_frame(
    history: &RecordingCursorHistory,
    page_frame: &CapturedVideoFrame,
    timestamp: f64,
) -> RecordingCursorState {
    let animated = history.interpolated_at(timestamp);
    let anchored = history.at(page_frame.timestamp);
    if animated.buttons != 0 || anchored.buttons != 0 {
        anchored
    } else {
        animated
    }
}

async fn encode_capture(
    capture: &CapturedRecording,
    output_path: &str,
    fps: u32,
    cursor: bool,
    history: &RecordingCursorHistory,
) -> Result<u64, String> {
    let mut command = build_ffmpeg_command(output_path, fps, cursor);
    let mut ffmpeg = command.spawn().map_err(|e| {
        format!(
            "ffmpeg not found or failed to execute: {}. Install ffmpeg to enable recording.",
            e
        )
    })?;
    let mut stdin = ffmpeg
        .stdin
        .take()
        .ok_or_else(|| "Failed to open ffmpeg stdin".to_string())?;
    let total = (capture.duration.as_secs_f64() * fps as f64)
        .ceil()
        .max(1.0) as u64;
    let timeline_origin = capture.frames[0].timestamp - capture.frames[0].elapsed.as_secs_f64();
    let mut page_index = 0usize;
    let mut loaded_index = usize::MAX;
    let mut clean_bytes = Vec::new();
    let mut clean_rgb: Option<image::RgbImage> = None;
    let click_starts = history.click_starts();
    let mut next_click = 0usize;
    let mut active_clicks = VecDeque::new();

    for slot in 0..total {
        let elapsed = Duration::from_secs_f64(slot as f64 / fps as f64);
        let output_timestamp = timeline_origin + elapsed.as_secs_f64();
        while page_index + 1 < capture.frames.len()
            && capture.frames[page_index + 1].elapsed <= elapsed
        {
            page_index += 1;
        }
        let page_frame = &capture.frames[page_index];
        if loaded_index != page_index {
            clean_bytes = std::fs::read(&page_frame.path)
                .map_err(|e| format!("Failed to read recording frame: {}", e))?;
            clean_rgb = cursor
                .then(|| {
                    image::load_from_memory(&clean_bytes)
                        .map(|image| image.to_rgb8())
                        .map_err(|e| format!("Failed to decode recording frame: {}", e))
                })
                .transpose()?;
            loaded_index = page_index;
        }

        if let Some(source) = clean_rgb.as_ref() {
            let state = cursor_for_video_frame(history, page_frame, output_timestamp);
            let state = scaled_cursor(state, page_frame, source.width(), source.height());
            let mut rendered = source.clone();
            while next_click < click_starts.len() && click_starts[next_click].0 <= output_timestamp
            {
                active_clicks.push_back(click_starts[next_click]);
                next_click += 1;
            }
            while active_clicks
                .front()
                .is_some_and(|(started, _)| output_timestamp - started >= RIPPLE_DURATION_SECS)
            {
                active_clicks.pop_front();
            }
            for &(started, click) in &active_clicks {
                let click = scaled_cursor(click, page_frame, source.width(), source.height());
                composite_ripple(
                    &mut rendered,
                    click.x,
                    click.y,
                    (output_timestamp - started) / RIPPLE_DURATION_SECS,
                );
            }
            composite_cursor(&mut rendered, state);
            let header = format!("P6\n{} {}\n255\n", rendered.width(), rendered.height());
            stdin
                .write_all(header.as_bytes())
                .await
                .map_err(|e| format!("ffmpeg write failed: {}", e))?;
            stdin
                .write_all(rendered.as_raw())
                .await
                .map_err(|e| format!("ffmpeg write failed: {}", e))?;
        } else {
            stdin
                .write_all(&clean_bytes)
                .await
                .map_err(|e| format!("ffmpeg write failed: {}", e))?;
        }
    }
    drop(stdin);

    let output = ffmpeg
        .wait_with_output()
        .await
        .map_err(|e| format!("ffmpeg wait failed: {}", e))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!(
            "ffmpeg failed: {}",
            stderr.chars().take(300).collect::<String>()
        ));
    }
    Ok(total)
}

pub async fn stop_recording_task(state: &mut RecordingState) -> Result<(), String> {
    if let Some(tx) = state.cancel_tx.take() {
        let _ = tx.send(());
    }

    let counter = state.shared_frame_count.take();
    let captured = state.shared_captured_count.take();
    let contact_sheet = state.shared_contact_sheet_count.take();
    let handle = state.capture_task.take();

    let result = if let Some(h) = handle {
        match h.await {
            Ok(Ok(())) => Ok(()),
            Ok(Err(e)) => Err(e),
            Err(e) => Err(format!("Recording task panicked: {}", e)),
        }
    } else {
        Ok(())
    };

    if let Some(c) = counter {
        state.frame_count = c.load(Ordering::Relaxed);
    }
    if let Some(c) = captured {
        state.captured_count = c.load(Ordering::Relaxed);
    }
    if let Some(c) = contact_sheet {
        state.contact_sheet_frame_count = c.load(Ordering::Relaxed);
    }
    if let Ok(mut guard) = state.capture_session.lock() {
        *guard = None;
    }

    result
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cursor_history_matches_capture_not_arrival() {
        let mut history = RecordingCursorHistory::default();
        for (time, x, buttons) in [(10.0, 100.0, 0), (11.0, 200.0, 1), (12.0, 300.0, 0)] {
            history.samples.push_back((
                time,
                RecordingCursorState {
                    x,
                    y: 50.0,
                    buttons,
                    visible: true,
                },
            ));
        }
        assert!(!history.at(9.0).visible);
        assert_eq!(history.at(11.5).x, 200.0);
        assert_eq!(history.at(11.5).buttons, 1);
        assert_eq!(history.at(12.0).x, 300.0);
        assert!(!history.at(f64::NAN).visible);
        history.samples.push_back((
            14.0,
            RecordingCursorState {
                x: 500.0,
                y: 150.0,
                buttons: 0,
                visible: true,
            },
        ));
        let interpolated = history.interpolated_at(13.0);
        assert_eq!((interpolated.x, interpolated.y), (400.0, 100.0));
    }

    #[test]
    fn test_cursor_interpolation_stops_while_button_is_down() {
        let mut history = RecordingCursorHistory::default();
        history.record_at(10.0, 10.0, 20.0, 0);
        history.record_at(11.0, 110.0, 120.0, 0);
        history.record_at(12.0, 210.0, 220.0, 1);

        let moving = history.interpolated_at(10.5);
        assert_eq!((moving.x, moving.y), (60.0, 70.0));

        let pressed = history.interpolated_at(11.5);
        assert_eq!((pressed.x, pressed.y, pressed.buttons), (110.0, 120.0, 0));
    }

    #[test]
    fn test_click_starts_only_on_button_transitions() {
        let mut history = RecordingCursorHistory::default();
        history.record_at(1.0, 10.0, 20.0, 0);
        history.record_at(2.0, 10.0, 20.0, 1);
        history.record_at(3.0, 20.0, 30.0, 1);
        history.record_at(4.0, 20.0, 30.0, 0);
        history.record_at(5.0, 30.0, 40.0, 1);

        let clicks = history.click_starts();
        assert_eq!(clicks.len(), 2);
        assert_eq!((clicks[0].0, clicks[0].1.x), (2.0, 10.0));
        assert_eq!((clicks[1].0, clicks[1].1.x), (5.0, 30.0));
    }

    #[test]
    fn test_pressed_cursor_uses_position_for_page_frame() {
        let mut history = RecordingCursorHistory::default();
        history.record_at(10.0, 10.0, 20.0, 0);
        history.record_at(11.0, 110.0, 120.0, 1);
        history.record_at(12.0, 210.0, 220.0, 1);
        history.record_at(13.0, 310.0, 320.0, 0);
        history.record_at(14.0, 410.0, 420.0, 0);
        let frame = CapturedVideoFrame {
            path: std::path::PathBuf::new(),
            elapsed: Duration::ZERO,
            timestamp: 11.0,
            device_width: 1000.0,
            device_height: 500.0,
        };

        let pressed = cursor_for_video_frame(&history, &frame, 12.5);
        assert_eq!((pressed.x, pressed.y, pressed.buttons), (110.0, 120.0, 1));

        let released_frame = CapturedVideoFrame {
            timestamp: 13.0,
            ..frame
        };
        let moving = cursor_for_video_frame(&history, &released_frame, 13.5);
        assert_eq!((moving.x, moving.y, moving.buttons), (360.0, 370.0, 0));
    }

    #[test]
    fn test_cursor_scales_from_page_to_screencast_pixels() {
        let frame = CapturedVideoFrame {
            path: std::path::PathBuf::new(),
            elapsed: Duration::ZERO,
            timestamp: 0.0,
            device_width: 640.0,
            device_height: 360.0,
        };
        let scaled = scaled_cursor(
            RecordingCursorState {
                x: 320.0,
                y: 180.0,
                buttons: 0,
                visible: true,
            },
            &frame,
            1280,
            720,
        );
        assert_eq!((scaled.x, scaled.y), (640.0, 360.0));
    }

    #[test]
    fn test_contact_sheet_selection_keeps_clean_source_pixels() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("clean.png");
        let sheet_path = directory.path().join("sheet.png");
        let clean = image::RgbImage::from_pixel(80, 80, image::Rgb([240, 240, 240]));
        clean.save(&path).unwrap();
        let original = std::fs::read(&path).unwrap();
        let captured: Vec<_> = (0..3)
            .map(|index| CapturedVideoFrame {
                path: path.clone(),
                elapsed: Duration::from_secs(index),
                timestamp: index as f64 + 1.0,
                device_width: 80.0,
                device_height: 80.0,
            })
            .collect();
        let mut history = RecordingCursorHistory::default();
        history.record_at(0.5, 10.0, 20.0, 0);
        history.record_at(1.5, 20.0, 30.0, 0);
        history.record_at(2.5, 30.0, 40.0, 0);

        let selected = select_contact_frames(&captured, 0.05, true, &history).unwrap();
        assert_eq!(selected.len(), 2);
        assert_eq!(selected[0].elapsed_ms, 0);
        assert_eq!(selected[1].elapsed_ms, 2000);
        assert_eq!(selected[0].image_data, original);
        assert_eq!(selected[0].cursor.x, 10.0);
        assert_eq!(selected[1].cursor.x, 30.0);

        write_contact_sheet(&sheet_path, &selected).unwrap();
        let sheet = image::open(sheet_path).unwrap().to_rgb8();
        assert_ne!(sheet.get_pixel(88, 192), &image::Rgb([240, 240, 240]));
    }

    #[test]
    fn test_cursor_hotspot_stays_at_input_when_pressed() {
        let frame = image::RgbImage::from_pixel(100, 100, image::Rgb([255, 255, 255]));
        for buttons in [0, 1] {
            let mut output = frame.clone();
            composite_cursor(
                &mut output,
                RecordingCursorState {
                    x: 40.0,
                    y: 30.0,
                    buttons,
                    visible: true,
                },
            );
            assert!(output.get_pixel(40, 31)[0] < 180);
            assert_eq!(output.get_pixel(10, 10), frame.get_pixel(10, 10));
        }
    }

    #[test]
    fn test_cursor_and_ripple_edges_are_antialiased() {
        let mut frame = image::RgbImage::from_pixel(100, 100, image::Rgb([255, 255, 255]));
        composite_ripple(&mut frame, 40.0, 30.0, 0.5);
        composite_cursor(
            &mut frame,
            RecordingCursorState {
                x: 40.0,
                y: 30.0,
                buttons: 0,
                visible: true,
            },
        );
        assert!(frame
            .pixels()
            .any(|pixel| { pixel.0.iter().any(|&channel| channel > 0 && channel < 255) }));
        assert_ne!(frame.get_pixel(40, 42), &image::Rgb([255, 255, 255]));
        assert_eq!(frame.get_pixel(5, 5), &image::Rgb([255, 255, 255]));
    }

    fn options(fps: Option<u32>) -> RecordingOptions {
        RecordingOptions {
            fps,
            ..RecordingOptions::default()
        }
    }

    #[test]
    fn test_cursor_video_transport_preserves_decoded_pixels() {
        let source =
            image::RgbImage::from_fn(64, 64, |x, y| image::Rgb([x as u8 * 3, y as u8 * 3, 77]));
        let mut png = Vec::new();
        image::DynamicImage::ImageRgb8(source.clone())
            .write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)
            .unwrap();
        let decoded = image::load_from_memory(&png).unwrap().to_rgb8();
        let header = b"P6\n64 64\n255\n";
        let plain = cursor_video_frame(&png, RecordingCursorState::default()).unwrap();
        assert_eq!(&plain[..header.len()], header);
        assert_eq!(&plain[header.len()..], decoded.as_raw());
        let composited = cursor_video_frame(
            &png,
            RecordingCursorState {
                x: 30.0,
                y: 20.0,
                buttons: 1,
                visible: true,
            },
        )
        .unwrap();
        for y in 0..64 {
            for x in 0..64 {
                if !(26..54).contains(&x) || !(16..46).contains(&y) {
                    let offset = header.len() + (y * 64 + x) * 3;
                    assert_eq!(&composited[offset..offset + 3], &plain[offset..offset + 3]);
                }
            }
        }
        let cmd = build_ffmpeg_command("/tmp/out.webm", 30, true);
        assert!(cmd.as_std().get_args().any(|arg| arg == "ppm"));
    }

    #[test]
    fn test_recording_state_new() {
        let state = RecordingState::new();
        assert!(!state.active);
        assert!(state.output_path.is_empty());
        assert_eq!(state.frame_count, 0);
        assert_eq!(state.fps, DEFAULT_FPS);
    }

    #[test]
    fn test_recording_start_sets_active() {
        let mut state = RecordingState::new();
        let result = recording_start(&mut state, "/tmp/test.mp4", options(None));
        assert!(result.is_ok());
        assert!(state.active);
        assert_eq!(state.output_path, "/tmp/test.mp4");
        assert_eq!(state.frame_count, 0);
        assert_eq!(state.fps, 30);
        assert_eq!(result.unwrap()["fps"], 30);
    }

    #[test]
    fn test_recording_start_honors_requested_fps() {
        let mut state = RecordingState::new();
        let result = recording_start(&mut state, "/tmp/test.webm", options(Some(60))).unwrap();
        assert_eq!(state.fps, 60);
        assert_eq!(result["fps"], 60);
    }

    #[test]
    fn test_recording_start_sets_cursor_and_contact_sheet_options() {
        let mut state = RecordingState::new();
        let result = recording_start(
            &mut state,
            "/tmp/demo.webm",
            RecordingOptions {
                cursor: true,
                contact_sheet: true,
                contact_sheet_threshold: 0.12,
                ..RecordingOptions::default()
            },
        )
        .unwrap();
        assert!(state.cursor);
        assert!(state.contact_sheet);
        assert_eq!(state.contact_sheet_threshold, 0.12);
        assert_eq!(
            state.contact_sheet_path.as_deref(),
            Some("/tmp/demo.contact-sheet.png")
        );
        assert_eq!(result["contactSheetPath"], "/tmp/demo.contact-sheet.png");
    }

    #[test]
    fn test_contact_sheet_path_replaces_extension() {
        assert_eq!(contact_sheet_path("demo.webm"), "demo.contact-sheet.png");
        assert_eq!(
            contact_sheet_path("artifacts/demo.capture.webm"),
            "artifacts/demo.capture.contact-sheet.png"
        );
    }

    #[test]
    fn test_validate_contact_sheet_threshold_range() {
        assert_eq!(validate_contact_sheet_threshold(0.0).unwrap(), 0.0);
        assert_eq!(validate_contact_sheet_threshold(1.0).unwrap(), 1.0);
        assert!(validate_contact_sheet_threshold(-0.01).is_err());
        assert!(validate_contact_sheet_threshold(1.01).is_err());
        assert!(validate_contact_sheet_threshold(f64::NAN).is_err());
    }

    #[test]
    fn test_changed_pixel_regions_reports_separate_bounds() {
        let before = image::RgbImage::from_pixel(64, 64, image::Rgb([0, 0, 0]));
        let mut after = before.clone();
        for y in 8..16 {
            for x in 8..16 {
                after.put_pixel(x, y, image::Rgb([255, 255, 255]));
            }
        }
        for y in 40..48 {
            for x in 48..56 {
                after.put_pixel(x, y, image::Rgb([255, 255, 255]));
            }
        }
        let (ratio, regions) = changed_pixel_regions(&before, &after);
        assert!((ratio - 0.03125).abs() < f64::EPSILON);
        assert_eq!(
            regions,
            vec![[0.0, 0.0, 0.375, 0.375], [0.625, 0.5, 0.375, 0.375]]
        );
    }

    #[test]
    fn test_contact_sheet_covers_every_flyout_control_and_distant_region() {
        let before = image::RgbImage::from_pixel(1024, 512, image::Rgb([255, 255, 255]));
        let mut after = before.clone();
        let mut changed_points = Vec::new();
        // Rows in a newly opened flyout, plus more than eight remote changes.
        for (x, y) in (0..6)
            .map(|row| (16, 16 + row * 32))
            .chain((0..12).map(|i| (256 + (i % 6) * 112, 32 + (i / 6) * 200)))
        {
            for py in y..y + 8 {
                for px in x..x + 16 {
                    after.put_pixel(px, py, image::Rgb([0, 0, 0]));
                    changed_points.push((px, py));
                }
            }
        }
        let (_, regions) = changed_pixel_regions(&before, &after);
        assert_eq!(regions.len(), 13, "flyout rows should form one region");
        for (x, y) in changed_points {
            assert!(
                regions.iter().any(|[rx, ry, rw, rh]| {
                    let x = x as f32 / 1024.0;
                    let y = y as f32 / 512.0;
                    x >= *rx && x < rx + rw && y >= *ry && y < ry + rh
                }),
                "uncovered change at {x},{y}"
            );
        }
    }

    #[test]
    fn test_contact_sheet_finish_includes_latest_frame() {
        let mut collector = ContactSheetCollector::new(0.05);
        collector.selected.push(ContactSheetFrame {
            image_data: vec![1],
            elapsed_ms: 10,
            cursor: RecordingCursorState::default(),
            device_width: 0.0,
            device_height: 0.0,
        });
        collector.latest = Some(ContactSheetFrame {
            image_data: vec![2],
            elapsed_ms: 20,
            cursor: RecordingCursorState::default(),
            device_width: 0.0,
            device_height: 0.0,
        });

        let frames = collector.finish();
        assert_eq!(frames.len(), 2);
        assert_eq!(frames.last().unwrap().image_data, vec![2]);
    }

    #[test]
    fn test_recording_start_rejects_out_of_range_fps() {
        let mut state = RecordingState::new();
        let too_high = recording_start(&mut state, "/tmp/test.webm", options(Some(61)));
        assert!(too_high.unwrap_err().contains("valid range: 1-60"));
        assert!(!state.active);

        let zero = recording_start(&mut state, "/tmp/test.webm", options(Some(0)));
        assert!(zero.is_err());
        assert!(!state.active);
    }

    #[test]
    fn test_recording_start_while_active() {
        let mut state = RecordingState::new();
        recording_start(&mut state, "/tmp/test1.mp4", options(None)).unwrap();
        let result = recording_start(&mut state, "/tmp/test2.mp4", options(None));
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("already active"));
    }

    #[test]
    fn test_recording_stop_not_active() {
        let mut state = RecordingState::new();
        let result = recording_stop(&mut state);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("No recording"));
    }

    #[test]
    fn test_recording_stop_no_frames() {
        let mut state = RecordingState::new();
        recording_start(&mut state, "/tmp/test.mp4", options(None)).unwrap();
        let result = recording_stop(&mut state);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("No frames"));
        assert!(!state.active);
    }

    #[test]
    fn test_recording_stop_reports_fps() {
        let mut state = RecordingState::new();
        recording_start(&mut state, "/tmp/test.webm", options(Some(60))).unwrap();
        state.frame_count = 120;
        let result = recording_stop(&mut state).unwrap();
        assert_eq!(result["frames"], 120);
        assert_eq!(result["fps"], 60);
    }

    #[test]
    fn test_capture_session_matches_by_target_while_attach_is_in_flight() {
        let shared: SharedCaptureSession = Arc::new(Mutex::new(Some(CaptureSession {
            target_id: "T-REC".into(),
            session_id: None,
        })));
        assert!(owns_attachment(&shared, "T-REC", "S-ANY"));
        assert!(!owns_attachment(&shared, "T-OTHER", "S-ANY"));
    }

    #[test]
    fn test_capture_session_matches_by_session_once_attached() {
        let shared: SharedCaptureSession = Arc::new(Mutex::new(Some(CaptureSession {
            target_id: "T-REC".into(),
            session_id: Some("S-REC".into()),
        })));
        assert!(owns_attachment(&shared, "T-REC", "S-REC"));
        // A later attachment to the same target (e.g. the daemon's own) is
        // not the recorder's.
        assert!(!owns_attachment(&shared, "T-REC", "S-OTHER"));
        assert!(!owns_attachment(
            &Arc::new(Mutex::new(None)),
            "T-REC",
            "S-REC"
        ));
    }

    #[test]
    fn test_validate_fps_range() {
        assert_eq!(validate_fps(1).unwrap(), 1);
        assert_eq!(validate_fps(DEFAULT_FPS).unwrap(), 30);
        assert_eq!(validate_fps(MAX_FPS).unwrap(), 60);
        assert!(validate_fps(0).is_err());
        assert!(validate_fps(MAX_FPS + 1).is_err());
    }

    #[test]
    fn test_build_ffmpeg_command_webm() {
        let cmd = build_ffmpeg_command("/tmp/out.webm", DEFAULT_FPS, false);
        let args: Vec<&std::ffi::OsStr> = cmd.as_std().get_args().collect();
        let args_str: Vec<&str> = args.iter().filter_map(|a| a.to_str()).collect();
        assert!(args_str.contains(&"libvpx"));
        assert!(args_str.contains(&"/tmp/out.webm"));
        assert!(args_str.contains(&"8000k"));
        assert!(args_str.contains(&"18"));
        assert!(args_str.contains(&"png"));
    }

    #[test]
    fn test_build_ffmpeg_command_mp4() {
        let cmd = build_ffmpeg_command("/tmp/out.mp4", DEFAULT_FPS, false);
        let args: Vec<&std::ffi::OsStr> = cmd.as_std().get_args().collect();
        let args_str: Vec<&str> = args.iter().filter_map(|a| a.to_str()).collect();
        assert!(args_str.contains(&"libx264"));
        assert!(args_str.contains(&"/tmp/out.mp4"));
    }

    #[test]
    fn test_build_ffmpeg_command_passes_framerate() {
        let cmd = build_ffmpeg_command("/tmp/out.webm", 60, false);
        let args: Vec<String> = cmd
            .as_std()
            .get_args()
            .filter_map(|a| a.to_str().map(String::from))
            .collect();
        let framerate = args
            .iter()
            .position(|a| a == "-framerate")
            .and_then(|i| args.get(i + 1))
            .map(String::as_str);
        assert_eq!(framerate, Some("60"));
        assert!(args.iter().any(|a| a == "8000k"));
        let threads = args
            .iter()
            .position(|a| a == "-threads")
            .and_then(|i| args.get(i + 1))
            .map(String::as_str);
        assert_eq!(threads, Some("2"));
    }

    #[test]
    fn test_build_ffmpeg_command_single_thread_at_default_fps() {
        let cmd = build_ffmpeg_command("/tmp/out.mp4", DEFAULT_FPS, false);
        let args: Vec<String> = cmd
            .as_std()
            .get_args()
            .filter_map(|a| a.to_str().map(String::from))
            .collect();
        let threads = args
            .iter()
            .position(|a| a == "-threads")
            .and_then(|i| args.get(i + 1))
            .map(String::as_str);
        assert_eq!(threads, Some("1"));
    }
}

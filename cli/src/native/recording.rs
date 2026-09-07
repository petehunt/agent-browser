use serde_json::{json, Value};
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

pub struct RecordingState {
    pub active: bool,
    pub output_path: String,
    /// Capture rate for the active (or most recent) recording.
    pub fps: u32,
    /// Frames written to the file, including frames held through gaps.
    pub frame_count: u64,
    /// Distinct frames received from the screencast.
    pub captured_count: u64,
    pub capture_task: Option<tokio::task::JoinHandle<Result<(), String>>>,
    pub shared_frame_count: Option<Arc<AtomicU64>>,
    pub shared_captured_count: Option<Arc<AtomicU64>>,
    pub cancel_tx: Option<oneshot::Sender<()>>,
    /// Shared with the daemon's event handlers.
    pub capture_session: SharedCaptureSession,
}

impl RecordingState {
    pub fn new() -> Self {
        Self {
            active: false,
            output_path: String::new(),
            fps: DEFAULT_FPS,
            frame_count: 0,
            captured_count: 0,
            capture_task: None,
            shared_frame_count: None,
            shared_captured_count: None,
            cancel_tx: None,
            capture_session: Arc::new(Mutex::new(None)),
        }
    }
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
    fps: Option<u32>,
) -> Result<Value, String> {
    if state.active {
        return Err("Recording already active".to_string());
    }

    let fps = validate_fps(fps.unwrap_or(DEFAULT_FPS))?;

    state.active = true;
    state.output_path = path.to_string();
    state.fps = fps;
    state.frame_count = 0;
    state.captured_count = 0;

    Ok(json!({ "started": true, "path": path, "fps": fps }))
}

pub fn recording_stop(state: &mut RecordingState) -> Result<Value, String> {
    if !state.active {
        return Err("No recording in progress".to_string());
    }

    state.active = false;

    if state.frame_count == 0 {
        return Err("No frames captured".to_string());
    }

    Ok(json!({
        "path": &state.output_path,
        "frames": state.frame_count,
        "capturedFrames": state.captured_count,
        "fps": state.fps,
    }))
}

fn build_ffmpeg_command(output_path: &str, fps: u32) -> tokio::process::Command {
    let mut cmd = tokio::process::Command::new("ffmpeg");
    let high_fps = fps > HIGH_FPS_THRESHOLD;

    cmd.args(["-y", "-loglevel", "error"])
        .args(["-avioflags", "direct"])
        .args([
            "-fpsprobesize",
            "0",
            "-probesize",
            "32",
            "-analyzeduration",
            "0",
        ])
        .args(["-f", "image2pipe", "-c:v", "png", "-framerate"])
        .arg(fps.to_string())
        .args(["-i", "pipe:0"])
        .args(["-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2"]);

    if output_path.ends_with(".webm") {
        cmd.args(["-c:v", "libvpx", "-crf", "18"])
            .args(["-b:v", &format!("{}k", WEBM_BITRATE_KBPS)]);
    } else {
        cmd.args(["-c:v", "libx264", "-preset", "ultrafast"]);
    }

    // One encoder thread is enough at ordinary rates; 60 fps benefits from a
    // second thread during deferred encoding.
    cmd.args(["-pix_fmt", "yuv420p"])
        .args(["-threads", if high_fps { "2" } else { "1" }])
        .arg(output_path)
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .kill_on_drop(true);

    cmd
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
}

#[derive(Debug)]
struct CapturedRecording {
    frames: Vec<CapturedVideoFrame>,
    duration: Duration,
}

/// Capture lossless frames first, then encode them after the take ends.
#[allow(clippy::too_many_arguments)]
pub fn spawn_recording_task(
    client: Arc<CdpClient>,
    capture_session: String,
    output_path: String,
    fps: u32,
    shared_count: Arc<AtomicU64>,
    shared_captured: Arc<AtomicU64>,
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
        let written = encode_capture(&capture, &output_path, fps).await?;
        shared_count.store(written, Ordering::Relaxed);
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
                        frames.push(CapturedVideoFrame {
                            path,
                            elapsed: started.elapsed(),
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

async fn encode_capture(
    capture: &CapturedRecording,
    output_path: &str,
    fps: u32,
) -> Result<u64, String> {
    let mut command = build_ffmpeg_command(output_path, fps);
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
    let mut page_index = 0usize;
    let mut loaded_index = usize::MAX;
    let mut clean_bytes = Vec::new();

    for slot in 0..total {
        let elapsed = Duration::from_secs_f64(slot as f64 / fps as f64);
        while page_index + 1 < capture.frames.len()
            && capture.frames[page_index + 1].elapsed <= elapsed
        {
            page_index += 1;
        }
        if loaded_index != page_index {
            clean_bytes = std::fs::read(&capture.frames[page_index].path)
                .map_err(|e| format!("Failed to read recording frame: {}", e))?;
            loaded_index = page_index;
        }
        stdin
            .write_all(&clean_bytes)
            .await
            .map_err(|e| format!("ffmpeg write failed: {}", e))?;
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
    if let Ok(mut guard) = state.capture_session.lock() {
        *guard = None;
    }

    result
}
#[cfg(test)]
mod tests {
    use super::*;

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
        let result = recording_start(&mut state, "/tmp/test.mp4", None);
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
        let result = recording_start(&mut state, "/tmp/test.webm", Some(60)).unwrap();
        assert_eq!(state.fps, 60);
        assert_eq!(result["fps"], 60);
    }

    #[test]
    fn test_recording_start_rejects_out_of_range_fps() {
        let mut state = RecordingState::new();
        let too_high = recording_start(&mut state, "/tmp/test.webm", Some(61));
        assert!(too_high.unwrap_err().contains("valid range: 1-60"));
        assert!(!state.active);

        let zero = recording_start(&mut state, "/tmp/test.webm", Some(0));
        assert!(zero.is_err());
        assert!(!state.active);
    }

    #[test]
    fn test_recording_start_while_active() {
        let mut state = RecordingState::new();
        recording_start(&mut state, "/tmp/test1.mp4", None).unwrap();
        let result = recording_start(&mut state, "/tmp/test2.mp4", None);
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
        recording_start(&mut state, "/tmp/test.mp4", None).unwrap();
        let result = recording_stop(&mut state);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("No frames"));
        assert!(!state.active);
    }

    #[test]
    fn test_recording_stop_reports_fps() {
        let mut state = RecordingState::new();
        recording_start(&mut state, "/tmp/test.webm", Some(60)).unwrap();
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
        let cmd = build_ffmpeg_command("/tmp/out.webm", DEFAULT_FPS);
        let args: Vec<&std::ffi::OsStr> = cmd.as_std().get_args().collect();
        let args_str: Vec<&str> = args.iter().filter_map(|a| a.to_str()).collect();
        assert!(args_str.contains(&"libvpx"));
        assert!(args_str.contains(&"/tmp/out.webm"));
        assert!(args_str.contains(&"png"));
        assert!(args_str.contains(&"18"));
        assert!(args_str.contains(&"8000k"));
    }

    #[test]
    fn test_build_ffmpeg_command_mp4() {
        let cmd = build_ffmpeg_command("/tmp/out.mp4", DEFAULT_FPS);
        let args: Vec<&std::ffi::OsStr> = cmd.as_std().get_args().collect();
        let args_str: Vec<&str> = args.iter().filter_map(|a| a.to_str()).collect();
        assert!(args_str.contains(&"libx264"));
        assert!(args_str.contains(&"/tmp/out.mp4"));
    }

    #[test]
    fn test_build_ffmpeg_command_passes_framerate() {
        let cmd = build_ffmpeg_command("/tmp/out.webm", 60);
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
        let cmd = build_ffmpeg_command("/tmp/out.mp4", DEFAULT_FPS);
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

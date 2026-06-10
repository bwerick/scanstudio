import cv2
import time
import math
from pathlib import Path
import numpy as np
from collections import deque

# ---------------------------
# Lightweight vision helpers
# ---------------------------


def variance_of_laplacian(gray: np.ndarray) -> float:
    """Focus metric: higher is sharper."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def apply_tint(
    frame_bgr: np.ndarray, color=(0, 255, 0), strength: float = 0.25
) -> np.ndarray:
    """Apply a color tint overlay. color is BGR."""
    overlay = frame_bgr.copy()
    overlay[:] = color
    return cv2.addWeighted(overlay, strength, frame_bgr, 1.0 - strength, 0)


# ---------------------------
# Spread detection state machine
# ---------------------------


class SpreadDetector:
    """
    Real-time spread detection using frame-to-frame motion.

    Tracks motion in a rolling window and detects when the page is stable
    (low motion = spread captured) vs turning (high motion = page flip).

    States:
      TURNING  → motion is high (page being flipped)
      SETTLING → motion dropping, waiting to confirm stable
      STABLE   → page is stable, spread captured

    Transitions:
      TURNING  → SETTLING  when motion drops below settle_threshold
      SETTLING → STABLE    when motion stays low for min_stable_frames
      SETTLING → TURNING   when motion spikes again (false settle)
      STABLE   → TURNING   when motion rises above turn_threshold
    """

    def __init__(
        self,
        fps=30,
        stable_threshold=2.5,  # motion below this = potentially stable
        turn_threshold=4.0,  # motion above this = definitely turning
        min_stable_frames=12,  # ~0.4s of low motion to confirm stable
        buffer_size=10,
    ):  # rolling buffer for smoothing
        self.fps = fps
        self.stable_threshold = stable_threshold
        self.turn_threshold = turn_threshold
        self.min_stable_frames = min_stable_frames

        self.buffer = deque(maxlen=buffer_size)
        self.prev_gray = None

        self.state = "TURNING"
        self.settle_count = 0  # frames spent in settling
        self.spread_count = 0  # total spreads detected
        self.just_captured = False  # true for one frame after new spread detected
        self.flash_frames = 0  # countdown for green flash

        self.current_motion = 0.0

    def update(self, frame_bgr: np.ndarray, work_size: int = 320) -> dict:
        """
        Feed a new frame. Returns state info dict.

        Args:
            frame_bgr: current frame (full or preview res)
            work_size: resize width for motion computation (smaller = faster)
        """
        h, w = frame_bgr.shape[:2]
        scale = work_size / w if w > work_size else 1.0
        if scale < 1.0:
            small = cv2.resize(
                frame_bgr,
                (int(w * scale), int(h * scale)),
                interpolation=cv2.INTER_AREA,
            )
        else:
            small = frame_bgr

        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        self.just_captured = False

        if self.prev_gray is not None and gray.shape == self.prev_gray.shape:
            diff = cv2.absdiff(self.prev_gray, gray)
            motion = float(np.mean(diff))
            self.buffer.append(motion)
            self.current_motion = np.mean(self.buffer) if self.buffer else motion

            # State machine
            if self.state == "TURNING":
                if self.current_motion < self.stable_threshold:
                    self.state = "SETTLING"
                    self.settle_count = 1

            elif self.state == "SETTLING":
                if self.current_motion >= self.turn_threshold:
                    # False settle — back to turning
                    self.state = "TURNING"
                    self.settle_count = 0
                elif self.current_motion < self.stable_threshold:
                    self.settle_count += 1
                    if self.settle_count >= self.min_stable_frames:
                        self.state = "STABLE"
                        self.spread_count += 1
                        self.just_captured = True
                        self.flash_frames = 8  # flash for ~8 frames
                else:
                    # Motion between thresholds — reset settle counter
                    self.settle_count = 0

            elif self.state == "STABLE":
                if self.current_motion >= self.turn_threshold:
                    self.state = "TURNING"
                    self.settle_count = 0

        self.prev_gray = gray

        # Flash countdown
        show_flash = self.flash_frames > 0
        if self.flash_frames > 0:
            self.flash_frames -= 1

        return {
            "state": self.state,
            "motion": self.current_motion,
            "spread_count": self.spread_count,
            "just_captured": self.just_captured,
            "show_flash": show_flash,
        }


# ---------------------------
# Post-recording analysis
# ---------------------------


def post_recording_analysis(video_path: str):
    """Quick motion analysis on the recorded video."""
    from scipy.ndimage import uniform_filter1d
    from scipy.signal import find_peaks

    print(f"\n{'='*50}")
    print(f"Post-Recording Analysis: {video_path}")
    print(f"{'='*50}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("ERROR: Cannot open recorded video")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps > 0 else 0

    print(
        f"Video: {orig_w}x{orig_h} @ {fps:.1f}fps, {total_frames} frames ({duration:.1f}s)"
    )
    print(f"Computing motion signal...")

    # Compute motion at 360p
    analysis_h = 360
    scale = analysis_h / orig_h
    analysis_w = int(orig_w * scale)

    diffs = []
    prev_gray = None
    frame_idx = 0
    t0 = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        small = cv2.resize(
            frame, (analysis_w, analysis_h), interpolation=cv2.INTER_AREA
        )
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            diffs.append(float(np.mean(cv2.absdiff(prev_gray, gray))))
        prev_gray = gray
        frame_idx += 1
        if frame_idx % 3000 == 0:
            elapsed = time.time() - t0
            pct = frame_idx / total_frames * 100
            print(f"  {frame_idx}/{total_frames} ({pct:.0f}%)...")

    cap.release()
    elapsed = time.time() - t0

    if not diffs:
        print("ERROR: No frames processed")
        return

    diffs = np.array(diffs)
    smoothed = uniform_filter1d(diffs, size=15)

    # Detect peaks
    peaks, _ = find_peaks(smoothed, height=5.0, distance=int(1.5 * fps), prominence=3.0)
    n_spreads = len(peaks) + 1
    estimated_pages = (n_spreads - 2) * 2  # subtract covers, each spread = 2 pages

    # Check for possible corrections (short spreads between peaks)
    spread_durs = []
    boundaries = [0] + list(peaks) + [len(diffs)]
    for i in range(len(boundaries) - 1):
        dur = (boundaries[i + 1] - boundaries[i]) / fps
        spread_durs.append(dur)

    short_spreads = sum(1 for d in spread_durs if d < 1.5)
    long_spreads = sum(1 for d in spread_durs if d > 4.0)

    print(f"\nAnalysis complete ({elapsed:.1f}s)")
    print(f"{'='*50}")
    print(f"  Detected spreads:     {n_spreads}")
    print(f"  Estimated pages:      ~{estimated_pages}")
    print(f"  Median spread time:   {np.median(spread_durs):.2f}s")
    print(
        f"  Short spreads (<1.5s): {short_spreads}  (possible corrections/duplicates)"
    )
    print(f"  Long spreads (>4.0s):  {long_spreads}  (possible missed turns)")
    print(f"  Motion range:         {diffs.min():.1f} — {diffs.max():.1f}")
    print(f"{'='*50}")

    if short_spreads > 0:
        print(
            f"  ⚠  {short_spreads} short spreads detected — you may have page-turn corrections"
        )
    if long_spreads > 3:
        print(f"  ⚠  {long_spreads} long spreads — check for missed page turns")
    if n_spreads < 10:
        print(f"  ⚠  Very few spreads detected — is the video correct?")

    print()


# ---------------------------
# Main app
# ---------------------------


def main():
    # Camera + recording defaults
    camera_index = 0
    backend = cv2.CAP_AVFOUNDATION

    req_w, req_h = 3840, 2160
    req_fps = 30

    # Recording output
    out_dir = Path("recordings")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = None  # set per-recording when R is pressed

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    # Preview settings
    preview_max_w = 1600
    show_center_line = True
    show_safe_frame = False
    show_blur_meter = True
    show_spread_counter = True

    # Subtle center line styling
    center_alpha = 0.18
    center_thickness = 2

    # Safe frame margins
    safe_margin = 0.05

    # Blur thresholds
    blur_thresh = 120.0

    # Open camera
    cap = cv2.VideoCapture(camera_index, backend)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera {camera_index}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(req_w))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(req_h))
    cap.set(cv2.CAP_PROP_FPS, float(req_fps))
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    reported_fps = cap.get(cv2.CAP_PROP_FPS)

    # Video writer
    writer = None
    recording = False
    frames_written = 0
    rec_start = None

    # Spread detector
    spread_detector = SpreadDetector(fps=req_fps)

    # FPS measurement
    last_fps_t = time.perf_counter()
    fps_counter = 0
    ui_fps = 0.0

    window_name = "ScanStudio - Book Flip Recorder"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print("Controls:")
    print("  R  : start/stop recording")
    print("  S  : save a snapshot (clean frame)")
    print("  M  : toggle center line")
    print("  F  : toggle safe frame")
    print("  B  : toggle blur meter")
    print("  C  : toggle spread counter")
    print("  [ ]: decrease/increase blur threshold")
    print("  Q/ESC: quit")
    print()

    snapshot_count = 0

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            continue

        H, W = frame.shape[:2]

        # Preview resize
        if W > preview_max_w:
            scale = preview_max_w / float(W)
            preview = cv2.resize(
                frame, (int(W * scale), int(H * scale)), interpolation=cv2.INTER_AREA
            )
        else:
            preview = frame.copy()

        pH, pW = preview.shape[:2]

        # Blur analysis
        work = preview
        if pW > 1100:
            work = cv2.resize(
                preview,
                (1100, int(pH * (1100 / float(pW)))),
                interpolation=cv2.INTER_AREA,
            )

        work_gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
        focus_score = variance_of_laplacian(work_gray)
        blurry = focus_score < blur_thresh

        # Spread detection (only while recording)
        spread_info = None
        if recording:
            spread_info = spread_detector.update(preview)

            # Green flash when new spread captured
            if spread_info["show_flash"]:
                preview = apply_tint(preview, color=(0, 200, 0), strength=0.15)

        # Overlays
        overlay = preview.copy()

        # Safe frame box
        if show_safe_frame:
            mx = int(safe_margin * pW)
            my = int(safe_margin * pH)
            cv2.rectangle(overlay, (mx, my), (pW - mx, pH - my), (255, 255, 255), 2)

        # Center line
        if show_center_line:
            x = pW // 2
            cv2.line(
                overlay,
                (x, int(0.06 * pH)),
                (x, int(0.94 * pH)),
                (255, 10, 10),
                center_thickness,
            )

        # Blend subtle overlays
        preview = cv2.addWeighted(overlay, center_alpha, preview, 1.0 - center_alpha, 0)

        # HUD text
        y = 28
        line_h = 26

        # Recording status
        if recording:
            rec_elapsed = time.perf_counter() - rec_start if rec_start else 0.0
            cv2.putText(
                preview,
                f"REC  {rec_elapsed:5.1f}s  frames:{frames_written}",
                (18, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )
        else:
            cv2.putText(
                preview,
                "READY (press R to record)",
                (18, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )
        y += line_h

        # Camera info
        cv2.putText(
            preview,
            f"Cam0  {actual_w}x{actual_h}  fps:{reported_fps:.0f}  ui:{ui_fps:.1f}",
            (18, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
        )
        y += line_h

        # Focus/blur
        if show_blur_meter:
            status = "BLURRY" if blurry else "OK"
            color = (0, 100, 255) if blurry else (255, 255, 255)
            cv2.putText(
                preview,
                f"Focus: {focus_score:7.1f}  [{status}]",
                (18, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                color,
                2,
            )
            y += line_h

        # Spread counter and state
        if show_spread_counter and recording and spread_info:
            state = spread_info["state"]
            motion = spread_info["motion"]
            count = spread_info["spread_count"]

            # State color
            if state == "STABLE":
                state_color = (0, 255, 0)  # green
            elif state == "SETTLING":
                state_color = (0, 200, 255)  # yellow
            else:
                state_color = (150, 150, 150)  # gray

            cv2.putText(
                preview,
                f"Spreads: {count}  [{state}]  motion:{motion:.1f}",
                (18, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                state_color,
                2,
            )
            y += line_h

        # FPS
        fps_counter += 1
        now = time.perf_counter()
        if now - last_fps_t >= 0.5:
            ui_fps = fps_counter / (now - last_fps_t)
            fps_counter = 0
            last_fps_t = now

        # Show
        cv2.imshow(window_name, preview)
        key = cv2.waitKey(1) & 0xFF

        # Keys
        if key in (ord("q"), 27):
            break

        elif key in (ord("r"), ord("R")):
            recording = not recording
            if recording:
                # Generate fresh filename for each recording
                out_path = out_dir / f"bookflip_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
                writer = cv2.VideoWriter(
                    str(out_path), fourcc, req_fps, (actual_w, actual_h)
                )
                if not writer.isOpened():
                    recording = False
                    writer = None
                    print("ERROR: VideoWriter failed to open.")
                else:
                    frames_written = 0
                    rec_start = time.perf_counter()
                    spread_detector = SpreadDetector(fps=req_fps)
                    print(f"Recording started: {out_path}")
            else:
                if writer is not None:
                    writer.release()
                    writer = None
                rec_elapsed = time.perf_counter() - rec_start if rec_start else 0
                print(
                    f"Recording stopped. Frames: {frames_written}, "
                    f"Spreads detected: {spread_detector.spread_count}, "
                    f"Duration: {rec_elapsed:.1f}s"
                )
                rec_start = None

                # Offer post-recording analysis
                print(f"\nRun post-recording analysis? (y/n): ", end="", flush=True)
                # We can't easily block for input while cv2 window is open,
                # so we'll use a simple approach: check for 'a' key press
                print("(or press A in the window)")

        elif key in (ord("a"), ord("A")):
            if not recording and out_path is not None and out_path.exists():
                post_recording_analysis(str(out_path))

        elif key in (ord("s"), ord("S")):
            snap_path = (
                out_dir
                / f"snapshot_{time.strftime('%Y%m%d_%H%M%S')}_{snapshot_count:03d}.png"
            )
            cv2.imwrite(str(snap_path), frame)
            snapshot_count += 1
            print(f"Saved snapshot: {snap_path}")

        elif key in (ord("m"), ord("M")):
            show_center_line = not show_center_line

        elif key in (ord("f"), ord("F")):
            show_safe_frame = not show_safe_frame

        elif key in (ord("b"), ord("B")):
            show_blur_meter = not show_blur_meter

        elif key in (ord("c"), ord("C")):
            show_spread_counter = not show_spread_counter

        elif key == ord("["):
            blur_thresh = max(5.0, blur_thresh - 10.0)

        elif key == ord("]"):
            blur_thresh = min(5000.0, blur_thresh + 10.0)

        # Write clean frame
        if recording and writer is not None:
            writer.write(frame)
            frames_written += 1

    # Cleanup
    if writer is not None:
        writer.release()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

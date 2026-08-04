"""
Online page-turn detection — the decision logic behind Phase 0.

Split out of the capture loop so that *when to capture* is independent of where
the frames come from. ``p0_live_capture.py`` drives this from a local
``cv2.VideoCapture``; a browser front end can drive the same detector over a
socket — which is the only way to reach a camera on ChromeOS, where the Linux
container has no ``uvcvideo`` and therefore no ``/dev/video*`` no matter what
the USB sharing toggles say. Both front ends emit identical artifacts because
both run this code, rather than a Python original and a JavaScript port that
quietly drift apart.

Everything here reads the *analysis-resolution grayscale* frame (~360p).
Nothing needs full resolution to decide anything: motion, sharpness and the
settle timer are all measured on the small frame, and a capture merely names a
frame index for the caller to resolve into pixels. That asymmetry is what makes
a remote front end practical — 640x360 gray at 30 fps is ~7 MB/s, while the 4K
frames those decisions select are ~750 MB/s.

The state machine:

    WAITING --still for settle_frames--> SETTLED   (captures: "initial")
    SETTLED --smooth > turn_threshold--> TURNING   (records a peak)
    TURNING --still for settle_frames--> SETTLED   (captures: "settle")

A capture always names the *sharpest* frame in the trailing settle window, not
the frame that happened to trip the threshold: the operator's hand leaves the
frame gradually and the camera needs a moment to refocus, so the last frame of
a settle is rarely the best one in it.
"""

from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.ndimage import uniform_filter1d


WAITING = "WAITING"
SETTLED = "SETTLED"
TURNING = "TURNING"


def laplacian_sharpness(gray):
    """Focus measure: variance of the Laplacian over the frame's centre 80%.

    The margins are excluded because they hold the table, the operator's hands
    and the fanned edge of the page stack — all high-frequency detail that
    stays sharp while the page itself is still blurred or moving, which would
    otherwise flatten the differences between candidate frames.
    """
    h, w = gray.shape
    center = gray[int(h * 0.1):int(h * 0.9), int(w * 0.1):int(w * 0.9)]
    return float(cv2.Laplacian(center, cv2.CV_64F).var())


def build_spreads(peaks, total_len, fps):
    """Spread list in the same shape P2 emits (boundaries between page turns)."""
    if len(peaks) == 0:
        bounds = [(0, total_len)]
    else:
        bounds = [(0, int(peaks[0]))]
        bounds += [(int(peaks[i]), int(peaks[i + 1])) for i in range(len(peaks) - 1)]
        bounds.append((int(peaks[-1]), total_len))
    return [{"spread_index": i + 1, "start_frame": s, "end_frame": e,
             "frame_count": e - s, "duration_sec": round((e - s) / fps, 3),
             "start_time": round(s / fps, 2), "end_time": round(e / fps, 2)}
            for i, (s, e) in enumerate(bounds)]


def keyframe_record(capture, fps, spread_start, filename, source="live"):
    """One keyframes.json entry, in the shape P3 emits.

    Both front ends (the local cv2 loop and the browser server) build their
    entries here, so the file downstream phases read cannot drift between them.
    ``spread_start`` is the previous keyframe's frame_index, or 0 for the first.
    """
    fi = capture.frame_index
    return {
        "frame_index": fi,
        "time_sec": round(fi / fps, 2),
        "motion_value": round(capture.motion, 4),
        "sharpness": round(capture.sharpness, 1),
        "filename": filename,
        "spread_start": spread_start,
        "spread_end": fi,
        "spread_duration": round((fi - spread_start) / fps, 3),
        "source": source,
    }


@dataclass(frozen=True)
class Capture:
    """The detector's choice of which frame to keep, for the caller to resolve.

    ``frame_index`` points into the trailing settle window, so a caller has
    until ``settle_frames`` further updates to turn it back into full-res
    pixels before the frame it names ages out of any buffer sized to match.
    """
    frame_index: int
    sharpness: float
    motion: float
    reason: str


@dataclass(frozen=True)
class Tick:
    """Everything one frame revealed — enough to drive a HUD without re-deriving it."""
    frame_index: int
    motion: float
    smooth: float
    brightness: float
    sharpness: float
    state: str
    capture: "Capture | None" = None


class LiveDetector:
    """Watches the motion signal and decides when the book has settled.

    Feed it every frame via :meth:`update`; act on ``tick.capture`` when set.
    The detector holds no pixels beyond the previous small frame — the trailing
    window it chooses from stores only ``(frame_index, sharpness)`` — so the
    caller decides how (and where) full-resolution frames are retained.

    Frame indices must be contiguous from 0. ``motion`` is accumulated
    positionally, and the artifacts built from it (``motion_signal.npy``,
    ``peaks.npy``, ``spreads.json``) are all keyed to a constant-rate timeline
    at ``fps``. A variable-rate source — anything driven by a browser's frame
    callbacks rather than a file — must map its timestamps onto that timeline
    and cover dropped frames via ``update(..., span=k)``, or the signal
    silently stops lining up with the recording it is supposed to describe.
    """

    def __init__(self, fps=30.0, settle_threshold=2.0, turn_threshold=5.0,
                 settle_time=0.4, smoothing_window=15):
        self.fps = fps
        self.settle_threshold = settle_threshold
        self.turn_threshold = turn_threshold
        self.smoothing_window = smoothing_window
        # Stillness is counted in frames, so the wall-clock settle time has to
        # be resolved against fps once, here, rather than per frame.
        self.settle_frames = max(1, int(settle_time * fps))

        self.state = WAITING
        self.paused = False
        self.motion = []            # one value per frame -> motion_signal.npy
        self.turn_frames = []       # frame index of each page turn -> peaks.npy

        self._window = deque(maxlen=self.settle_frames)   # (frame_index, sharpness)
        self._prev = None
        self._still_run = 0         # consecutive low-motion frames
        self._saw_turn = False

    # ── Driving ──────────────────────────────────────────────

    def update(self, frame_index, gray, span=1) -> Tick:
        """Advance the timeline to ``frame_index``. ``gray`` is the
        analysis-resolution grayscale frame.

        ``span`` is how many timeline slots this frame stands for — 1 for a
        source that delivers every frame (the local capture loop). A
        variable-rate source that dropped ``span - 1`` frames since the last
        update passes the count instead of pretending the book sat still: the
        measured difference between this frame and the previous one is
        credited to every slot in the gap. A page turn that fell entirely into
        dropped frames still changes the page, so the boundary difference is
        large and the turn registers; genuine stillness across a drop diffs
        to nothing and counts toward settling, exactly as it should.

        At most one capture can fire per call: a capture requires the smoothed
        signal to sit below settle_threshold, after which the constant
        in-span motion value can never climb back over turn_threshold.
        """
        motion = (float(np.mean(cv2.absdiff(self._prev, gray)))
                  if self._prev is not None else 0.0)
        self._prev = gray
        sharpness = laplacian_sharpness(gray)

        capture, smooth = None, 0.0
        for idx in range(frame_index - span + 1, frame_index + 1):
            self.motion.append(motion)

            # Trailing mean over the smoothing window — the online counterpart
            # of P1's uniform filter. Thresholding the raw per-frame difference
            # would chatter between states on sensor noise alone.
            win_n = min(self.smoothing_window, len(self.motion))
            smooth = float(np.mean(self.motion[-win_n:]))

            self._window.append((idx, sharpness))

            if not self.paused:
                still = smooth < self.settle_threshold
                self._still_run = self._still_run + 1 if still else 0
                settled = still and self._still_run >= self.settle_frames

                if self.state == WAITING:
                    if settled:
                        capture = self._take("initial") or capture
                        self.state = SETTLED
                elif self.state == SETTLED:
                    if smooth > self.turn_threshold:
                        self.state = TURNING
                        self._saw_turn = True
                        self.turn_frames.append(idx)
                elif self.state == TURNING:
                    if settled and self._saw_turn:
                        capture = self._take("settle") or capture
                        self.state = SETTLED
                        self._saw_turn = False

        return Tick(frame_index=frame_index, motion=motion, smooth=smooth,
                    brightness=float(np.mean(gray)), sharpness=sharpness,
                    state=self.state, capture=capture)

    def force_capture(self, reason="manual") -> "Capture | None":
        """Capture now (the operator's override), and treat the book as settled.

        Returns None if no frame has been seen yet. Resetting to SETTLED means
        the next page turn is detected normally afterwards, rather than the
        forced capture being followed by a duplicate when the book settles.
        """
        capture = self._take(reason)
        if capture is not None:
            self.state = SETTLED
            self._saw_turn = False
        return capture

    def set_paused(self, paused: bool):
        """Pause or resume auto-capture, restarting the stillness count either way.

        The count is reset on resume because the frames observed while paused
        say nothing about whether the operator is ready now; it is reset on
        pause so the two directions stay symmetric.
        """
        self.paused = bool(paused)
        self._still_run = 0

    def _take(self, reason) -> "Capture | None":
        if not self._window:
            return None
        frame_index, sharpness = max(self._window, key=lambda w: w[1])
        return Capture(frame_index=frame_index, sharpness=sharpness,
                       motion=self.motion[frame_index], reason=reason)

    # ── Artifacts ────────────────────────────────────────────

    def motion_signal(self):
        """The raw per-frame motion signal, as P1 would have written it."""
        return np.array(self.motion, dtype=np.float64)

    def smoothed_signal(self):
        """The offline smoothing of that signal — P1's, not the online trailing mean.

        The two differ by design: thresholding live can only look backwards,
        but the saved signal is what the plots and any offline re-analysis
        read, and those should match what P1 would have produced from the
        recording.
        """
        sig = self.motion_signal()
        return uniform_filter1d(sig, size=self.smoothing_window) if len(sig) else sig

    def peaks(self):
        """Detected page turns, in the shape P2 writes to peaks.npy."""
        return np.array(sorted(self.turn_frames), dtype=np.int64)

    def spreads(self, total_frames):
        """Spread boundaries, in the shape P2 writes to spreads.json."""
        return build_spreads(self.peaks(), total_frames, self.fps)

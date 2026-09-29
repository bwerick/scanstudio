"""
Capture sounds, synthesized rather than borrowed from the system.

Most notification dings grate because of their attack: they start at full
amplitude, which the ear hears as a click. Every sound here opens with a
short raised-cosine fade-in and dies away exponentially, so it reads as a
soft pluck rather than a strike.

Each set is a family — the same timbre voicing different events — so you
can tell a manual capture from an undo without looking up, while the whole
thing still sounds like one piece of software.

    python ui/live/sounds.py        # (re)generate ui/live/sounds/*.wav
"""

import platform
import shutil
import subprocess
import wave
from pathlib import Path

import numpy as np

SR = 44100
SOUND_DIR = Path(__file__).resolve().parent / "sounds"
SETS = ("pluck", "chime", "tick")
EVENTS = ("capture", "manual", "undo", "start", "stop")

# note names → Hz, kept in a gentle mid register (lower reads as softer)
N = {"C5": 523.25, "E5": 659.25, "G5": 783.99, "A5": 880.00,
     "B5": 987.77, "C6": 1046.50, "E6": 1318.51, "G4": 392.00}


def _env(n, attack=0.008, tau=0.14):
    t = np.arange(n) / SR
    a = max(1, int(attack * SR))
    env = np.exp(-t / tau)
    env[:a] *= 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, a))   # soft onset
    return env


def _tone(freq, dur, partials=((1, 1.0), (2, 0.28), (3, 0.08)),
          tau=0.14, attack=0.008):
    n = int(dur * SR)
    t = np.arange(n) / SR
    x = sum(a * np.sin(2 * np.pi * freq * k * t) for k, a in partials)
    return x * _env(n, attack, tau)


def _marimba(freq, dur=0.32):
    # wooden bars ring a strong 4th partial that dies fast
    return (_tone(freq, dur, ((1, 1.0),), tau=0.16)
            + 0.25 * _tone(freq * 4, dur, ((1, 1.0),), tau=0.03))


def _tick(dur=0.07):
    n = int(dur * SR)
    rng = np.random.default_rng(7)
    noise = rng.standard_normal(n)
    noise = np.convolve(noise, np.ones(18) / 18, mode="same")  # take the hiss off
    body = _tone(N["G4"] / 2, dur, ((1, 1.0),), tau=0.02, attack=0.002)
    return 0.35 * noise * _env(n, 0.002, 0.012) + 0.8 * body


def _seq(parts, gap):
    """Lay voices out in time: parts=[(offset_s, signal), ...]."""
    end = max(int(o * SR) + len(s) for o, s in parts)
    out = np.zeros(end + int(gap * SR))
    for o, s in parts:
        i = int(o * SR)
        out[i:i + len(s)] += s
    return out


def build(set_name, event):
    if set_name == "pluck":
        v = lambda f, d=0.30: _tone(f, d, tau=0.12)
    elif set_name == "chime":
        v = lambda f, d=0.40: _tone(f, d, ((1, 1.0), (2, 0.18)), tau=0.20)
    else:
        v = lambda f, d=0.30: _marimba(f, d)

    if event == "capture":
        if set_name == "chime":
            return _seq([(0.0, v(N["G5"])), (0.085, v(N["B5"]))], 0.05)
        if set_name == "tick":
            return _tick()
        return v(N["E5"])
    if event == "manual":        # the capture sound, twice, the second a step up
        if set_name == "tick":
            return _seq([(0.0, _tick()), (0.09, _tick())], 0.03)
        return _seq([(0.0, v(N["E5"], 0.2)), (0.09, v(N["G5"], 0.24))], 0.03)
    if event == "undo":          # falling: something was taken back
        return _seq([(0.0, v(N["B5"], 0.2)), (0.1, v(N["G5"], 0.28))], 0.03)
    if event == "start":         # rising: capture is live
        return _seq([(0.0, v(N["C5"], 0.2)), (0.08, v(N["E5"], 0.2)),
                     (0.16, v(N["G5"], 0.34))], 0.03)
    if event == "stop":          # resolving down to the root
        return _seq([(0.0, v(N["G5"], 0.24)), (0.12, v(N["C5"], 0.5))], 0.03)
    raise ValueError(event)


def _write(path, x, peak=0.32):
    x = x / max(1e-9, np.max(np.abs(x))) * peak           # modest, never loud
    fade = min(len(x), int(0.004 * SR))
    x[-fade:] *= np.linspace(1, 0, fade)                   # no click on release
    pcm = (x * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


def generate(out_dir=SOUND_DIR):
    out_dir.mkdir(parents=True, exist_ok=True)
    for s in SETS:
        for e in EVENTS:
            _write(out_dir / f"{s}_{e}.wav", build(s, e))
    return sorted(out_dir.glob("*.wav"))


class Player:
    """Non-blocking playback. Missing player or sound set → silent, never a crash."""

    def __init__(self, set_name="pluck", sound_dir=SOUND_DIR):
        self.set = set_name
        self.dir = Path(sound_dir)
        self.muted = set_name == "off"
        if not self.muted and not (self.dir / f"{set_name}_capture.wav").exists():
            generate(self.dir)
        sysname = platform.system()
        if sysname == "Darwin" and shutil.which("afplay"):
            self._cmd = ["afplay"]
        elif shutil.which("paplay"):
            self._cmd = ["paplay"]
        elif shutil.which("aplay"):
            self._cmd = ["aplay", "-q"]
        else:
            self._cmd = None
        self._win = sysname == "Windows"

    def play(self, event):
        if self.muted:
            return
        path = self.dir / f"{self.set}_{event}.wav"
        if not path.exists():
            return
        try:
            if self._win:
                import winsound
                winsound.PlaySound(str(path),
                                   winsound.SND_FILENAME | winsound.SND_ASYNC)
            elif self._cmd:
                subprocess.Popen(self._cmd + [str(path)],
                                 stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL)
        except Exception:
            pass


if __name__ == "__main__":
    for p in generate():
        print(p.name)

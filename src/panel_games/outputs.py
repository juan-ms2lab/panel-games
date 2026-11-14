"""Where frames go. Every output takes the game's own canvas (any size) in show() and blanks in clear().

DDPOutput maps the game canvas onto a fixed target canvas: the largest whole-number nearest-neighbour upscale
that fits, centered (scale 1 when it fits exactly). A game larger than the target is center-cropped, never
shrunk; the catalog normally keeps such games from starting at all.
"""

import threading

import cv2
import numpy as np

from .ddp import PORT, DDPSender


def fit(frame: np.ndarray, tw: int, th: int) -> np.ndarray:
    """Place frame (h, w, 3) on a black (th, tw, 3) canvas: integer upscale, centered; crop if too big."""
    h, w = frame.shape[:2]
    s = max(1, min(tw // w, th // h))
    if s > 1:
        frame = frame.repeat(s, axis=0).repeat(s, axis=1)
        h, w = h * s, w * s
    if w > tw:
        frame, w = frame[:, (w - tw) // 2 : (w - tw) // 2 + tw], tw
    if h > th:
        frame, h = frame[(h - th) // 2 : (h - th) // 2 + th], th
    out = np.zeros((th, tw, 3), np.uint8)
    y, x = (th - h) // 2, (tw - w) // 2
    out[y : y + h, x : x + w] = frame
    return out


class DDPOutput:
    def __init__(self, host: str, port: int = PORT, target_w: int = 768, target_h: int = 64):
        self.tw, self.th = target_w, target_h
        self.sender = DDPSender(host, port)

    def show(self, frame: np.ndarray):
        try:
            self.sender.send(fit(frame, self.tw, self.th).tobytes())
        except OSError:
            pass  # receiver unreachable or network down: drop the frame, try again next one

    def clear(self):
        self.show(np.zeros((self.th, self.tw, 3), np.uint8))


class ScreenOutput:
    """Keeps the latest frame for the web server's /screen page, encoded to PNG on demand."""

    def __init__(self):
        self._lock = threading.Lock()
        self.frame = np.zeros((128, 768, 3), np.uint8)
        self.count = 0
        self._png = (-1, b"")

    def show(self, frame: np.ndarray):
        with self._lock:
            self.frame = frame.copy()
            self.count += 1

    def clear(self):
        with self._lock:
            self.frame = np.zeros_like(self.frame)
            self.count += 1

    def png(self) -> tuple[int, bytes]:
        with self._lock:
            count, frame = self.count, self.frame
        if self._png[0] != count:
            ok, buf = cv2.imencode(".png", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, 1])
            self._png = (count, buf.tobytes())
        return self._png

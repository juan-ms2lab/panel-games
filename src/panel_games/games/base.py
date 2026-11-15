"""Helpers shared by multiplayer games: lobby, movement on a tile grid, podium and text drawing."""

import math

import cv2
import numpy as np

from .pad import START

FONT = cv2.FONT_HERSHEY_SIMPLEX
PLAYER_COLORS = [(0, 200, 255), (255, 140, 0), (120, 255, 60), (255, 60, 200),
                 (255, 235, 0), (255, 80, 80), (160, 110, 255), (235, 235, 235)]


class Lobby:
    """Waits for the host to press START. The host is the first player to join, then the next in line."""

    def __init__(self):
        self.order: list[int] = []
        self._held = True

    @property
    def host(self) -> int | None:
        return self.order[0] if self.order else None

    def reset(self):
        self._held = True  # a START still held from the last round must not restart instantly

    def update(self, joined, pads: list) -> bool:
        """Track who has joined; return True on the frame the host presses START."""
        self.order = [s for s in self.order if s in joined] + [s for s in sorted(joined) if s not in self.order]
        if not self.order:
            self._held = False
            return False
        pressed = bool((pads[self.host] or 0) & START)
        go = pressed and not self._held
        self._held = pressed
        return go

    @property
    def message(self) -> str:
        return "JOIN ON YOUR PHONE" if self.host is None else f"PLAYER {self.host + 1}: PRESS START"


class Mover:
    """An entity on a tile grid: at tile (tx, ty), moving (dx, dy) with progress p toward the next tile."""

    def __init__(self, tx, ty):
        self.tx, self.ty = tx, ty
        self.dx = self.dy = 0
        self.p = 0.0

    def pos(self):
        return self.tx + self.dx * self.p, self.ty + self.dy * self.p

    def tile(self):
        x, y = self.pos()
        return int(math.floor(x + 0.5)), int(math.floor(y + 0.5))


def reverse(e: Mover):
    e.tx, e.ty, e.dx, e.dy, e.p = e.tx + e.dx, e.ty + e.dy, -e.dx, -e.dy, 1.0 - e.p


def advance(e: Mover, dist: float, choose):
    """Move e by dist tiles. choose(e) is asked for a direction at each tile centre, (0, 0) to stop."""
    for _ in range(12):
        if dist <= 1e-9:
            return
        if (e.dx, e.dy) == (0, 0):
            e.dx, e.dy = choose(e)
            if (e.dx, e.dy) == (0, 0):
                e.p = 0.0
                return
        remaining = 1.0 - e.p
        if dist < remaining:
            e.p += dist
            return
        dist -= remaining
        e.tx, e.ty, e.p = e.tx + e.dx, e.ty + e.dy, 0.0
        e.dx, e.dy = choose(e)
        if (e.dx, e.dy) == (0, 0):
            return


def draw_podium(canvas: np.ndarray, ranking: list, t: float):
    """ranking is [(slot, score), ...] best first; draws the top three. Laid out for 128 rows, scaled to fit."""
    h, w = canvas.shape[:2]
    k = min(1.0, h / 128)
    canvas[:] = (10, 8, 30)
    for i in range(70):
        x, y = (i * 97 + int(t * 10) * (i % 3)) % w, int(t * 45 + i * 37) % h
        canvas[y : y + 2, x : x + 2] = PLAYER_COLORS[i % 8]
    center_text(canvas, "PODIUM", int(18 * k), 0.7 * k, (255, 230, 0), box=False)
    floor = int(122 * k)
    for place, cx, bh, color in ((0, w // 2, 56, (255, 200, 0)), (1, w // 2 - 130, 40, (205, 205, 215)),
                                  (2, w // 2 + 130, 26, (205, 127, 50))):
        if place >= len(ranking):
            continue
        slot, score = ranking[place]
        bh = int(bh * k)
        top = floor - bh
        canvas[top:floor, cx - 50 : cx + 50] = color
        text(canvas, str(place + 1), cx - int(6 * k), top + min(bh - 2, int(26 * k)), 0.9 * k, (30, 30, 30), 2 if k == 1 else 1)
        cv2.circle(canvas, (cx, top - int(10 * k)), max(2, int(6 * k)), PLAYER_COLORS[slot], -1)
        label = f"P{slot + 1} {score}"
        (tw, _), _ = cv2.getTextSize(label, FONT, 0.5 * k, 1)
        text(canvas, label, cx - tw // 2, top - int(22 * k), 0.5 * k, PLAYER_COLORS[slot])


def text(canvas: np.ndarray, s: str, x: int, y: int, scale: float = 0.45, color=(255, 255, 255), thickness: int = 1):
    cv2.putText(canvas, s, (x, y), FONT, scale, color, thickness, cv2.LINE_AA)


def center_text(canvas: np.ndarray, s: str, y: int, scale: float = 0.8, color=(255, 255, 255),
                thickness: int = 1, box: bool = True):
    (tw, th), _ = cv2.getTextSize(s, FONT, scale, thickness)
    x = (canvas.shape[1] - tw) // 2
    if box:
        cv2.rectangle(canvas, (x - 4, y - th - 4), (x + tw + 4, y + 5), (0, 0, 0), -1)
    cv2.putText(canvas, s, (x, y), FONT, scale, color, thickness, cv2.LINE_AA)

"""Panel Sweep: competitive Minesweeper. Everyone has a cursor on one shared board.

Revealing safe squares scores points, correct flags earn a bonus at the end, and each mine costs a life.
Clear the board for a podium; if every player is out, the last one standing wins.
"""

import random
from collections import deque

import cv2
import numpy as np

from .base import FONT, PLAYER_COLORS, Lobby, center_text, draw_podium, text
from .pad import A, B, DOWN, LEFT, RIGHT, UP

CELL = 16
GW, GH = 48, 7
OY = 16
MINES = 50
LIVES, STUN = 2, 3.0
REPEAT_DELAY, REPEAT_RATE = 0.28, 0.07
FLAG_BONUS = 10
RESULT_TIME_PODIUM, RESULT_TIME_LAST = 14.0, 7.0
NUMBER_COLORS = [None, (80, 140, 255), (80, 220, 100), (255, 90, 90), (150, 120, 255),
                 (255, 150, 60), (60, 220, 220), (240, 240, 240), (170, 170, 170)]
KEY_DIRS = ((UP, (0, -1)), (DOWN, (0, 1)), (LEFT, (-1, 0)), (RIGHT, (1, 0)))


class Cursor:
    def __init__(self, slot: int):
        self.slot = slot
        self.x, self.y = 4 + slot * 8, GH // 2
        self.lives, self.score = LIVES, 0
        self.stun, self.hold, self.prev = 0.0, 0.0, 0
        self.alive, self.died_at = True, 0.0


class Sweep:
    id = "sweep"
    title = "Panel Sweep"
    description = "Minesweeper for everyone at once. Clear squares for points, flag mines for a bonus, avoid the blasts."
    max_players = 6
    size = (768, 128)
    controls = "D-pad moves your cursor, A reveals (or clears around a number), B flags"

    def __init__(self, width: int, height: int, seed=None):
        if width < GW * CELL or height < OY + GH * CELL:
            raise ValueError(f"Panel Sweep needs a {GW * CELL}x{OY + GH * CELL} display")
        self.width, self.height = width, height
        self.rng = random.Random(seed)
        self.t = 0.0
        self.lobby = Lobby()
        self.joined: set[int] = set()
        self.new_round()

    # ---- setup

    def new_round(self, joined=None):
        self.state = "lobby"
        self.lobby.reset()
        self.joined = set(joined or ())
        self.mine = [[False] * GW for _ in range(GH)]
        self.shown = [[False] * GW for _ in range(GH)]
        self.flag: list[list[int | None]] = [[None] * GW for _ in range(GH)]  # slot that flagged the square
        self.boom = [[False] * GW for _ in range(GH)]
        self.cursors: dict[int, Cursor] = {}
        self.placed = False
        self.safe_left = GW * GH - MINES
        self.timer, self.result = 0.0, {}

    def _start(self, roster):
        self.cursors = {s: Cursor(s) for s in sorted(roster)}
        self.state = "play"

    def info(self) -> dict:
        return {"state": self.state, "host": self.lobby.host}

    def _place_mines(self, fx, fy):
        """Lay the mines after the first reveal, keeping that square and its neighbours safe."""
        cells = [(x, y) for y in range(GH) for x in range(GW) if abs(x - fx) > 1 or abs(y - fy) > 1]
        for x, y in self.rng.sample(cells, MINES):
            self.mine[y][x] = True
        self.placed = True

    def count(self, x, y) -> int:
        return sum(self.mine[ny][nx] for nx, ny in self._neighbours(x, y))

    @staticmethod
    def _neighbours(x, y):
        return [(x + dx, y + dy) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
                if (dx or dy) and 0 <= x + dx < GW and 0 <= y + dy < GH]

    # ---- actions

    def _reveal(self, c: Cursor, x, y):
        if self.flag[y][x] is not None:
            return
        if self.shown[y][x]:
            self._chord(c, x, y)
            return
        if not self.placed:
            self._place_mines(x, y)
        if self.mine[y][x]:
            self._hit(c, x, y)
            return
        queue = deque([(x, y)])
        self.shown[y][x] = True
        while queue:
            cx, cy = queue.popleft()
            self.safe_left -= 1
            c.score += 1
            if self.count(cx, cy) == 0:
                for nx, ny in self._neighbours(cx, cy):
                    if not self.shown[ny][nx] and self.flag[ny][nx] is None and not self.mine[ny][nx]:
                        self.shown[ny][nx] = True
                        queue.append((nx, ny))

    def _chord(self, c: Cursor, x, y):
        n = self.count(x, y)
        around = self._neighbours(x, y)
        if n and sum(1 for nx, ny in around if self.flag[ny][nx] is not None) == n:
            for nx, ny in around:
                if not self.shown[ny][nx] and self.flag[ny][nx] is None:
                    self._reveal(c, nx, ny)
                    if c.stun > 0:
                        break

    def _hit(self, c: Cursor, x, y):
        self.shown[y][x] = self.boom[y][x] = True
        c.lives -= 1
        c.stun = STUN
        if c.lives <= 0:
            c.alive, c.died_at = False, self.t

    def _toggle_flag(self, c: Cursor, x, y):
        if self.shown[y][x]:
            return
        self.flag[y][x] = None if self.flag[y][x] is not None else c.slot

    # ---- update

    def update(self, dt: float, pads: list):
        self.t += dt
        joined = {i for i, k in enumerate(pads) if k is not None}
        if self.state == "lobby":
            self.joined = joined
            if self.lobby.update(joined, pads):
                self._start(joined)
        elif self.state == "play":
            self._play(dt, pads)
        else:
            self.timer -= dt
            if self.timer <= 0:
                self.new_round(joined)

    def _play(self, dt, pads):
        for c in self.cursors.values():
            if not c.alive:
                continue
            keys = pads[c.slot] if c.slot < len(pads) else None
            if keys is None:
                c.alive, c.died_at = False, self.t
                continue
            c.stun = max(0.0, c.stun - dt)
            if c.stun > 0:
                c.prev = keys
                continue
            d = next((v for bit, v in KEY_DIRS if keys & bit), None)
            if d is None:
                c.hold = 0.0
            else:
                c.hold -= dt
                if c.hold <= 0:
                    c.x = min(GW - 1, max(0, c.x + d[0]))
                    c.y = min(GH - 1, max(0, c.y + d[1]))
                    c.hold = REPEAT_RATE if c.prev & sum(bit for bit, v in KEY_DIRS if v == d) else REPEAT_DELAY
            if keys & A and not c.prev & A:
                self._reveal(c, c.x, c.y)
            if keys & B and not c.prev & B:
                self._toggle_flag(c, c.x, c.y)
            c.prev = keys
        self._check_end()

    def _check_end(self):
        cleared = self.safe_left <= 0
        out = not any(c.alive for c in self.cursors.values())
        if not (cleared or out):
            return
        for y in range(GH):
            for x in range(GW):
                owner = self.flag[y][x]
                if owner is not None and owner in self.cursors:
                    self.cursors[owner].score += FLAG_BONUS if self.mine[y][x] else -FLAG_BONUS
        cs = list(self.cursors.values())
        alive = sorted((c for c in cs if c.alive), key=lambda c: -c.score)
        fallen = sorted((c for c in cs if not c.alive), key=lambda c: (-c.died_at, -c.score))
        mode = "podium" if cleared else "last"
        self.result = {"mode": mode, "solo": len(cs) == 1, "ranking": [(c.slot, c.score) for c in alive + fallen]}
        self.state = "over"
        self.timer = RESULT_TIME_PODIUM if cleared else RESULT_TIME_LAST

    # ---- drawing

    def _draw_cell(self, canvas, x, y, reveal_all):
        px, py = x * CELL, OY + y * CELL
        cell = canvas[py : py + CELL, px : px + CELL]
        if self.shown[y][x] or (reveal_all and self.mine[y][x]):
            cell[:] = (120, 20, 20) if self.boom[y][x] else (24, 26, 38)
            cell[0, :] = cell[:, 0] = (14, 16, 26)
            if self.mine[y][x]:
                cv2.circle(canvas, (px + 8, py + 8), 4, (230, 230, 230), -1)
                for ax, ay in ((-6, 0), (6, 0), (0, -6), (0, 6)):
                    cv2.line(canvas, (px + 8, py + 8), (px + 8 + ax, py + 8 + ay), (230, 230, 230), 1)
            else:
                n = self.count(x, y)
                if n:
                    cv2.putText(canvas, str(n), (px + 4, py + 13), FONT, 0.55, NUMBER_COLORS[n], 1, cv2.LINE_AA)
        else:
            cell[:] = (150, 160, 195)
            cell[0, :] = cell[:, 0] = (215, 222, 245)
            cell[-1, :] = cell[:, -1] = (85, 92, 120)
            owner = self.flag[y][x]
            if owner is not None:
                color = PLAYER_COLORS[owner] if owner < len(PLAYER_COLORS) else (255, 60, 60)
                cell[3:13, 7:9] = (30, 30, 30)
                pts = np.array([(px + 8, py + 2), (px + 8, py + 9), (px + 2, py + 5)], np.int32)
                cv2.fillConvexPoly(canvas, pts, color)

    def render(self, canvas: np.ndarray):
        if self.state == "over" and self.result["mode"] == "podium":
            return draw_podium(canvas, self.result["ranking"], self.t)
        canvas[:] = 0
        reveal_all = self.state == "over"
        for y in range(GH):
            for x in range(GW):
                self._draw_cell(canvas, x, y, reveal_all)
        roster = self.cursors if self.state != "lobby" else {s: Cursor(s) for s in self.joined}
        for s, c in sorted(roster.items()):
            if c.alive and not (c.stun > 0 and int(self.t * 8) % 2):
                px, py = c.x * CELL, OY + c.y * CELL
                inset = s % 3
                cv2.rectangle(canvas, (px + inset, py + inset), (px + CELL - 1 - inset, py + CELL - 1 - inset),
                              PLAYER_COLORS[s], 2 if inset == 0 else 1)
        for s, c in sorted(roster.items()):
            life = "x" * max(0, c.lives)
            text(canvas, f"P{s + 1} {c.score} {life}", 6 + s * 120, 12, 0.42, PLAYER_COLORS[s] if c.alive else (110, 110, 110))
        flags = sum(1 for row in self.flag for f in row if f is not None)
        text(canvas, f"MINES {MINES - flags}", self.width - 90, 12, 0.42)
        if self.state == "lobby":
            canvas[OY:] = canvas[OY:] // 3
            center_text(canvas, "PANEL SWEEP", 62, 1.0, (255, 230, 0))
            center_text(canvas, "A reveals, B flags. Two lives each. Clear the board for a podium.", 84, 0.5)
            center_text(canvas, self.lobby.message, 108, 0.8, (120, 255, 120) if self.joined else (255, 255, 255))
        elif self.state == "over":
            ranking = self.result["ranking"]
            if self.result["solo"]:
                center_text(canvas, "GAME OVER", 70, 1.0, (255, 80, 80))
            else:
                center_text(canvas, f"PLAYER {ranking[0][0] + 1} WINS", 70, 1.0, PLAYER_COLORS[ranking[0][0]])
                center_text(canvas, "LAST ONE STANDING", 96, 0.6)

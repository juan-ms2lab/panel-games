"""Panel Hop: cross the whole display, biome by biome, from a quiet country road to the finish pond.

Lanes are columns, so everyone hops left to right across the long side. Traffic and platforms move up and
down. The biomes get harder towards the end, finishing with a huge city highway.
"""

import math
import random

import cv2
import numpy as np

from .base import PLAYER_COLORS, Lobby, center_text, draw_podium, text
from .pad import DOWN, LEFT, RIGHT, UP

T = 8
GW, GH = 96, 14
OY = 16
P = GH + 12                      # length of each lane's looping belt, in rows
MARGIN = 6                       # belt position 0 sits this many rows above the screen
HOP_T, HOP_GAP = 0.10, 0.065   # a full hop-and-recover cycle is 10 frames at 60 fps
LIVES = 3
TIME_LIMIT, LAST_CALL = 300.0, 45.0
POND_START = 88
FINISH_BONUS = [3000, 2400, 1900, 1500]
TRAIN_LEN, TRAIN_SPEED, TRAIN_WARN = 18.0, 28.0, 1.3
RESULT_TIME = 14.0
KEY_DIRS = ((UP, (0, -1)), (DOWN, (0, 1)), (LEFT, (-1, 0)), (RIGHT, (1, 0)))
LENGTHS = {"car": 2, "moto": 1, "boulder": 1, "bus": 4}
CAR_COLORS = [(230, 60, 60), (60, 120, 240), (240, 200, 40), (70, 200, 110), (230, 230, 235), (220, 110, 30), (170, 80, 220)]


def turtle_state(offset: float, t: float) -> str:
    """Turtles dive for a moment every few seconds; 'warn' flickers just before they go under."""
    phase = (t + offset) % 7.0
    return "up" if phase < 4.8 else "warn" if phase < 5.4 else "down"


class Lane:
    def __init__(self, kind: str, biome: str, direction: int = 1, speed: float = 0.0, objs=(), trains=()):
        self.kind, self.biome, self.dir, self.speed = kind, biome, direction, speed
        self.objs = list(objs)       # (offset, length, style)
        self.trains = list(trains)   # start times of trains (rail lanes)

    def intervals(self, t: float, offscreen: bool = False):
        """Objects at time t as (row start, row end, style, offset); only those on screen unless offscreen."""
        out = []
        for off, length, style in self.objs:
            y0 = (off + self.dir * self.speed * t) % P - MARGIN
            for y in (y0, y0 - P):
                if offscreen or (y < GH and y + length > 0):
                    out.append((y, y + length, style, off))
        if self.kind == "rail":
            for ts in self.trains:
                phase = t - ts
                if 0 <= phase <= (GH + TRAIN_LEN) / TRAIN_SPEED:
                    y0 = -TRAIN_LEN + phase * TRAIN_SPEED if self.dir > 0 else GH - phase * TRAIN_SPEED
                    out.append((y0, y0 + TRAIN_LEN, "train", 0.0))
        return out

    def hits(self, y0: float, y1: float, t: float, offscreen: bool = False) -> bool:
        return any(a < y1 and b > y0 for a, b, _, _ in self.intervals(t, offscreen))

    def entering(self, t: float, horizon: float = 0.9):
        """'top' or 'bottom' when a vehicle is about to come on screen there, so the player gets a warning."""
        if self.kind != "road":
            return None
        reach = self.speed * horizon
        for off, length, _ in self.objs:
            y0 = (off + self.dir * self.speed * t) % P - MARGIN
            for y in (y0, y0 - P):
                if self.dir > 0 and -reach < y + length <= 0:
                    return "top"
                if self.dir < 0 and GH <= y < GH + reach:
                    return "bottom"
        return None

    def platform(self, yc: float, t: float):
        """Vertical velocity (rows per second) of a platform under row position yc, else None."""
        for a, b, style, off in self.intervals(t):
            if a <= yc < b and not (style == "turtle" and turtle_state(off, t) == "down"):
                return self.dir * self.speed
        return None

    def warning(self, t: float) -> bool:
        return any(ts - TRAIN_WARN <= t < ts for ts in self.trains)


def _spread(rng, lengths, gap_min, jitter=(0.7, 1.3)):
    """Offsets for objects of the given lengths looping on a belt, with roughly even random gaps."""
    n = len(lengths)
    free = P - sum(lengths)
    weights = [rng.uniform(*jitter) for _ in range(n)]
    gaps = [gap_min + (free - gap_min * n) * w / sum(weights) for w in weights]
    cursor, offsets = rng.uniform(0, P), []
    for length, gap in zip(lengths, gaps):
        offsets.append(cursor % P)
        cursor += length + gap
    return offsets


def road_lane(rng, biome, direction, kinds, cover, gap_min, speeds):
    lengths, styles = [], []
    while True:
        style = rng.choice(kinds)
        length = rng.choice((3, 4)) if style == "truck" else LENGTHS[style]
        if sum(lengths) + length + gap_min * (len(lengths) + 1) > P or (lengths and sum(lengths) / P >= cover):
            break
        lengths.append(length)
        styles.append(style)
    speed = rng.uniform(*speeds)
    objs = [(o, ln, st) for o, ln, st in zip(_spread(rng, lengths, gap_min), lengths, styles)]
    return Lane("road", biome, direction, speed, objs)


def water_lane(rng, biome, direction, styles, cover, speeds, length_range):
    lengths, kinds = [], []
    while sum(lengths) / P < cover:
        style = rng.choice(styles)
        length = 3 if style == "turtle" else rng.randint(*length_range)
        if lengths and sum(lengths) + length + 0.6 * (len(lengths) + 1) > P:
            break
        lengths.append(length)
        kinds.append(style)
    objs = [(o, ln, st) for o, ln, st in zip(_spread(rng, lengths, 0.6, (0.75, 1.25)), lengths, kinds)]
    return Lane("water", biome, direction, rng.uniform(*speeds), objs)


def rail_lane(rng, direction):
    times, t = [], rng.uniform(2.0, 7.0)
    while t < 400:
        times.append(t)
        t += rng.uniform(6.0, 10.0)
    return Lane("rail", "rail", direction, 0.0, (), times)


def build_level(rng: random.Random) -> list[Lane]:
    lanes: list[Lane] = []
    flip = [1]

    def safe(biome, n=1):
        lanes.extend(Lane("safe", biome) for _ in range(n))

    def next_dir(block, i):
        return 1 if (i // block) % 2 == 0 else -1

    def roads(n, biome, kinds, cover, gap_min, speeds, block=1, first=1):
        for i in range(n):
            lanes.append(road_lane(rng, biome, first * next_dir(block, i), kinds, cover, gap_min, speeds))

    def waters(n, biome, styles, cover, speeds, lens, first=1):
        for i in range(n):
            lanes.append(water_lane(rng, biome, first * (1 if i % 2 == 0 else -1), styles, cover, speeds, lens))

    safe("start", 3)
    roads(6, "country", ("car", "car", "moto"), 0.14, 3.5, (1.8, 2.8))
    safe("median")
    waters(5, "stream", ("log", "log", "turtle"), 0.72, (1.0, 1.7), (4, 5))
    safe("bank")
    for kind in "bbrbbrsbbrbbrb":
        if kind == "b":
            lanes.append(road_lane(rng, "desert", flip[0], ("boulder",), 0.12, 3.0, (2.8, 4.2)))
            flip[0] = -flip[0]
        elif kind == "r":
            lanes.append(rail_lane(rng, flip[0]))
            flip[0] = -flip[0]
        else:
            safe("rock")
    safe("oasis")
    waters(8, "lava", ("rock",), 0.52, (1.8, 2.8), (2, 3))
    safe("rock")
    roads(8, "suburb", ("car", "car", "truck", "moto"), 0.24, 2.6, (3.2, 4.6), block=2)
    safe("median")
    waters(6, "canal", ("log", "turtle"), 0.55, (3.0, 4.2), (3, 4))
    safe("bank")
    # Four six-lane highway blocks; each is busier and faster than the last. Tuned with a perfect-information
    # planner so every block can always be crossed, while still getting much harder than the biomes before it.
    city = [(0.25, 2.6, (3.4, 4.6)), (0.27, 2.5, (3.8, 5.0)), (0.29, 2.5, (4.0, 5.3)), (0.30, 2.4, (4.2, 5.5))]
    for block, (cover, gap, speeds) in enumerate(city):
        roads(6, "city", ("car", "car", "truck", "bus", "moto"), cover, gap, speeds,
              block=6, first=1 if block % 2 == 0 else -1)
        if block < 3:
            safe("barrier")
    safe("bank", 5)
    lanes.extend(Lane("pond", "pond") for _ in range(GW - len(lanes)))
    assert len(lanes) == GW
    return lanes


class Hopper:
    w = 0.7

    def __init__(self, slot: int):
        self.slot, self.lives = slot, LIVES
        self.max_col, self.cp = 0, 0
        self.place = 0
        self.finished = False
        self.reset_position(0)

    def reset_position(self, col):
        self.x, self.y = float(col), float(2 + self.slot * 3)
        self.hop = None
        self.alive, self.invuln, self.cool, self.death_t = True, 1.6, 0.0, 0.0
        self.face, self.cause = (1, 0), ""


class Hop:
    id = "hop"
    title = "Panel Hop"
    description = "Hop across the whole display: country road, rivers, a railway desert, lava, and a huge city highway to the finish pond."
    max_players = 4
    size = (768, 128)
    controls = "Hold or tap the D-pad to hop. Ride logs over water. Dodge everything else"

    def __init__(self, width: int, height: int, seed=None):
        if width < GW * T or height < OY + GH * T:
            raise ValueError(f"Panel Hop needs a {GW * T}x{OY + GH * T} display")
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
        self.lanes = build_level(self.rng)
        self.hoppers: dict[int, Hopper] = {}
        self.finished: list[int] = []
        self.play_t, self.deadline, self.timer, self.result = 0.0, TIME_LIMIT, 0.0, {}
        self._build_bg()

    def _start(self, roster):
        self.hoppers = {s: Hopper(s) for s in sorted(roster)}
        self.state, self.play_t = "play", 0.0

    def info(self) -> dict:
        return {"state": self.state, "host": self.lobby.host}

    # ---- rules shared by the game and the tests

    def cell_state(self, col: int, y: float, t: float):
        """(safe, carry velocity) for a hopper whose top edge is at row y in column col at time t."""
        lane = self.lanes[col]
        if lane.kind in ("safe", "pond"):
            return True, 0.0
        if lane.kind == "water":
            v = lane.platform(y + 0.5, t)
            return v is not None, v or 0.0
        return not lane.hits(y + 0.12, y + 0.88, t), 0.0

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
        self.play_t += dt
        for h in self.hoppers.values():
            keys = pads[h.slot] if h.slot < len(pads) else None
            if keys is None:
                if not h.finished:
                    h.lives, h.alive = 0, False  # left the game
                continue
            self._update_hopper(h, dt, keys)
        active = [h for h in self.hoppers.values() if not h.finished and (h.alive or h.lives > 0)]
        if not active or self.play_t >= self.deadline:
            self._end()

    def _update_hopper(self, h: Hopper, dt: float, keys: int):
        h.invuln = max(0.0, h.invuln - dt)
        h.cool = max(0.0, h.cool - dt)
        if h.finished:
            return
        if not h.alive:
            h.death_t -= dt
            if h.death_t <= 0 and h.lives > 0:
                h.reset_position(h.cp)
            return
        t = self.play_t
        if h.hop:
            x0, y0, x1, y1, el = h.hop
            el += dt
            f = min(1.0, el / HOP_T)
            h.x, h.y = x0 + (x1 - x0) * f, y0 + (y1 - y0) * f
            h.hop = [x0, y0, x1, y1, el]
            if f >= 1.0:
                h.hop = None
                self._land(h)
        else:
            lane = self.lanes[int(round(h.x))]
            if lane.kind == "water":
                v = lane.platform(h.y + 0.5, t)
                if v is None:
                    return self._die(h, "drown")
                h.y += v * dt
                if h.y < -0.45 or h.y > GH - 0.55:
                    return self._die(h, "swept")
            d = next((v for bit, v in KEY_DIRS if keys & bit), None)
            if d and h.cool <= 0:
                nx, ny = int(round(h.x)) + d[0], h.y + d[1]
                if 0 <= nx < GW and 0 <= ny <= GH - 1:
                    h.hop, h.face, h.cool = [h.x, h.y, float(nx), ny, 0.0], d, HOP_T + HOP_GAP
        if h.alive and h.invuln <= 0 and not h.finished:
            for c in range(int(math.floor(h.x + 0.15)), int(math.floor(h.x + 0.85)) + 1):
                if 0 <= c < GW and self.lanes[c].kind in ("road", "rail") and self.lanes[c].hits(h.y + 0.12, h.y + 0.88, t):
                    self._die(h, "squash")
                    break

    def _land(self, h: Hopper):
        col = int(round(h.x))
        lane = self.lanes[col]
        h.x = float(col)
        if lane.kind != "water":
            h.y = float(min(GH - 1, max(0, round(h.y))))
        h.max_col = max(h.max_col, col)
        if lane.kind == "safe":
            h.cp = max(h.cp, col)
        elif lane.kind == "pond":
            self._arrive(h)
        elif lane.kind == "water" and lane.platform(h.y + 0.5, self.play_t) is None:
            self._die(h, "drown")

    def _arrive(self, h: Hopper):
        h.finished = True
        self.finished.append(h.slot)
        h.place = len(self.finished)
        h.x, h.y = POND_START + 1 + (h.place - 1) * 1.8, float(2 + h.slot * 3)
        if h.place == 1:
            self.deadline = min(self.deadline, self.play_t + LAST_CALL)

    def _die(self, h: Hopper, cause: str):
        if h.alive:
            h.alive, h.cause, h.hop = False, cause, None
            h.lives -= 1
            h.death_t = 1.2

    def score(self, h: Hopper) -> int:
        bonus = FINISH_BONUS[min(h.place, 4) - 1] + h.lives * 25 if h.finished else 0
        return h.max_col * 10 + bonus

    def _end(self):
        ranked = sorted(self.hoppers.values(), key=lambda h: (-self.score(h), h.slot))
        self.result = {"solo": len(ranked) == 1, "finished": [h.slot for h in ranked if h.finished],
                       "ranking": [(h.slot, self.score(h)) for h in ranked]}
        self.state, self.timer = "over", RESULT_TIME

    # ---- drawing

    def _build_bg(self):
        bg = np.zeros((GH * T, GW * T, 3), np.uint8)
        for c, lane in enumerate(self.lanes):
            x = c * T
            col = bg[:, x : x + T]
            b = lane.biome
            if lane.kind == "safe":
                col[:] = {"start": (40, 150, 60), "median": (35, 130, 55), "bank": (45, 150, 70), "oasis": (190, 165, 95),
                          "rock": (95, 95, 105), "barrier": (110, 110, 118)}[b]
                if b in ("start", "median", "bank"):
                    for y in range(3, GH * T, 11):
                        col[y : y + 2, (y * 7 + c * 5) % 6 : (y * 7 + c * 5) % 6 + 2] = (255, 230, 90) if (y + c) % 2 else (255, 130, 190)
                if b == "barrier":
                    col[:, 3:5] = (200, 200, 60)
            elif lane.kind == "road":
                col[:] = {"country": (70, 68, 62), "desert": (200, 165, 95), "suburb": (58, 58, 66), "city": (36, 36, 44)}[b]
                prev = self.lanes[c - 1] if c else None
                if prev is not None and prev.kind == "road":
                    line = (235, 200, 40) if prev.dir != lane.dir else (210, 210, 215)
                    for y in range(0, GH * T, 8):
                        col[y : y + 5, 0] = line
            elif lane.kind == "rail":
                col[:] = (205, 170, 105)
                col[:, 1] = col[:, 6] = (120, 120, 130)
                for y in range(0, GH * T, 6):
                    col[y : y + 2, 0:8] = (120, 80, 40)
            elif lane.kind == "water":
                col[:] = {"stream": (35, 105, 205), "lava": (150, 35, 10), "canal": (20, 75, 120)}[b]
                spark = (255, 140, 30) if b == "lava" else (90, 160, 240)
                for y in range(2, GH * T, 9):
                    col[(y + c * 3) % (GH * T), (c * 5 + y) % 7] = spark
            else:  # pond
                col[:] = (30, 140, 205)
                for y in range(4, GH * T, 14):
                    cv2.circle(bg, (x + 2 + (y + c * 5) % 5, y + (c % 3) * 2), 3, (40, 170, 70), -1)
        for y in range(0, GH * T, 8):  # checkered finish line at the edge of the pond
            bg[y : y + 4, POND_START * T : POND_START * T + 4] = (240, 240, 240)
            bg[y + 4 : y + 8, POND_START * T + 4 : POND_START * T + 8] = (240, 240, 240)
        self.bg = bg

    @staticmethod
    def _rect(canvas, x, y, w, h, color):
        x0, y0, x1, y1 = max(x, 0), max(y, OY), min(x + w, canvas.shape[1]), min(y + h, OY + GH * T)
        if x0 < x1 and y0 < y1:
            canvas[y0:y1, x0:x1] = color

    def _draw_object(self, canvas, c, a, b, style, off, direction, t):
        x, y, h = c * T, OY + int(round(a * T)), int(round((b - a) * T))
        r = self._rect
        if style in ("car", "bus", "truck", "moto"):
            color = (250, 205, 40) if style == "bus" else (150, 150, 160) if style == "truck" else CAR_COLORS[int(off * 7) % len(CAR_COLORS)]
            wd = 3 if style == "moto" else 6
            x0 = x + (T - wd) // 2
            r(canvas, x0, y + 1, wd, h - 2, color)
            front = y + h - 5 if direction > 0 else y + 1
            r(canvas, x0 + 1, front, wd - 2, 3, (25, 35, 60))
            if style == "truck":
                r(canvas, x0, y + 1 if direction < 0 else y + h - 9, wd, 8, (235, 110, 40))
            lights_y = y + h - 2 if direction > 0 else y + 1
            r(canvas, x0, lights_y, 1, 1, (255, 255, 200))
            r(canvas, x0 + wd - 1, lights_y, 1, 1, (255, 255, 200))
        elif style == "boulder":
            cv2.circle(canvas, (x + 4, y + h // 2), 3, (120, 105, 95), -1)
            cv2.circle(canvas, (x + 3, y + h // 2 - 1), 1, (160, 145, 130), -1)
        elif style == "log":
            r(canvas, x, y, T, h, (120, 75, 35))
            for k in range(0, h, 8):
                r(canvas, x, y + k, T, 1, (160, 105, 55))
            r(canvas, x + 1, y, 1, h, (95, 58, 28))
        elif style == "turtle":
            state = turtle_state(off, t)
            color = (60, 175, 75) if state == "up" else (180, 190, 90) if state == "warn" and int(t * 8) % 2 else (45, 130, 80)
            if state == "down":
                color = (25, 80, 130)
            for k in range(0, h, 8):
                cv2.circle(canvas, (x + 4, y + k + 4), 3, color, -1)
        elif style == "rock":
            r(canvas, x, y, T, h, (70, 55, 55))
            r(canvas, x + 1, y + 1, T - 2, h - 2, (100, 80, 78))
            r(canvas, x + 3, y + h // 2, 2, 1, (255, 150, 40))
        elif style == "train":
            r(canvas, x, y, T, h, (130, 130, 145))
            r(canvas, x + 2, y, 4, h, (180, 60, 60))
            for k in range(0, h, 9):
                r(canvas, x + 3, y + k, 2, 2, (255, 230, 120))

    def _draw_hopper(self, canvas, h: Hopper, bob: float = 0.0):
        px, py = int((h.x + 0.5) * T), int(OY + (h.y + 0.5) * T + bob)
        color = PLAYER_COLORS[h.slot]
        if not h.alive:
            cv2.circle(canvas, (px, py), 3, tuple(c // 3 for c in color), -1)
            cv2.line(canvas, (px - 3, py - 3), (px + 3, py + 3), (255, 255, 255), 1)
            cv2.line(canvas, (px - 3, py + 3), (px + 3, py - 3), (255, 255, 255), 1)
            return
        if h.invuln > 0 and int(self.t * 12) % 2:
            return
        cv2.circle(canvas, (px, py), 3, color, -1)
        fx, fy = h.face
        for side in (-1, 1):  # two eyes either side of the facing direction
            ex, ey = px + fx * 2 - fy * side, py + fy * 2 + fx * side
            if 0 <= ey < canvas.shape[0] and 0 <= ex < canvas.shape[1]:
                canvas[ey, ex] = (255, 255, 255)

    def render(self, canvas: np.ndarray):
        if self.state == "over" and not self.result["solo"]:
            return draw_podium(canvas, self.result["ranking"], self.t)
        canvas[:] = 0
        canvas[OY : OY + GH * T, : GW * T] = self.bg
        t = self.play_t if self.state in ("play", "over") else self.t
        for c, lane in enumerate(self.lanes):
            if lane.kind == "rail" and lane.warning(t) and int(t * 6) % 2:
                self._rect(canvas, c * T, OY, T, 4, (255, 40, 40))
                self._rect(canvas, c * T, OY + GH * T - 4, T, 4, (255, 40, 40))
            edge = lane.entering(t)
            if edge and int(t * 8) % 2:  # flashing arrow where a vehicle is about to appear
                y = OY if edge == "top" else OY + GH * T - 3
                self._rect(canvas, c * T + 1, y, 6, 3, (255, 70, 40))
                self._rect(canvas, c * T + 3, y + (0 if edge == "top" else 1), 2, 2, (255, 230, 60))
            for a, b, style, off in lane.intervals(t):
                self._draw_object(canvas, c, a, b, style, off, lane.dir, t)
        roster = self.hoppers if self.state != "lobby" else {s: Hopper(s) for s in self.joined}
        for s, h in sorted(roster.items()):
            self._draw_hopper(canvas, h, math.sin(self.t * 6 + s) * 1.5 if h.finished else 0.0)
        for s, h in sorted(roster.items()):
            pct = int(100 * h.max_col / (POND_START))
            label = f"P{s + 1} {'x' * max(0, h.lives)} {'DONE' if h.finished else str(min(100, pct)) + '%'}"
            text(canvas, label, 6 + s * 130, 13, 0.42, PLAYER_COLORS[s] if h.lives > 0 or h.finished else (110, 110, 110))
        if self.state == "play":
            left = max(0, int(self.deadline - self.play_t))
            text(canvas, f"TIME {left // 60}:{left % 60:02d}", self.width - 100, 13, 0.45, (255, 90, 90) if left < 15 else (255, 255, 255))
        if self.state == "lobby":
            canvas[OY:] = canvas[OY:] // 3
            center_text(canvas, "PANEL HOP", 56, 1.0, (255, 230, 0))
            center_text(canvas, "Cross every biome to the pond. Logs carry you over water.", 78, 0.5)
            center_text(canvas, self.lobby.message, 102, 0.8, (120, 255, 120) if self.joined else (255, 255, 255))
        elif self.state == "over":
            canvas[:] = canvas // 3
            fin = self.result["finished"]
            center_text(canvas, "MADE IT!" if fin else "GAME OVER", 70, 1.0, (120, 255, 120) if fin else (255, 80, 80))

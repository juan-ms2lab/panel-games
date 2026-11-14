"""Panel Flap: multiplayer flap-to-fly game that travels across the whole display.

Birds fly right through pipes. At the right edge play pauses and the whole screen scrolls over, so the
course covers many screens. The last bird flying wins.
"""

import math
import random

import numpy as np

from . import sprites
from .base import PLAYER_COLORS, Lobby, center_text, text
from .pad import A, UP

SW, SH = 768, 128
GROUND = 6
PIPE_W = 12
SPACING, FIRST, PER_PAGE = 100, 150, 6
START_X, STAGGER = 24, 16
GRAV, FLAP_V, MAX_FALL = 330.0, -105.0, 220.0
BASE_SPEED, SPEED_STEP, MAX_SPEED = 56.0, 4.0, 84.0
SCROLL_TIME = 1.6
TRIGGER = SW - 14
RESULT_TIME = 7.0
SKY = (12, 22, 64)

_BIRD_A = ["..yyyy....", ".yyyyyyee.", "yyyyyyyeko", "yyllyyyyoo", "yyllllyyo.", ".yyyyyyy..", "..yyyy...."]
_BIRD_B = ["..yyyy....", ".yyyyyyee.", "yyllyyyeko", "yyyyyyyyoo", "yyyyyyyyo.", ".yyyyyyy..", "..yyyy...."]


def _palette(color):
    return {"y": color, "l": tuple(int(c * 0.55) for c in color), "e": (255, 255, 255),
            "k": (10, 10, 10), "o": (255, 190, 40)}


class Bird:
    def __init__(self, slot: int):
        self.slot = slot
        self.sx = START_X + slot * STAGGER
        self.x, self.y, self.vy = float(self.sx), SH / 2.0, 0.0
        self.alive, self.score, self.prev = True, 0, 0
        self.next_pipe, self.died_at = 0, 0.0
        self.x0 = self.x


class Flap:
    id = "flap"
    title = "Panel Flap"
    description = "Fly through the pipes. At the right edge the whole screen scrolls over, so you travel every panel. Last bird flying wins."
    max_players = 4
    size = (768, 128)
    controls = "Tap A (or Up) to flap"

    def __init__(self, width: int, height: int, seed=None):
        if width < SW or height < SH:
            raise ValueError(f"Panel Flap needs a {SW}x{SH} display")
        self.width, self.height = width, height
        self.rng = random.Random(seed)
        self.t = 0.0
        self.lobby = Lobby()
        self.joined: set[int] = set()
        self.state = "lobby"
        self._sprites: dict = {}
        self._stars = [(self.rng.randrange(SW), self.rng.randrange(SH - GROUND), self.rng.choice((90, 160, 230)))
                       for _ in range(120)]
        self._reset_course()

    # ---- setup

    def _reset_course(self):
        self.birds: dict[int, Bird] = {}
        self.pipes: list[tuple[float, float, float]] = []
        self.page, self.cam, self.speed = 0, 0.0, BASE_SPEED
        self.last_gap_y = SH / 2
        self.scroll_t = 0.0
        self.result: dict = {}
        self.timer = 0.0
        self._make_page(0)
        self._make_page(1)

    def _make_page(self, page: int):
        gap = max(34.0, 54.0 - 3 * page)
        lo, hi = gap / 2 + 10, SH - GROUND - gap / 2 - 10
        for i in range(PER_PAGE):
            y = min(hi, max(lo, self.last_gap_y + self.rng.uniform(-45, 45)))
            self.pipes.append((page * SW + FIRST + i * SPACING, y, gap))
            self.last_gap_y = y

    def _start_round(self, roster):
        self._reset_course()
        self.birds = {s: Bird(s) for s in sorted(roster)}
        self.state = "play"

    # ---- update

    def info(self) -> dict:
        return {"state": self.state, "host": self.lobby.host}

    def update(self, dt: float, pads: list):
        self.t += dt
        joined = {i for i, k in enumerate(pads) if k is not None}
        if self.state == "lobby":
            self.joined = joined
            if self.lobby.update(joined, pads):
                self._start_round(joined)
        elif self.state == "play":
            self._play(dt, pads)
        elif self.state == "scroll":
            self._scroll(dt, pads)
        else:
            self._fall_dead(dt)
            self.timer -= dt
            if self.timer <= 0:
                self.state, self.joined = "lobby", joined
                self.lobby.reset()
                self._reset_course()

    def _kill(self, b: Bird):
        b.alive, b.died_at, b.vy = False, self.t, -60.0

    def _fall_dead(self, dt):
        floor = SH - GROUND - 3
        for b in self.birds.values():
            if not b.alive and b.y < floor:
                b.vy = min(MAX_FALL, b.vy + GRAV * dt)
                b.y = min(floor, b.y + b.vy * dt)

    def _play(self, dt, pads):
        page_pipes = self.pipes[self.page * PER_PAGE : (self.page + 1) * PER_PAGE]
        for b in self.birds.values():
            if not b.alive:
                continue
            keys = pads[b.slot] if b.slot < len(pads) else None
            if keys is None:
                self._kill(b)  # left the game
                continue
            if keys & (A | UP) and not b.prev & (A | UP):
                b.vy = FLAP_V
            b.prev = keys
            b.vy = min(MAX_FALL, b.vy + GRAV * dt)
            b.y += b.vy * dt
            b.x += self.speed * dt
            if b.y - 3 < 0:
                b.y, b.vy = 3.0, max(0.0, b.vy)
            hit = b.y + 3 >= SH - GROUND
            for px, gy, gap in page_pipes:
                if b.x + 4 > px and b.x - 4 < px + PIPE_W and (b.y - 3 < gy - gap / 2 or b.y + 3 > gy + gap / 2):
                    hit = True
            if hit:
                self._kill(b)
                continue
            while b.next_pipe < len(self.pipes) and self.pipes[b.next_pipe][0] + PIPE_W < b.x:
                b.score += 1
                b.next_pipe += 1
        self._fall_dead(dt)
        alive = [b for b in self.birds.values() if b.alive]
        if len(alive) <= (1 if len(self.birds) > 1 else 0):
            self._finish()
        elif any(b.x - self.cam >= TRIGGER for b in alive):
            self.state, self.scroll_t = "scroll", 0.0
            self.cam0 = self.cam
            for b in alive:
                b.x0 = b.x

    def _scroll(self, dt, pads):
        self.scroll_t = min(1.0, self.scroll_t + dt / SCROLL_TIME)
        s = self.scroll_t * self.scroll_t * (3 - 2 * self.scroll_t)
        self.cam = self.cam0 + SW * s
        for b in self.birds.values():
            if b.alive:
                b.x = b.x0 + s * (self.cam0 + SW + b.sx - b.x0)
                keys = pads[b.slot] if b.slot < len(pads) else None
                if keys is None:
                    self._kill(b)
        self._fall_dead(dt)
        if self.scroll_t >= 1.0:
            self.page += 1
            self.cam = self.page * SW
            self.speed = min(MAX_SPEED, BASE_SPEED + SPEED_STEP * self.page)
            self._make_page(self.page + 1)
            self.state = "play"
            if not any(b.alive for b in self.birds.values()):
                self._finish()

    def _finish(self):
        ranked = sorted(self.birds.values(), key=lambda b: (b.alive, b.died_at, b.score), reverse=True)
        top = ranked[0]
        self.result = {"winner": top.slot, "score": top.score, "page": self.page + 1, "solo": len(self.birds) == 1}
        self.state, self.timer = "over", RESULT_TIME

    # ---- drawing

    def _bird_sprite(self, slot, frame):
        key = (slot, frame)
        if key not in self._sprites:
            self._sprites[key] = sprites.make(_BIRD_A if frame == 0 else _BIRD_B, _palette(PLAYER_COLORS[slot]))
        return self._sprites[key]

    def _draw_world(self, canvas):
        canvas[:] = SKY
        for x, y, v in self._stars:
            canvas[y, x] = (v, v, v)
        canvas[SH - GROUND :, :] = (110, 70, 30)
        canvas[SH - GROUND : SH - GROUND + 2, :] = (70, 190, 70)
        for px, gy, gap in self.pipes:
            x = int(px - self.cam)
            if x < -20 or x > SW + 20:
                continue
            top, bottom = int(gy - gap / 2), int(gy + gap / 2)
            for y0, y1, cap_y in ((0, top, top - 4), (bottom, SH - GROUND, bottom)):
                self._rect(canvas, x, y0, PIPE_W, y1 - y0, (40, 170, 70))
                self._rect(canvas, x + 2, y0, 2, y1 - y0, (110, 230, 120))
                self._rect(canvas, x - 2, cap_y, PIPE_W + 4, 4, (60, 210, 90))

    @staticmethod
    def _rect(canvas, x, y, w, h, color):
        x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, canvas.shape[1]), min(y + h, canvas.shape[0])
        if x0 < x1 and y0 < y1:
            canvas[y0:y1, x0:x1] = color

    def render(self, canvas: np.ndarray):
        self._draw_world(canvas)
        frame = int(self.t * 8) % 2
        if self.state == "lobby":
            for s in sorted(self.joined):
                y = SH / 2 + 8 * math.sin(self.t * 3 + s)
                sprites.blit(canvas, self._bird_sprite(s, frame), START_X + s * STAGGER - 5, int(y) - 3)
            center_text(canvas, "PANEL FLAP", 40, 1.0, (255, 230, 0))
            center_text(canvas, "Tap A to flap. Last bird flying wins.", 62, 0.5)
            center_text(canvas, self.lobby.message, 92, 0.8, (120, 255, 120) if self.joined else (255, 255, 255))
            return
        for b in sorted(self.birds.values(), key=lambda b: b.alive):
            sx = int(b.x - self.cam)
            if -12 < sx < SW + 12:
                sprites.blit(canvas, self._bird_sprite(b.slot, frame if b.alive else 0), sx - 5, int(b.y) - 3,
                             flip_y=not b.alive)
        for b in self.birds.values():
            text(canvas, f"P{b.slot + 1} {b.score}", 6 + b.slot * 90, 12, 0.45,
                 PLAYER_COLORS[b.slot] if b.alive else (120, 120, 120))
        text(canvas, f"SCREEN {self.page + 1}", SW - 90, 12, 0.45)
        if self.state == "over":
            r = self.result
            if r["solo"]:
                center_text(canvas, "GAME OVER", 52, 1.0, (255, 80, 80))
            else:
                center_text(canvas, f"PLAYER {r['winner'] + 1} WINS", 52, 1.0, PLAYER_COLORS[r["winner"]])
            center_text(canvas, f"SCREEN {r['page']}   SCORE {r['score']}", 78, 0.5)

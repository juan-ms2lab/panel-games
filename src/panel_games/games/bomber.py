"""Panel Bomber: a Bomberman-style arena for up to four players across the whole display.

Drop bombs, blow up crates for power-ups, and be the last player standing. After a while the arena closes in
from the edges so rounds always end.
"""

import random

import cv2
import numpy as np

from . import sprites
from .base import PLAYER_COLORS, Lobby, Mover, advance, center_text, reverse, text
from .pad import A, DOWN, LEFT, RIGHT, UP

T = 8
GW, GH = 95, 13
OX, OY = 4, 20
DIRS = ((1, 0), (-1, 0), (0, -1), (0, 1))
KEY_DIRS = ((UP, (0, -1)), (DOWN, (0, 1)), (LEFT, (-1, 0)), (RIGHT, (1, 0)))
SPAWNS = [(1, 1), (GW - 2, GH - 2), (GW - 2, 1), (1, GH - 2)]
FUSE, FLAME_T = 2.2, 0.5
SOFT_DENSITY, POWERUP_CHANCE = 0.6, 0.3
BASE_SPEED, SPEED_STEP, MAX_SPEED = 5.0, 0.7, 8.5
SUDDEN_DEATH_AT, SUDDEN_STEP = 75.0, 0.15
RESULT_TIME = 7.0
POWERUPS = {"range": (255, 90, 40), "bomb": (80, 160, 255), "speed": (80, 255, 120)}

_BOMBER = ["..hhhh..", ".hhhhhh.", ".hkffkh.", ".hffffh.", "..bbbb..", ".bbbbbb.", ".bb..bb.", ".ll..ll."]


def _palette(color):
    return {"h": color, "b": tuple(int(c * 0.55) for c in color), "f": (255, 205, 160), "k": (20, 20, 20),
            "l": (120, 70, 30)}


def spiral(w: int, h: int) -> list[tuple[int, int]]:
    """Tiles in order from the outer edge inward, clockwise."""
    out, x0, y0, x1, y1 = [], 0, 0, w - 1, h - 1
    while x0 <= x1 and y0 <= y1:
        out += [(x, y0) for x in range(x0, x1 + 1)]
        out += [(x1, y) for y in range(y0 + 1, y1 + 1)]
        if y0 < y1:
            out += [(x, y1) for x in range(x1 - 1, x0 - 1, -1)]
        if x0 < x1:
            out += [(x0, y) for y in range(y1 - 1, y0, -1)]
        x0, y0, x1, y1 = x0 + 1, y0 + 1, x1 - 1, y1 - 1
    return out


class Bomber(Mover):
    def __init__(self, slot: int):
        super().__init__(*SPAWNS[slot])
        self.slot, self.want, self.prev = slot, (0, 0), 0
        self.alive, self.died_at = True, 0.0
        self.max_bombs, self.range, self.speed = 1, 2, BASE_SPEED


class Bomb:
    def __init__(self, tile, owner: Bomber):
        self.tile, self.owner, self.left, self.range = tile, owner.slot, FUSE, owner.range
        self.passable = {owner.slot}  # players may walk off a bomb they are standing on


class Bombers:
    id = "bomber"
    title = "Panel Bomber"
    description = "Blow up crates, grab power-ups and trap your friends. Last player standing wins."
    max_players = 4
    size = (768, 128)
    controls = "D-pad to move, A to drop a bomb"

    def __init__(self, width: int, height: int, seed=None):
        if width < OX * 2 + GW * T or height < OY + GH * T:
            raise ValueError(f"Panel Bomber needs a {OX * 2 + GW * T}x{OY + GH * T} display")
        self.width, self.height = width, height
        self.rng = random.Random(seed)
        self.t = 0.0
        self.lobby = Lobby()
        self.joined: set[int] = set()
        self._sprites: dict = {}
        self.new_round()

    # ---- setup

    def new_round(self, joined=None):
        self.state = "lobby"
        self.lobby.reset()
        self.joined = set(joined or ())
        self.grid = [[1 if x in (0, GW - 1) or y in (0, GH - 1) or (x % 2 == 0 and y % 2 == 0) else 0
                      for x in range(GW)] for y in range(GH)]
        keep_clear = set()
        for sx, sy in SPAWNS:  # an L of three open tiles at each corner so nobody starts boxed in
            dx, dy = (1 if sx < GW // 2 else -1), (1 if sy < GH // 2 else -1)
            keep_clear |= {(sx, sy), (sx + dx, sy), (sx, sy + dy)}
        for y in range(GH):
            for x in range(GW):
                if not self.grid[y][x] and (x, y) not in keep_clear and self.rng.random() < SOFT_DENSITY:
                    self.grid[y][x] = 2
        self.players: dict[int, Bomber] = {}
        self.bombs: dict[tuple, Bomb] = {}
        self.flames: dict[tuple, float] = {}
        self.powerups: dict[tuple, str] = {}
        self.spiral, self.sd_i, self.sd_acc = spiral(GW, GH), 0, 0.0
        self.play_t, self.timer, self.result = 0.0, 0.0, {}
        self._build_bg()

    def _start(self, roster):
        self.players = {s: Bomber(s) for s in sorted(roster)}
        self.state, self.play_t = "play", 0.0

    def info(self) -> dict:
        return {"state": self.state, "host": self.lobby.host}

    def _build_bg(self):
        self.bg = np.zeros((GH * T, GW * T, 3), np.uint8)
        for y in range(GH):
            for x in range(GW):
                self._draw_tile(x, y)

    def _draw_tile(self, tx, ty):
        x, y = tx * T, ty * T
        cell = self.bg[y : y + T, x : x + T]
        kind = self.grid[ty][tx]
        if kind == 1:
            cell[:] = (105, 105, 120)
            cell[0, :] = cell[:, 0] = (160, 160, 175)
            cell[-1, :] = cell[:, -1] = (60, 60, 72)
        elif kind == 2:
            cell[:] = (170, 105, 55)
            cell[0, :] = cell[-1, :] = cell[:, 0] = cell[:, -1] = (110, 65, 30)
            cell[3:5, 1:-1] = (120, 72, 35)
            cell[1:-1, 3:5] = (120, 72, 35)
        else:
            cell[:] = (16, 46, 28) if (tx + ty) % 2 else (20, 54, 32)

    # ---- helpers

    def _blocked(self, p: Bomber, tile) -> bool:
        x, y = tile
        if not (0 <= x < GW and 0 <= y < GH) or self.grid[y][x]:
            return True
        bomb = self.bombs.get(tile)
        return bomb is not None and p.slot not in bomb.passable

    def _choose(self, p: Bomber):
        d = p.want
        return d if d != (0, 0) and not self._blocked(p, (p.tx + d[0], p.ty + d[1])) else (0, 0)

    def _die(self, p: Bomber):
        if p.alive:
            p.alive, p.died_at = False, self.t

    def _detonate(self, first):
        queue = list(first)
        while queue:
            tile = queue.pop()
            bomb = self.bombs.pop(tile, None)
            if bomb is None:
                continue
            self.flames[tile] = FLAME_T
            for dx, dy in DIRS:
                for i in range(1, bomb.range + 1):
                    t = (tile[0] + dx * i, tile[1] + dy * i)
                    kind = self.grid[t[1]][t[0]]
                    if kind == 1:
                        break
                    self.flames[t] = FLAME_T
                    self.powerups.pop(t, None)
                    if kind == 2:
                        self.grid[t[1]][t[0]] = 0
                        self._draw_tile(*t)
                        if self.rng.random() < POWERUP_CHANCE:
                            self.powerups[t] = self.rng.choice(list(POWERUPS))
                        break
                    if t in self.bombs:
                        queue.append(t)
                        break

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
        due = []
        for tile, b in self.bombs.items():
            b.left -= dt
            if b.left <= 0:
                due.append(tile)
        self._detonate(due)
        self.flames = {t: left - dt for t, left in self.flames.items() if left - dt > 0}
        for p in self.players.values():
            if not p.alive:
                continue
            keys = pads[p.slot] if p.slot < len(pads) else None
            if keys is None:
                self._die(p)
                continue
            held = next((d for bit, d in KEY_DIRS if keys & bit), (0, 0))
            if held != (0, 0) and (p.dx, p.dy) != (0, 0) and held == (-p.dx, -p.dy):
                reverse(p)
            p.want = held
            if keys & A and not p.prev & A:
                tile = p.tile()
                mine = sum(1 for b in self.bombs.values() if b.owner == p.slot)
                if mine < p.max_bombs and tile not in self.bombs and not self.grid[tile[1]][tile[0]]:
                    bomb = self.bombs[tile] = Bomb(tile, p)
                    bomb.passable |= {q.slot for q in self.players.values() if q.alive and q.tile() == tile}
            p.prev = keys
            advance(p, p.speed * dt, self._choose)
            tile = p.tile()
            kind = self.powerups.pop(tile, None)
            if kind == "range":
                p.range = min(10, p.range + 1)
            elif kind == "bomb":
                p.max_bombs = min(6, p.max_bombs + 1)
            elif kind == "speed":
                p.speed = min(MAX_SPEED, p.speed + SPEED_STEP)
        for bomb in self.bombs.values():
            bomb.passable = {s for s in bomb.passable if s in self.players and self.players[s].alive
                             and self.players[s].tile() == bomb.tile}
        for p in self.players.values():
            if p.alive and p.tile() in self.flames:
                self._die(p)
        self._sudden_death(dt)
        alive = [p for p in self.players.values() if p.alive]
        if len(alive) <= (1 if len(self.players) > 1 else 0):
            self._finish(alive)

    def _sudden_death(self, dt):
        if self.play_t < SUDDEN_DEATH_AT:
            return
        self.sd_acc += dt
        while self.sd_acc >= SUDDEN_STEP and self.sd_i < len(self.spiral):
            self.sd_acc -= SUDDEN_STEP
            tile = self.spiral[self.sd_i]
            self.sd_i += 1
            if self.grid[tile[1]][tile[0]] == 1:
                continue
            self.grid[tile[1]][tile[0]] = 1
            self.bombs.pop(tile, None)
            self.powerups.pop(tile, None)
            self._draw_tile(*tile)
            for p in self.players.values():
                if p.alive and p.tile() == tile:
                    self._die(p)

    def _finish(self, alive):
        ranked = sorted(self.players.values(), key=lambda p: (p.alive, p.died_at), reverse=True)
        tie = not alive and len(ranked) > 1 and ranked[0].died_at == ranked[1].died_at
        self.result = {"winner": None if tie else ranked[0].slot, "solo": len(self.players) == 1,
                       "ranking": [p.slot for p in ranked]}
        self.state, self.timer = "over", RESULT_TIME

    # ---- drawing

    def _sprite(self, slot):
        if slot not in self._sprites:
            self._sprites[slot] = sprites.make(_BOMBER, _palette(PLAYER_COLORS[slot]))
        return self._sprites[slot]

    @staticmethod
    def _px(x, y):
        return int(OX + x * T), int(OY + y * T)

    def render(self, canvas: np.ndarray):
        canvas[:] = 0
        canvas[OY : OY + GH * T, OX : OX + GW * T] = self.bg
        for (tx, ty), kind in self.powerups.items():
            x, y = self._px(tx, ty)
            canvas[y + 1 : y + 7, x + 1 : x + 7] = POWERUPS[kind]
            canvas[y + 3 : y + 5, x + 3 : x + 5] = 255
        for tile, b in self.bombs.items():
            x, y = self._px(*tile)
            fast = b.left < 0.8 and int(self.t * 10) % 2
            cv2.circle(canvas, (x + 4, y + 4), 3, (255, 255, 255) if fast else (15, 15, 15), -1)
            canvas[y, x + 5 : x + 7] = (255, 200, 60) if int(self.t * 12) % 2 else (255, 80, 20)
        for tile in self.flames:
            x, y = self._px(*tile)
            canvas[y : y + T, x : x + T] = (255, 140, 20)
            canvas[y + 2 : y + 6, x + 2 : x + 6] = (255, 235, 90)
        roster = self.players if self.state != "lobby" else {s: Bomber(s) for s in self.joined}
        for s, p in sorted(roster.items()):
            if p.alive:
                px, py = p.pos()
                x, y = self._px(px, py)
                sprites.blit(canvas, self._sprite(s), x, y)
        for s, p in sorted(roster.items()):
            gray = (110, 110, 110)
            col = PLAYER_COLORS[s] if p.alive else gray
            text(canvas, f"P{s + 1} B{p.max_bombs} R{p.range} S{int((p.speed - BASE_SPEED) / SPEED_STEP + 0.5)}",
                 6 + s * 130, 14, 0.42, col)
        if self.state == "play" and self.play_t >= SUDDEN_DEATH_AT and int(self.t * 3) % 2:
            text(canvas, "SUDDEN DEATH", self.width - 130, 14, 0.45, (255, 80, 80))
        if self.state == "lobby":
            center_text(canvas, "PANEL BOMBER", 56, 1.0, (255, 230, 0))
            center_text(canvas, "D-pad to move, A drops a bomb. Last one standing wins.", 78, 0.5)
            center_text(canvas, self.lobby.message, 100, 0.8, (120, 255, 120) if self.joined else (255, 255, 255))
        elif self.state == "over":
            r = self.result
            canvas[:] = canvas // 3
            if r["solo"]:
                center_text(canvas, "GAME OVER", 66, 1.0, (255, 80, 80))
            elif r["winner"] is None:
                center_text(canvas, "DRAW", 66, 1.0, (255, 230, 0))
            else:
                center_text(canvas, f"PLAYER {r['winner'] + 1} WINS", 66, 1.0, PLAYER_COLORS[r["winner"]])

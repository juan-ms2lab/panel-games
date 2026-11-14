"""Panel Chomp: a multiplayer maze game that stretches across every panel.

Eat all the dots before the wisps catch you. One life each. If someone clears the maze the game ends on a
podium ranked by score; if every player is caught, the last one standing wins.
"""

import math
import random
from collections import deque

import cv2
import numpy as np

from . import sprites
from .base import PLAYER_COLORS, Lobby, Mover, advance, center_text, draw_podium, reverse, text
from .pad import DOWN, LEFT, RIGHT, UP

T = 8
CELLS_W, CELLS_H = 47, 6
GW, GH = 2 * CELLS_W + 1, 2 * CELLS_H + 1
OX, OY = 4, 20
RIGHT_D, LEFT_D, UP_D, DOWN_D = (1, 0), (-1, 0), (0, -1), (0, 1)
DIRS = (RIGHT_D, LEFT_D, UP_D, DOWN_D)
KEY_DIRS = ((UP, UP_D), (DOWN, DOWN_D), (LEFT, LEFT_D), (RIGHT, RIGHT_D))
PLAYER_SPEED, GHOST_SPEED, SCARED_SPEED, EYES_SPEED = 6.4, 5.2, 3.4, 10.0
SCARE_TIME = 7.0
HOME = (47, 7)
STARTS = [(9, 7), (85, 7), (29, 3), (65, 11)]
PELLETS = [(1, 1), (93, 1), (1, 11), (93, 11)]
GHOST_COLORS = [(255, 60, 60), (200, 80, 255), (60, 230, 200), (255, 170, 40)]
RESULT_TIME_PODIUM, RESULT_TIME_LAST = 14.0, 7.0
WALL, WALL_EDGE = (22, 34, 150), (90, 120, 255)

_GHOST = ["r.r..r.r", ".rrrrrr.", "rwwrrwwr", "rwbrrwbr", "rrrrrrrr", "rrrrrrrr", "rrrrrrrr", "r.rr.rr."]
_SCARED_MOUTH = ["........", "........", "........", "........", "........", "........", ".w.w.w..", "w.w.w.w."]


def make_maze(rng: random.Random) -> list[list[int]]:
    """Random maze on a cell grid (1 = wall). Depth-first carving, then a few extra openings for loops."""
    g = [[1] * GW for _ in range(GH)]
    start = (rng.randrange(CELLS_W), rng.randrange(CELLS_H))
    seen, stack = {start}, [start]
    g[2 * start[1] + 1][2 * start[0] + 1] = 0
    while stack:
        cx, cy = stack[-1]
        nbrs = [(cx + dx, cy + dy) for dx, dy in DIRS
                if 0 <= cx + dx < CELLS_W and 0 <= cy + dy < CELLS_H and (cx + dx, cy + dy) not in seen]
        if not nbrs:
            stack.pop()
            continue
        nx, ny = rng.choice(nbrs)
        seen.add((nx, ny))
        g[2 * ny + 1][2 * nx + 1] = 0
        g[cy + ny + 1][cx + nx + 1] = 0
        stack.append((nx, ny))
    for cy in range(CELLS_H):
        for cx in range(CELLS_W):
            for dx, dy in ((1, 0), (0, 1)):
                nx, ny = cx + dx, cy + dy
                if nx < CELLS_W and ny < CELLS_H and g[cy + ny + 1][cx + nx + 1] and rng.random() < 0.18:
                    g[cy + ny + 1][cx + nx + 1] = 0
    return g


class Chomper(Mover):
    def __init__(self, slot):
        super().__init__(*STARTS[slot])
        self.slot, self.want = slot, (0, 0)
        self.alive, self.score, self.died_at = True, 0, 0.0


class Wisp(Mover):
    def __init__(self, idx):
        super().__init__(*HOME)
        self.idx, self.state = idx, "home"
        self.release = 2.0 + idx * 4.0
        self.chase = (1.0, 0.75, 0.55, 0.4)[idx % 4]


class Chomp:
    id = "chomp"
    title = "Panel Chomp"
    description = "A maze across every panel. Eat the dots, dodge the wisps, grab power pellets. Last chomper standing wins, or climb the podium if you clear it."
    max_players = 4
    size = (768, 128)
    controls = "D-pad to steer"

    def __init__(self, width: int, height: int, seed=None):
        if width < OX * 2 + GW * T or height < OY + GH * T:
            raise ValueError(f"Panel Chomp needs a {OX * 2 + GW * T}x{OY + GH * T} display or larger")
        self.width, self.height = width, height
        self.rng = random.Random(seed)
        self.t = 0.0
        self._wisp_sprites: dict = {}
        self.joined: set[int] = set()
        self.lobby = Lobby()
        self.new_round()

    # ---- setup

    def new_round(self, joined=None):
        self.state = "lobby"
        self.lobby.reset()
        self.joined = set(joined or ())
        self.grid = make_maze(self.rng)
        reserved = set(STARTS) | {HOME}
        self.pellets = set(PELLETS)
        self.dots = {(x, y) for y in range(GH) for x in range(GW)
                     if not self.grid[y][x] and (x, y) not in reserved and (x, y) not in self.pellets}
        self.chompers: dict[int, Chomper] = {}
        self.wisps: list[Wisp] = []
        self.scare = 0.0
        self.play_t, self.timer, self.result = 0.0, 0.0, {}
        self._build_bg()

    def _start(self, roster):
        self.chompers = {s: Chomper(s) for s in sorted(roster)}
        self.wisps = [Wisp(i) for i in range(min(5, max(3, 2 + len(roster))))]
        self.state, self.play_t = "play", 0.0

    def _open(self, x, y):
        return 0 <= x < GW and 0 <= y < GH and not self.grid[y][x]

    def _build_bg(self):
        bg = np.zeros((GH * T, GW * T, 3), np.uint8)
        for ty in range(GH):
            for tx in range(GW):
                x, y = tx * T, ty * T
                if self.grid[ty][tx]:
                    bg[y : y + T, x : x + T] = WALL
                    if self._open(tx, ty - 1):
                        bg[y, x : x + T] = WALL_EDGE
                    if self._open(tx, ty + 1):
                        bg[y + T - 1, x : x + T] = WALL_EDGE
                    if self._open(tx - 1, ty):
                        bg[y : y + T, x] = WALL_EDGE
                    if self._open(tx + 1, ty):
                        bg[y : y + T, x + T - 1] = WALL_EDGE
                elif (tx, ty) in self.dots:
                    bg[y + 3 : y + 5, x + 3 : x + 5] = (255, 205, 130)
        self.bg = bg

    def _erase_dot(self, tx, ty):
        self.bg[ty * T + 3 : ty * T + 5, tx * T + 3 : tx * T + 5] = 0

    # ---- movement

    def _chomper_choose(self, c: Chomper):
        if c.want != (0, 0) and self._open(c.tx + c.want[0], c.ty + c.want[1]):
            return c.want
        if (c.dx, c.dy) != (0, 0) and self._open(c.tx + c.dx, c.ty + c.dy):
            return (c.dx, c.dy)
        return (0, 0)

    def _first_step(self, src, goals):
        """Direction of the first step on a shortest path from src to any goal tile, or None."""
        first, queue = {}, deque()
        for d in DIRS:
            n = (src[0] + d[0], src[1] + d[1])
            if self._open(*n):
                first[n] = d
                queue.append(n)
        while queue:
            cur = queue.popleft()
            if cur in goals:
                return first[cur]
            for d in DIRS:
                n = (cur[0] + d[0], cur[1] + d[1])
                if n != src and n not in first and self._open(*n):
                    first[n] = first[cur]
                    queue.append(n)
        return None

    def _wisp_choose(self, w: Wisp):
        here = (w.tx, w.ty)
        opts = [d for d in DIRS if self._open(here[0] + d[0], here[1] + d[1])]
        if not opts:
            return (0, 0)
        if w.state == "eyes":
            return self._first_step(here, {HOME}) or self.rng.choice(opts)
        back = (-w.dx, -w.dy)
        if len(opts) > 1 and back in opts and w.state == "scared":
            opts.remove(back)
        targets = {c.tile() for c in self.chompers.values() if c.alive}
        if w.state == "chase" and targets and self.rng.random() < w.chase:
            step = self._first_step(here, targets)
            if step:
                return step
        if len(opts) > 1 and back in opts:
            opts.remove(back)
        return self.rng.choice(opts)

    # ---- update

    def info(self) -> dict:
        return {"state": self.state, "host": self.lobby.host}

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

    def _die(self, c: Chomper):
        c.alive, c.died_at = False, self.t

    def _play(self, dt, pads):
        self.play_t += dt
        for c in self.chompers.values():
            if not c.alive:
                continue
            keys = pads[c.slot] if c.slot < len(pads) else None
            if keys is None:
                self._die(c)
                continue
            pressed = [d for bit, d in KEY_DIRS if keys & bit]
            if pressed:
                c.want = ([d for d in pressed if d != c.want] or pressed)[0]
                if (c.dx, c.dy) != (0, 0) and c.want == (-c.dx, -c.dy):
                    reverse(c)
            advance(c, PLAYER_SPEED * dt, self._chomper_choose)
            tile = c.tile()
            if tile in self.dots:
                self.dots.discard(tile)
                self._erase_dot(*tile)
                c.score += 10
            elif tile in self.pellets:
                self.pellets.discard(tile)
                c.score += 50
                self.scare = SCARE_TIME
                for w in self.wisps:
                    if w.state in ("chase", "scared"):
                        if w.state == "chase" and (w.dx, w.dy) != (0, 0):
                            reverse(w)
                        w.state = "scared"
        self.scare = max(0.0, self.scare - dt)
        for w in self.wisps:
            if w.state == "home" and self.play_t >= w.release:
                w.state = "chase"
            if w.state == "scared" and self.scare <= 0:
                w.state = "chase"
            speed = {"scared": SCARED_SPEED, "eyes": EYES_SPEED}.get(w.state, GHOST_SPEED + 0.15 * w.idx)
            if w.state != "home":
                advance(w, speed * dt, self._wisp_choose)
            if w.state == "eyes" and w.tile() == HOME:
                w.state = "chase"
            self._collide(w)
        self._check_end()

    def _collide(self, w: Wisp):
        if w.state not in ("chase", "scared"):
            return
        wx, wy = w.pos()
        for c in self.chompers.values():
            if not c.alive:
                continue
            cx, cy = c.pos()
            if (wx - cx) ** 2 + (wy - cy) ** 2 < 0.55:
                if w.state == "scared":
                    w.state, c.score = "eyes", c.score + 200
                    return
                self._die(c)

    def _check_end(self):
        cleared = not self.dots and not self.pellets
        everyone_dead = not any(c.alive for c in self.chompers.values())
        if not (cleared or everyone_dead):
            return
        ps = list(self.chompers.values())
        alive = sorted((c for c in ps if c.alive), key=lambda c: -c.score)
        fallen = sorted((c for c in ps if not c.alive), key=lambda c: (-c.died_at, -c.score))
        self.result = {"mode": "podium" if cleared else "last", "solo": len(ps) == 1,
                       "ranking": [(c.slot, c.score) for c in alive + fallen]}
        self.state = "over"
        self.timer = RESULT_TIME_PODIUM if cleared else RESULT_TIME_LAST

    # ---- drawing

    def _wisp_sprite(self, color, scared_look):
        key = (color, scared_look)
        if key not in self._wisp_sprites:
            pal = {"r": (40, 60, 255) if scared_look else color, "w": (255, 255, 255), "b": (10, 10, 10)}
            spr = sprites.make(_GHOST, pal)
            if scared_look:
                mouth = sprites.make(_SCARED_MOUTH, {"w": (255, 255, 255)})
                spr = np.where(mouth[:, :, 3:4] > 0, mouth, spr)
            self._wisp_sprites[key] = spr
        return self._wisp_sprites[key]

    def _center(self, e: Mover):
        x, y = e.pos()
        return int(OX + (x + 0.5) * T), int(OY + (y + 0.5) * T)

    def _draw_chomper(self, canvas, c: Chomper):
        cx, cy = self._center(c)
        color = PLAYER_COLORS[c.slot]
        cv2.circle(canvas, (cx, cy), 3, color, -1)
        moving = (c.dx, c.dy) != (0, 0)
        face = math.atan2(c.dy, c.dx) if moving else math.atan2(c.want[1], c.want[0]) if c.want != (0, 0) else 0.0
        open_angle = math.radians(10 + 35 * abs(math.sin(self.t * 12))) if moving else math.radians(25)
        pts = np.array([(cx, cy), (cx + 5 * math.cos(face - open_angle), cy + 5 * math.sin(face - open_angle)),
                        (cx + 5 * math.cos(face + open_angle), cy + 5 * math.sin(face + open_angle))], np.int32)
        cv2.fillConvexPoly(canvas, pts, (0, 0, 0))

    def render(self, canvas: np.ndarray):
        if self.state == "over" and self.result["mode"] == "podium":
            return draw_podium(canvas, self.result["ranking"], self.t)
        canvas[:] = 0
        canvas[OY : OY + GH * T, OX : OX + GW * T] = self.bg
        if int(self.t * 4) % 2 == 0:
            for px, py in self.pellets:
                cv2.circle(canvas, (OX + px * T + 4, OY + py * T + 4), 2, (255, 230, 200), -1)
        for w in self.wisps:
            cx, cy = self._center(w)
            if w.state == "eyes":
                canvas[cy - 2 : cy, cx - 3 : cx - 1] = 255
                canvas[cy - 2 : cy, cx + 1 : cx + 3] = 255
                continue
            scared = w.state == "scared"
            look = scared and not (self.scare < 2.0 and int(self.t * 6) % 2)
            sprites.blit(canvas, self._wisp_sprite(GHOST_COLORS[w.idx % 4], look), cx - 4, cy - 4)
        roster = self.chompers if self.state != "lobby" else {s: Chomper(s) for s in self.joined}
        for c in roster.values():
            if c.alive:
                self._draw_chomper(canvas, c)
        for s, c in sorted(roster.items()):
            text(canvas, f"P{s + 1} {c.score}", 6 + s * 100, 14, 0.45, PLAYER_COLORS[s] if c.alive else (110, 110, 110))
        if self.state != "lobby":
            text(canvas, f"DOTS {len(self.dots) + len(self.pellets)}", self.width - 110, 14, 0.45)
        if self.state == "lobby":
            center_text(canvas, "PANEL CHOMP", 50, 1.0, (255, 230, 0))
            center_text(canvas, "Steer with the D-pad. Last chomper standing wins.", 72, 0.5)
            center_text(canvas, self.lobby.message, 96, 0.8, (120, 255, 120) if self.joined else (255, 255, 255))
        elif self.state == "over":
            ranking = self.result["ranking"]
            canvas[:] = canvas // 3
            if self.result["solo"]:
                center_text(canvas, "GAME OVER", 56, 1.0, (255, 80, 80))
                center_text(canvas, f"SCORE {ranking[0][1]}", 82, 0.6)
            else:
                center_text(canvas, f"PLAYER {ranking[0][0] + 1} WINS", 56, 1.0, PLAYER_COLORS[ranking[0][0]])
                center_text(canvas, "LAST CHOMPER STANDING", 82, 0.6)

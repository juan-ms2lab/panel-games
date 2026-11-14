"""Panel Run: an original one-screen platformer for up to four players.

The whole level (96 x 16 tiles of 8 pixels) fits on a 768 x 128 display, so nothing scrolls.
"""

import cv2
import numpy as np

from . import sprites
from .pad import A, B, LEFT, RIGHT, UP

T = 8
LW, LH = 96, 16
WALK, RUN = 36.0, 62.0
ACCEL_GROUND, ACCEL_AIR, FRICTION = 420.0, 260.0, 520.0
JUMP = 205.0
GRAV_HELD, GRAV_CUT, GRAV_FALL, MAX_FALL = 650.0, 1500.0, 900.0, 260.0
ENEMY_SPEED = 14.0
SOLID = set("#B?UP")
SKY = (18, 52, 140)
RESET_DELAY = 5.0
HERO_COLORS = [(0, 200, 255), (255, 140, 0), (120, 255, 60), (255, 60, 200)]


def build_level() -> list[list[str]]:
    """Return the level grid. Legend: # ground, B brick, ? coin box, P pipe, o coin, e enemy, S start, F goal."""
    g = [[" "] * LW for _ in range(LH)]

    def put(x, y, ch):
        g[y][x] = ch

    def row(x, y, s):
        for i, ch in enumerate(s):
            put(x + i, y, ch)

    def pit(x, w):
        for xx in range(x, x + w):
            g[14][xx] = g[15][xx] = " "

    def pipe(x, h):
        for yy in range(14 - h, 14):
            put(x, yy, "P")
            put(x + 1, yy, "P")

    def stairs(x, n):
        for i in range(n):
            for yy in range(13 - i, 14):
                put(x + i, yy, "B")

    for x in range(LW):
        put(x, 14, "#")
        put(x, 15, "#")
    put(3, 13, "S")

    row(11, 10, "?B?")
    row(11, 8, "ooo")
    pipe(18, 2)
    put(24, 13, "e")
    pit(28, 2)
    row(30, 10, "BBBBB")
    row(30, 8, "ooooo")
    put(38, 13, "e")
    put(44, 13, "e")
    pipe(47, 3)
    pit(53, 3)
    for i, dy in enumerate((11, 10, 10, 11)):
        put(52 + i, dy - 1, "o")
    row(58, 10, "B?B?B")
    put(61, 13, "e")
    pipe(66, 2)
    put(70, 13, "e")
    pit(74, 3)
    stairs(80, 7)
    put(88, 13, "e")
    for yy in range(3, 14):
        put(92, yy, "F")
    return g


def _palette(color):
    return {"h": color, "b": tuple(int(c * 0.55) for c in color), "f": (255, 205, 160),
            "k": (20, 20, 20), "l": (110, 60, 25)}


_HERO_STAND = ["..hhhh..", ".hhhhhh.", ".ffffff.", ".fkffkf.", ".ffffff.", "..bbbb..",
               ".bbbbbb.", "bbbbbbbb", ".bb..bb.", ".bb..bb.", ".ll..ll.", "lll..lll"]
_HERO_RUN = ["..hhhh..", ".hhhhhh.", ".ffffff.", ".fkffkf.", ".ffffff.", "..bbbb..",
             ".bbbbbb.", "bbbbbbbb", "..bbbb..", ".bb..bb.", "bb....ll", "l.....ll"]
_HERO_JUMP = ["..hhhh..", ".hhhhhh.", ".ffffff.", ".fkffkf.", ".ffffff.", "b.bbbb.b",
              "bbbbbbbb", "b.bbbb.b", "..bbbb..", ".bb..bb.", ".ll..ll.", ".l....l."]
_ENEMY = [["..gggg..", ".gggggg.", "gwwggwwg", "gwbggbwg", "gggggggg", ".gggggg.", ".gg..gg.", "gg....gg"],
          ["..gggg..", ".gggggg.", "gwwggwwg", "gwbggbwg", "gggggggg", ".gggggg.", "..gggg..", ".gg..gg."]]
_ENEMY_PAL = {"g": (170, 90, 40), "w": (255, 255, 255), "b": (10, 10, 10)}
_COIN = ["..yy..", ".yyyy.", "yyywyy", "yyyyyy", "yyyyyy", "yyyyyy", ".yyyy.", "..yy.."]
_COIN_PAL = {"y": (255, 210, 0), "w": (255, 255, 200)}
_GROUND_TOP = ["gggggggg", "GGGGGGGG", "dddddddd", "dDddddDd", "dddddddd", "ddDddddd", "dddddDdd", "dddddddd"]
_DIRT = ["dddddddd", "dDddddDd", "dddddddd", "ddDddddd", "dddddDdd", "dddddddd", "dDddddDd", "dddddddd"]
_GROUND_PAL = {"g": (60, 200, 60), "G": (30, 140, 40), "d": (150, 90, 40), "D": (110, 60, 25)}
_BRICK = ["rrrmrrrr", "rrrmrrrr", "rrrmrrrr", "mmmmmmmm", "rmrrrrrr", "rmrrrrrr", "rmrrrrrr", "mmmmmmmm"]
_BRICK_PAL = {"r": (200, 80, 40), "m": (90, 40, 20)}
_BOX = ["oooooooo", "oyywwyyo", "oyyyywyo", "oyyywyyo", "oyyyyyyo", "oyyywyyo", "oyyyyyyo", "oooooooo"]
_BOX_PAL = {"o": (150, 90, 0), "y": (255, 200, 0), "w": (255, 255, 255)}
_USED = ["dddddddd", "dkkkkkkd", "dkkkkkkd", "dkkkkkkd", "dkkkkkkd", "dkkkkkkd", "dkkkkkkd", "dddddddd"]
_USED_PAL = {"d": (40, 40, 40), "k": (80, 80, 80)}


class Hero:
    w, h = 6, 12

    def __init__(self, slot: int, sx: float, sy: float):
        self.slot, self.sx, self.sy = slot, sx, sy
        self.lives, self.score = 3, 0
        self.prev = 0
        self.spawn()

    def spawn(self):
        self.x, self.y = self.sx + self.slot * 8, self.sy
        self.vx = self.vy = 0.0
        self.on_ground, self.alive = False, True
        self.face, self.invuln = 1, 1.5
        self.respawn = self.jump_buf = self.coyote = 0.0


class Enemy:
    w, h = 6, 7

    def __init__(self, x, y):
        self.x, self.y = x, y
        self.vy, self.dir, self.squish, self.dead = 0.0, -1, 0.0, False


class Platformer:
    id = "platformer"
    title = "Panel Run"
    description = "Race to the flag. Run, jump, stomp the walkers and grab coins. The whole level is on screen."
    max_players = 4
    size = (768, 128)
    controls = "Left / Right to move, A to jump (hold for higher), B to run"

    def __init__(self, width: int, height: int):
        if width < LW * T or height < LH * T:
            raise ValueError(f"Panel Run needs a {LW * T}x{LH * T} display")
        self.width, self.height = width, height
        self.heroes: dict[int, Hero] = {}
        self.t = 0.0
        self.reset()

    # ---- setup

    def reset(self):
        self.grid = build_level()
        self.enemies, self.pops = [], []
        self.start = (24.0, 14 * T - Hero.h)
        for ty, line in enumerate(self.grid):
            for tx, ch in enumerate(line):
                if ch == "e":
                    self.enemies.append(Enemy(tx * T + 1, ty * T + 1))
                    line[tx] = " "
                elif ch == "S":
                    self.start = (tx * T + 1.0, 14 * T - Hero.h)
                    line[tx] = " "
        self.state, self.timer, self.winner = "play", 0.0, None
        self._build_bg()
        for h in self.heroes.values():
            h.lives, h.score, h.prev = 3, 0, 0
            h.sx, h.sy = self.start
            h.spawn()
        self._hero_sprites = {}

    def _build_bg(self):
        sky = np.zeros((LH * T, LW * T, 3), np.uint8)
        sky[:] = SKY
        for cx, cy in ((40, 22), (150, 30), (300, 20), (430, 28), (560, 22), (690, 30)):
            for dx, dy, r in ((0, 0, 7), (-9, 3, 5), (9, 3, 5), (0, 4, 6)):
                cv2.circle(sky, (cx + dx, cy + dy), r, (215, 230, 255), -1)
        self.sky, self.bg = sky, sky.copy()
        self._sprites = {k: sprites.make(rows, pal) for k, (rows, pal) in {
            "top": (_GROUND_TOP, _GROUND_PAL), "dirt": (_DIRT, _GROUND_PAL), "brick": (_BRICK, _BRICK_PAL),
            "box": (_BOX, _BOX_PAL), "used": (_USED, _USED_PAL), "coin": (_COIN, _COIN_PAL)}.items()}
        self._enemy_sprites = [sprites.make(r, _ENEMY_PAL) for r in _ENEMY]
        for ty in range(LH):
            for tx in range(LW):
                self._draw_tile(tx, ty)

    def _cell(self, tx, ty):
        return self.grid[ty][tx] if 0 <= tx < LW and 0 <= ty < LH else " "

    def _draw_tile(self, tx, ty):
        x, y = tx * T, ty * T
        self.bg[y : y + T, x : x + T] = self.sky[y : y + T, x : x + T]
        ch, s = self.grid[ty][tx], self._sprites
        if ch == "#":
            sprites.blit(self.bg, s["dirt"] if self._cell(tx, ty - 1) in SOLID else s["top"], x, y)
        elif ch == "B":
            sprites.blit(self.bg, s["brick"], x, y)
        elif ch == "?":
            sprites.blit(self.bg, s["box"], x, y)
        elif ch == "U":
            sprites.blit(self.bg, s["used"], x, y)
        elif ch == "o":
            sprites.blit(self.bg, s["coin"], x + 1, y)
        elif ch == "P":
            self._draw_pipe(tx, ty, x, y)
        elif ch == "F":
            self._draw_goal(tx, ty, x, y)

    def _draw_pipe(self, tx, ty, x, y):
        left = self._cell(tx - 1, ty) != "P"
        cell = self.bg[y : y + T, x : x + T]
        cell[:] = (30, 170, 60)
        if left:
            cell[:, 1:3] = (100, 230, 120)
        else:
            cell[:, 6:] = (15, 100, 35)
        if self._cell(tx, ty - 1) != "P":
            cell[:3] = (50, 200, 80)
            cell[3] = (15, 100, 35)

    def _draw_goal(self, tx, ty, x, y):
        cell = self.bg[y : y + T, x : x + T]
        cell[:, 3:5] = (210, 210, 210)
        if self._cell(tx, ty - 1) != "F":
            cell[0:3, 2:6] = (255, 200, 0)
            for i, wd in enumerate((3, 3, 2, 1)):
                cell[3 + i, 5 : 5 + wd] = (230, 40, 40)

    # ---- physics

    def _solid(self, tx, ty):
        if tx < 0 or tx >= LW:
            return True
        return 0 <= ty < LH and self.grid[ty][tx] in SOLID

    def _move(self, e, dx, dy):
        """Move a box with tile collision. Returns (hit_x, hit_y, bumped tiles); hit_y is 1 landed, -1 head bump."""
        hit_x, hit_y, bumps = False, 0, []
        e.x += dx
        if dx:
            top, bottom = int(e.y // T), int((e.y + e.h - 0.01) // T)
            tx = int((e.x + e.w - 0.01) // T) if dx > 0 else int(e.x // T)
            if any(self._solid(tx, ty) for ty in range(top, bottom + 1)):
                e.x = tx * T - e.w if dx > 0 else (tx + 1) * T
                hit_x = True
        e.y += dy
        if dy:
            left, right = int(e.x // T), int((e.x + e.w - 0.01) // T)
            ty = int((e.y + e.h - 0.01) // T) if dy > 0 else int(e.y // T)
            hits = [tx for tx in range(left, right + 1) if self._solid(tx, ty)]
            if hits:
                if dy > 0:
                    e.y, hit_y = ty * T - e.h, 1
                else:
                    e.y, hit_y, bumps = (ty + 1) * T, -1, [(tx, ty) for tx in hits]
        return hit_x, hit_y, bumps

    def _tiles(self, e):
        for ty in range(int(e.y // T), int((e.y + e.h - 0.01) // T) + 1):
            for tx in range(int(e.x // T), int((e.x + e.w - 0.01) // T) + 1):
                yield tx, ty

    def _update_hero(self, h, dt, keys):
        h.invuln = max(0.0, h.invuln - dt)
        if not h.alive:
            h.vy += GRAV_FALL * dt
            h.y += h.vy * dt
            h.respawn -= dt
            if h.respawn <= 0 and h.lives > 0:
                h.spawn()
            return
        held = bool(keys & (A | UP))
        h.jump_buf = 0.12 if held and not h.prev & (A | UP) else max(0.0, h.jump_buf - dt)
        h.coyote = 0.08 if h.on_ground else max(0.0, h.coyote - dt)
        h.prev = keys
        direction = (1 if keys & RIGHT else 0) - (1 if keys & LEFT else 0)
        target = direction * (RUN if keys & B else WALK)
        accel = (ACCEL_GROUND if h.on_ground else ACCEL_AIR) if direction else (FRICTION if h.on_ground else 60.0)
        h.vx = min(target, h.vx + accel * dt) if h.vx < target else max(target, h.vx - accel * dt)
        if direction:
            h.face = direction
        if h.jump_buf > 0 and h.coyote > 0:
            h.vy, h.jump_buf, h.coyote, h.on_ground = -JUMP, 0.0, 0.0, False
        gravity = GRAV_FALL if h.vy >= 0 else (GRAV_HELD if held else GRAV_CUT)
        h.vy = min(MAX_FALL, h.vy + gravity * dt)
        hx, hy, bumps = self._move(h, h.vx * dt, h.vy * dt)
        if hx:
            h.vx = 0.0
        h.on_ground = hy == 1
        if hy:
            h.vy = 0.0
        center = int((h.x + h.w / 2) // T)
        for tx, ty in bumps:
            if tx == center and self.grid[ty][tx] == "?":
                self.grid[ty][tx] = "U"
                self._draw_tile(tx, ty)
                self.pops.append([tx * T + 1.0, ty * T - 8.0, 0.45])
                h.score += 10

    def _update_enemies(self, dt):
        for e in self.enemies:
            if e.squish > 0:
                e.squish -= dt
                e.dead = e.squish <= 0
                continue
            e.vy = min(MAX_FALL, e.vy + GRAV_FALL * dt)
            hx, hy, _ = self._move(e, e.dir * ENEMY_SPEED * dt, e.vy * dt)
            if hx:
                e.dir = -e.dir
            if hy:
                e.vy = 0.0
        self.enemies = [e for e in self.enemies if not e.dead and e.y < LH * T + 16]
        for p in self.pops:
            p[1] -= 30 * dt
            p[2] -= dt
        self.pops = [p for p in self.pops if p[2] > 0]

    def _kill(self, h):
        h.alive, h.vy, h.vx, h.respawn = False, -170.0, 0.0, 1.4
        h.lives -= 1

    def _interact(self):
        for h in self.heroes.values():
            if not h.alive:
                continue
            if h.y > LH * T + 4:
                self._kill(h)
                continue
            for tx, ty in self._tiles(h):
                ch = self._cell(tx, ty)
                if ch == "o":
                    self.grid[ty][tx] = " "
                    self._draw_tile(tx, ty)
                    h.score += 10
                elif ch == "F" and self.state == "play":
                    h.score += 500
                    self.state, self.timer, self.winner = "won", RESET_DELAY, h.slot
            for e in self.enemies:
                if e.squish > 0 or h.x + h.w <= e.x or e.x + e.w <= h.x or h.y + h.h <= e.y or e.y + e.h <= h.y:
                    continue
                if h.vy > 0 and (h.y + h.h) - e.y < 6:
                    e.squish, h.vy, h.score = 0.35, -150.0, h.score + 100
                elif h.invuln <= 0:
                    self._kill(h)
                    break

    def update(self, dt: float, pads: list):
        self.t += dt
        for slot, keys in enumerate(pads):
            if keys is None:
                self.heroes.pop(slot, None)
            elif slot not in self.heroes:
                self.heroes[slot] = Hero(slot, *self.start)
        if self.state != "play":
            self.timer -= dt
            if self.timer <= 0:
                self.reset()
            return
        if not self.heroes:
            return
        for slot, h in self.heroes.items():
            self._update_hero(h, dt, pads[slot] or 0)
        self._update_enemies(dt)
        self._interact()
        if all(h.lives <= 0 for h in self.heroes.values()):
            self.state, self.timer = "lost", RESET_DELAY

    # ---- drawing

    def _text(self, canvas, text, x, y, scale, color, center=False, box=False):
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        if center:
            x = (self.width - tw) // 2
        if box:
            cv2.rectangle(canvas, (x - 4, y - th - 4), (x + tw + 4, y + 5), (0, 0, 0), -1)
        cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)

    def _hero_sprite(self, h):
        if not h.on_ground:
            key = "jump"
        elif abs(h.vx) > 6:
            key = "run" if int(self.t * 10) % 2 else "stand"
        else:
            key = "stand"
        cache = self._hero_sprites.setdefault((h.slot, key), None)
        if cache is None:
            rows = {"stand": _HERO_STAND, "run": _HERO_RUN, "jump": _HERO_JUMP}[key]
            cache = self._hero_sprites[(h.slot, key)] = sprites.make(rows, _palette(HERO_COLORS[h.slot]))
        return cache

    def render(self, canvas: np.ndarray):
        canvas[:] = 0
        canvas[: LH * T, : LW * T] = self.bg
        coin = self._sprites["coin"]
        for x, y, _ in self.pops:
            sprites.blit(canvas, coin, int(x), int(y))
        for e in self.enemies:
            frame = self._enemy_sprites[int(self.t * 4) % 2]
            if e.squish > 0:
                sprites.blit(canvas, frame[4:], int(e.x) - 1, int(e.y) + 3)
            else:
                sprites.blit(canvas, frame, int(e.x) - 1, int(e.y) - 1, flip_x=e.dir > 0)
        for slot, h in sorted(self.heroes.items()):
            if h.y < self.height and not (h.alive and h.invuln > 0 and int(self.t * 12) % 2):
                sprites.blit(canvas, self._hero_sprite(h), int(h.x) - 1, int(h.y), flip_x=h.face < 0, flip_y=not h.alive)
        for slot, h in sorted(self.heroes.items()):
            self._text(canvas, f"P{slot + 1} x{h.lives} {h.score}", 6 + slot * 120, 10, 0.4, HERO_COLORS[slot])
        if not self.heroes:
            self._text(canvas, "JOIN ON YOUR PHONE", 0, 62, 0.9, (255, 255, 255), center=True, box=True)
        elif self.state == "won":
            self._text(canvas, "LEVEL CLEAR", 0, 52, 1.0, (255, 230, 0), center=True, box=True)
            self._text(canvas, f"PLAYER {self.winner + 1} WINS", 0, 72, 0.6, HERO_COLORS[self.winner], center=True, box=True)
        elif self.state == "lost":
            self._text(canvas, "GAME OVER", 0, 62, 1.0, (255, 60, 60), center=True, box=True)

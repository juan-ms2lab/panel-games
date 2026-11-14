"""El Birdo: single-player elbirdo for a wide, short strip (768x32, or taller: everything scales by h/32).

The screen is static: the bird flies left to right, and each level is one screen-width of pipes. 16 themed
stages of LEVELS_PER_STAGE levels, then endless. Stages introduce moving gaps, coins, darkness, flipped
gravity, wind and low gravity. Losing a life restarts the current level.

With two stacked displays (h >= 64) the bottom half is the underworld: the bird can drop down and becomes
skeleton birdo; obstacles there are magma pillars and hanging fire whose flames grow and shrink in sync,
slowly moving the hole.

Ported from the standalone original (bytearray canvas) to the games interface: update(dt, pads) and
render(canvas) on an (h, w, 3) RGB numpy array. A or Up flaps, START pauses, Left/Right pick the starting
stage (among those unlocked) before the first flap of a new game.
"""

import math
import random

import numpy as np

from .pad import A, LEFT, RIGHT, START, UP

LEVELS_PER_STAGE = 3
FIRST_PIPE_X = 60
PIPE_W = 6
START_LIVES = 3
MAX_LIVES = 5
COINS_PER_LIFE = 20
GRAVITY = 70.0
FLAP_V = 22.0
MAX_V = 36.0

# name, speed px/s, gap px, spacing (min, max), move amp px, move freq Hz, flags, theme
STAGES = [
    ("MEADOW", 22, 15, (70, 110), 0, 0.0, "", "day"),
    ("HILLS", 50, 13, (65, 100), 0, 0.0, "", "day"),
    ("BREEZE", 52, 13, (60, 95), 3, 0.5, "", "day"),
    ("SUNSET", 55, 12, (60, 90), 4, 0.7, "coins", "sunset"),
    ("CANYON", 60, 12, (55, 85), 2, 0.5, "coins", "canyon"),
    ("NIGHT", 56, 12, (60, 90), 3, 0.6, "dark coins", "night"),
    ("UPSIDE DOWN", 54, 13, (70, 100), 0, 0.0, "flip", "dusk"),
    ("STORM", 62, 12, (55, 85), 4, 0.9, "wind", "storm"),
    ("CAVES", 60, 11, (50, 80), 3, 0.8, "dark coins", "cave"),
    ("MIRROR", 60, 12, (60, 90), 3, 0.8, "flip coins", "dusk"),
    ("RUSH", 78, 12, (60, 90), 2, 0.6, "coins", "day"),
    ("GLACIER", 66, 11, (50, 80), 5, 1.1, "", "ice"),
    ("ABYSS", 68, 11, (50, 78), 4, 1.0, "dark wind", "night"),
    ("TEMPEST", 72, 10, (48, 75), 5, 1.3, "wind coins", "storm"),
    ("ORBIT", 74, 10, (45, 72), 5, 1.4, "lowg coins", "space"),
    ("FINALE", 80, 10, (42, 68), 6, 1.5, "dark wind", "space"),
]

THEMES = {
    "day": dict(top=(30, 90, 190), bot=(130, 190, 255), pipe=(40, 190, 60), hill=(50, 140, 60), deco="clouds"),
    "sunset": dict(top=(90, 30, 90), bot=(255, 130, 50), pipe=(50, 160, 70), hill=(80, 40, 70), deco="clouds"),
    "canyon": dict(top=(160, 70, 30), bot=(255, 170, 100), pipe=(140, 80, 40), hill=(120, 50, 25), deco=""),
    "night": dict(top=(2, 2, 15), bot=(15, 20, 55), pipe=(40, 140, 60), hill=(10, 20, 35), deco="stars"),
    "dusk": dict(top=(50, 30, 110), bot=(190, 100, 140), pipe=(60, 170, 90), hill=(60, 40, 90), deco="stars"),
    "storm": dict(top=(30, 30, 45), bot=(85, 85, 105), pipe=(70, 130, 80), hill=(40, 45, 50), deco="rain"),
    "cave": dict(top=(15, 10, 6), bot=(45, 30, 20), pipe=(110, 90, 70), hill=(30, 20, 12), deco=""),
    "ice": dict(top=(110, 160, 230), bot=(215, 235, 255), pipe=(140, 210, 255), hill=(180, 210, 235), deco="clouds"),
    "space": dict(top=(0, 0, 5), bot=(12, 0, 30), pipe=(160, 80, 220), hill=(25, 10, 40), deco="stars"),
}

BIRD = [".yyy.", "yywky", "yyyoo", ".yy.."]
BIRD_FLAP = [".yyy.", "lywky", "yyyoo", ".yy.."]
BIRD_PAL = {"y": (255, 215, 30), "w": (255, 255, 255), "k": (0, 0, 0), "o": (255, 110, 0), "l": (255, 245, 170)}
HEART = [".r.r.", "rrrrr", ".rrr.", "..r.."]
SKEL = [".www.", "wwkrw", "wwwgg", ".w.w."]
SKEL_FLAP = [".www.", "lwkrw", "wwwgg", ".w.w."]
SKEL_PAL = {"w": (225, 225, 210), "k": (0, 0, 0), "r": (255, 40, 0), "g": (150, 150, 140), "l": (255, 255, 255)}
FLAME = [(255, 255, 160), (255, 220, 60), (255, 150, 20), (230, 70, 0), (160, 20, 0)]
TITLE, TITLE2 = "EL BIRDO", "RESURRECTION"
MELT_AT, RISE_AT, PROMPT_AT = 3.0, 4.5, 7.0  # intro timeline, seconds
UNDER_BONUS = 2  # extra points per column passed in the underworld

# ---- tiny 3x5 bitmap font

GLYPHS = {
    "0": "111101101101111", "1": "010110010010111", "2": "111001111100111",
    "3": "111001111001111", "4": "101101111001001", "5": "111100111001111",
    "6": "111100111101111", "7": "111001001001001", "8": "111101111101111",
    "9": "111101111001111",
    "A": "010101111101101", "B": "110101110101110", "C": "011100100100011",
    "D": "110101101101110", "E": "111100110100111", "F": "111100110100100",
    "G": "011100101101011", "H": "101101111101101", "I": "111010010010111",
    "J": "001001001101010", "K": "101101110101101", "L": "100100100100111",
    "M": "101111111101101", "N": "110101101101101", "O": "010101101101010",
    "P": "110101110100100", "Q": "010101101110011", "R": "110101110101101",
    "S": "011100010001110", "T": "111010010010010", "U": "101101101101111",
    "V": "101101101101010", "W": "101101111111101", "X": "101101010101101",
    "Y": "101101010010010", "Z": "111001010100111",
    " ": "000000000000000", "!": "010010010000010", ":": "000010000010000",
    "-": "000000111000000", ".": "000000000000010", "/": "001001010100100",
    "?": "110001010000010", "'": "010010000000000", "+": "000010111010000",
}


def text_width(text, scale=1):
    return max(0, len(text) * 4 * scale - scale)


def draw_text(cv, x, y, text, color, scale=1, shadow=None):
    if shadow:
        draw_text(cv, x + scale, y + scale, text, shadow, scale)
    for ch in text.upper():
        g = GLYPHS.get(ch, GLYPHS["?"])
        for i, bit in enumerate(g):
            if bit == "1":
                cv.rect(x + (i % 3) * scale, y + (i // 3) * scale, scale, scale, color)
        x += 4 * scale


# ---- drawing on a numpy RGB canvas, same rounding as the original bytearray canvas

class Cv:
    def __init__(self, a: np.ndarray):
        self.a = a
        self.h, self.w = a.shape[:2]

    def clear(self, c=(0, 0, 0)):
        self.a[:] = c

    def rect(self, x, y, w, h, c):
        w, h = round(w), round(h)
        x0, y0 = max(0, math.floor(x)), max(0, math.floor(y))
        x1, y1 = min(self.w, math.floor(x) + w), min(self.h, math.floor(y) + h)
        if x1 > x0 and y1 > y0:
            self.a[y0:y1, x0:x1] = c

    def px(self, x, y, c):
        x, y = math.floor(x), math.floor(y)
        if 0 <= x < self.w and 0 <= y < self.h:
            self.a[y, x] = c

    def sprite(self, x, y, rows, palette, flip_y=False, scale=1):
        if flip_y:
            rows = rows[::-1]
        for dy, row in enumerate(rows):
            for dx, ch in enumerate(row):
                if ch in palette:
                    self.rect(x + dx * scale, y + dy * scale, scale, scale, palette[ch])


def dim(c, f):
    return (int(c[0] * f), int(c[1] * f), int(c[2] * f))


def mix(a, b, t):
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def stage_params(i):
    if i < len(STAGES):
        name, speed, gap, spacing, amp, freq, flags, theme = STAGES[i]
    else:
        k = i - len(STAGES) + 1
        name, speed, gap, spacing, amp, freq, flags, theme = STAGES[-1]
        name = f"ENDLESS {k}"
        speed += 3 * k
        gap = max(8, gap - k // 3)
        flags = random.Random(i).choice(["dark", "wind", "flip", "lowg coins", "dark wind", "coins"])
        theme = random.Random(i * 7).choice(list(THEMES))
    return dict(name=name, speed=speed, gap=gap, spacing=spacing, amp=amp, freq=freq,
                flags=set(flags.split()), theme=theme)


class Pipe:
    __slots__ = ("x", "base", "gap", "amp", "freq", "phase", "passed")

    def center(self, t, h):
        c = self.base + self.amp * math.sin(2 * math.pi * self.freq * t + self.phase)
        lo, hi = self.gap / 2 + 2, h - self.gap / 2 - 2
        return min(hi, max(lo, c))


class ElBirdo:
    id = "elbirdo"
    title = "El Birdo"
    description = ("One bird, 16 themed stages then endless: moving gaps, coins, darkness, flipped gravity, wind. "
                   "On a 64-pixel-tall display, drop into the underworld as skeleton birdo between fire columns.")
    max_players = 1
    size = [(768, 128), (768, 64), (768, 32)]
    controls = "A (or Up) to flap, START to pause, Left / Right pick the starting stage"
    record = {"best": 0, "unlocked": 1}  # kept for the life of the server, shared by every game

    def __init__(self, width: int, height: int, seed=None):
        if width < 128 or height < 32:
            raise ValueError("El Birdo needs at least 128x32")
        self.width, self.height = width, height
        self.w, self.h = width, height
        self.rng = random.Random(seed)
        self.split = height // 2 if height >= 64 else height  # rows above this are sky; below is the underworld
        self.k = k = max(1, self.split // 32)
        self.gravity, self.flap_v, self.max_v, self.pipe_w = GRAVITY * k, FLAP_V * k, MAX_V * k, PIPE_W * k
        self.bg_cache = {}
        self._held = 0
        self.new_game(0)

    @property
    def best(self):
        return self.record["best"]

    @property
    def unlocked(self):
        return self.record["unlocked"]

    # ---- controller

    def update(self, dt, pads):
        keys = pads[0] if pads else None
        if keys is None:
            if self.state == "play":
                self.state = "paused"  # the player left or lost signal
            self._held = 0
        else:
            pressed = keys & ~self._held
            self._held = keys
            if pressed & (A | UP):
                self.flap()
            if pressed & START:
                self.pause()
            if pressed & (LEFT | RIGHT) and self.state in ("intro", "ready") and self.score == 0 \
                    and self.level == self.checkpoint and self.lives == START_LIVES:
                step = 1 if pressed & RIGHT else -1
                self.new_game(self.stage + step)
                self.state = "ready"
        self.tick(dt)

    def render(self, canvas: np.ndarray):
        self.draw(Cv(canvas))

    def info(self) -> dict:
        return self.status()

    # ---- state

    def new_game(self, stage):
        self.score = 0
        self.coins = 0
        self.lives = START_LIVES
        self.checkpoint = max(0, min(stage, self.unlocked - 1)) * LEVELS_PER_STAGE
        self.respawn()
        self.state = "intro"

    def respawn(self):
        self.level = self.checkpoint
        self.stage = self.level // LEVELS_PER_STAGE
        self.p = stage_params(self.stage)
        self.t = 0.0
        self.y, self.vy = self.split / 2 - 2 * self.k, 0.0
        self.particles, self.popups = [], []
        self.was_under = False
        self._build_level()
        self.state, self.timer = "ready", 0.0
        self.flap_req = False
        self.banner = (f"STAGE {self.stage + 1}", self.p["name"])

    def flap(self):
        self.flap_req = True

    def pause(self):
        if self.state == "play":
            self.state = "paused"
        elif self.state == "paused":
            self.state = "play"

    def status(self):
        return dict(state=self.state, stage=self.stage + 1, stage_name=self.p["name"],
                    progress=self.level % LEVELS_PER_STAGE + 1, per_stage=LEVELS_PER_STAGE,
                    score=self.score, lives=self.lives, coins=self.coins, best=self.best,
                    unlocked=self.unlocked, total_stages=len(STAGES))

    def _build_level(self):
        sp, k, rng = self.p, self.k, self.rng
        self.bx = 2.0
        self.pipes, self.coin_list = [], []
        gap = sp["gap"] * k
        lo, hi = gap / 2 + 2, self.split - gap / 2 - 2
        last = self.split / 2
        x = FIRST_PIPE_X
        while x < self.w - 20:
            pipe = Pipe()
            pipe.x, pipe.gap, pipe.passed = float(x), gap, False
            pipe.base = last = rng.uniform(max(lo, last - 12 * k), min(hi, last + 12 * k))
            pipe.amp = sp["amp"] * k * rng.choice((0, 1, 1))
            pipe.freq, pipe.phase = sp["freq"], rng.uniform(0, 2 * math.pi)
            self.pipes.append(pipe)
            step = rng.uniform(*sp["spacing"])
            if "coins" in sp["flags"] and rng.random() < 0.5 and x + step < self.w - 10:
                self.coin_list.append([x + step / 2, pipe.base + rng.uniform(-3, 3) * k, False])
            x += step

    # ---- simulation

    def tick(self, dt):
        if self.state == "paused":
            return
        self.timer += dt
        flap, self.flap_req = self.flap_req, False
        if self.state == "intro":
            if flap:
                self.state, self.timer = "ready", 0.0
            return
        if self.state == "ready":
            self.y = self.split / 2 - 2 * self.k + math.sin(self.timer * 4) * 2
            if flap:
                self.state, self.timer = "play", 0.0
                self.banner = None
                self.vy = -self.flap_v * self._gdir()
            return
        if self.state == "over":
            if flap and self.timer > 1.5:
                self.new_game(0)
            return
        if self.state == "dead":
            self.vy = min(self.max_v, self.vy + self.gravity * dt)
            self.y = min(self.h + 4 * self.k, self.y + self.vy * dt)
            if self.timer > 1.5:
                self.lives -= 1
                if self.lives > 0:
                    self.respawn()
                else:
                    self.state, self.timer = "over", 0.0
                    self.record["best"] = max(self.best, self.score)
            return
        self._physics(dt, flap)

    def _gdir(self):
        return -1 if "flip" in self.p["flags"] else 1

    def _physics(self, dt, flap):
        p, h = self.p, self.h
        self.t += dt
        g = self.gravity * (0.55 if "lowg" in p["flags"] else 1) * self._gdir()
        if flap:
            self.vy = -self.flap_v * self._gdir()
        self.vy += g * dt
        if "wind" in p["flags"]:
            wind = (18 * math.sin(self.t * 0.8) + 8 * math.sin(self.t * 2.3)) * self.k
            self.vy += wind * dt
            if self.rng.random() < 0.6:
                self.particles.append([self.w, self.rng.uniform(0, h), wind * 0.4])
        self.vy = max(-self.max_v, min(self.max_v, self.vy))
        self.y += self.vy * dt

        # floor kills, ceiling clamps (swapped when gravity is flipped)
        flipped = self._gdir() < 0
        if self.y < 0:
            if flipped:
                return self._die()
            self.y, self.vy = 0, max(0, self.vy)
        if self.y > h - 4 * self.k:
            if not flipped:
                return self._die()
            self.y, self.vy = h - 4 * self.k, min(0, self.vy)

        self.bx += p["speed"] * dt
        for q in self.particles:
            q[0] -= p["speed"] * 3 * dt
            q[1] += q[2] * dt
        self.particles = [q for q in self.particles if q[0] > -4]
        for pop in self.popups:
            pop[2] -= 8 * dt
            pop[3] -= dt
        self.popups = [pop for pop in self.popups if pop[3] > 0]

        k = self.k
        bx0, bx1, by0, by1 = self.bx + 0.5 * k, self.bx + 4.5 * k, self.y + 0.5 * k, self.y + 3.5 * k
        under = self.split < h
        if under:
            utop, ubot = self._hole()
            if self.y + 2 * k > self.split and not self.was_under:
                self.popups.append(["SKELETON!", self.bx + 8, self.split + 2, 1.2, (255, 120, 40)])
            self.was_under = self.y + 2 * k > self.split
        for pipe in self.pipes:
            c = pipe.center(self.t, self.split)
            blocked = [(0, c - pipe.gap / 2), (c + pipe.gap / 2, self.split)]
            if under:
                blocked += [(self.split, utop), (ubot, h)]
            if pipe.x - k < bx1 and pipe.x + self.pipe_w + k > bx0 and any(by0 < b and by1 > a for a, b in blocked):
                return self._die()
            if not pipe.passed and pipe.x + self.pipe_w < bx0:
                pipe.passed = True
                self.score += 1 + (UNDER_BONUS if self.y > self.split else 0)
        for c in self.coin_list:
            if not c[2] and abs(c[0] + k - (self.bx + 2.5 * k)) < 3 * k and abs(c[1] + k - (self.y + 2 * k)) < 3 * k:
                c[2] = True
                self.coins += 1
                self.score += 5
                self.popups.append(["+5", c[0], c[1] - 6, 0.8, (255, 230, 80)])
                if self.coins % COINS_PER_LIFE == 0 and self.lives < MAX_LIVES:
                    self.lives += 1
                    self.popups.append(["+1 LIFE", self.bx + 8, 8, 1.5, (255, 80, 80)])

        if self.bx > self.w:
            self._level_clear()

    def _hole(self):
        """Underworld opening (top, bottom), shared by every fire column so they move in sync."""
        k, uh = self.k, self.h - self.split - 2 * self.k  # minus the lava lake
        gap = max(9, 15 - self.stage // 3) * k
        amp = (uh - gap) / 2 - 2 * k
        mid = self.split + uh / 2 + amp * math.sin(2 * math.pi * (0.12 + 0.008 * self.stage) * self.t)
        return mid - gap / 2, mid + gap / 2

    def _level_clear(self):
        self.level += 1
        self.checkpoint = self.level
        if self.level % LEVELS_PER_STAGE:
            self.score += 5
            self._build_level()
            return
        self.score += 10 * self.stage + 10
        self.record["unlocked"] = max(self.unlocked, self.stage + 2)
        self.respawn()
        if self.stage == len(STAGES):
            self.banner = ("YOU BEAT IT!", f"{self.p['name']} - FLAP")

    def _die(self):
        self.state, self.timer = "dead", 0.0
        self.vy = -self.flap_v * 0.6

    # ---- drawing

    def _background(self, theme):
        """Sky gradient, decoration and hills for the sky rows, as an (split, w, 3) array."""
        if theme in self.bg_cache:
            return self.bg_cache[theme]
        th, w2, h, w = THEMES[theme], self.w * 2, self.split, self.w
        rng = random.Random(theme)
        clouds = [(rng.uniform(0, w2), rng.uniform(3, 12) * h / 32, rng.uniform(6, 16)) for _ in range(14)]
        stars = {(rng.randrange(w2), rng.randrange(h - 6)) for _ in range(140)}
        sky = np.array([mix(th["top"], th["bot"], y / (h - 1)) for y in range(h)], np.uint8)
        img = np.repeat(sky[:, None], w, axis=1)
        xs, ys = np.arange(w)[None, :], np.arange(h)[:, None]
        if th["deco"] == "clouds":
            cloud = np.zeros((h, w), bool)
            for cx, cy, r in clouds:
                ddx = np.minimum(abs(xs - cx), w2 - abs(xs - cx))
                cloud |= (ddx / r) ** 2 + ((ys - cy) / (r * 0.3)) ** 2 < 1
            light = np.array([mix(c, (255, 255, 255), 0.55) for c in sky], np.uint8)
            img[cloud] = np.broadcast_to(light[:, None], img.shape)[cloud]
        elif th["deco"] == "stars":
            for x, y in stars:
                if x < w:
                    img[y, x] = (200, 200, 230) if (x + y) % 3 else (255, 255, 180)
        hills = np.array([h - 3 - 3 * (math.sin(x / 37) + 1) * 0.5 - 2 * (math.sin(x / 13 + 1) + 1) * 0.5
                          for x in range(w)])[None, :]
        img[ys >= hills] = mix(th["hill"], (255, 255, 255), 0.15)
        img[ys > hills + 0.5] = th["hill"]
        self.bg_cache[theme] = img
        return img

    def _draw_intro(self, cv):
        t, w, h, sp, k = self.timer, self.w, self.h, self.split, self.k
        sky = self._background("day")
        if sp < h and t > MELT_AT:  # sky darkens as the title melts
            f = 1 - 0.6 * min(1, (t - MELT_AT) / 2.5)
            sky = np.array([int(i * f) for i in range(256)], np.uint8)[sky]
        cv.a[:sp] = sky
        if sp < h:
            self._draw_underworld(cv, columns=False)
        bx = -12 * k + t * (w + 24 * k) / 3.2  # bird crosses the meadow before the melt
        cv.sprite(bx, sp * 0.15 + math.sin(t * 6) * 2 * k, BIRD_FLAP if int(t * 8) % 2 else BIRD,
                  BIRD_PAL, scale=k)
        if sp == h:  # single display: no room to melt, show the full title
            text = f"{TITLE} {TITLE2}"
            draw_text(cv, (w - text_width(text, 2 * k)) // 2, (h - 10 * k) // 2, text, (255, 220, 40), 2 * k,
                      shadow=(120, 30, 0))
        else:
            self._draw_melt(cv, t)
            self._draw_rise(cv, t)
        if t > PROMPT_AT and int(t * 2) % 2:
            msg = "FLAP TO START"
            draw_text(cv, (w - text_width(msg, k)) // 2, sp - 7 * k, msg, (255, 255, 255), k, shadow=(0, 0, 0))

    def _draw_melt(self, cv, t):
        """Title on the top display; after MELT_AT each column drips down into the lava."""
        w, h, sp, k = self.w, self.h, self.split, self.k
        if not hasattr(self, "_title_px"):
            tmp = Cv(np.zeros((sp, w, 3), np.uint8))
            s = 4 * k
            draw_text(tmp, (w - text_width(TITLE, s)) // 2, (sp - 5 * s) // 2 + k, TITLE, (255, 220, 40), s,
                      shadow=(200, 70, 0))
            ys, xs = np.nonzero(tmp.a.any(axis=2))
            self._title_px = (xs, ys, tmp.a[ys, xs].astype(float))
            cols = np.unique(xs)
            self._title_bottom = {int(x): int(ys[xs == x].max()) for x in cols}
            self._melt_delay = np.zeros(w)
            for x in cols:
                self._melt_delay[x] = random.Random(int(x)).uniform(0, 1.2)
        lava = h - 2 * k
        drops = 40 * k * np.maximum(0.0, t - MELT_AT - self._melt_delay) ** 2
        for x, yb in self._title_bottom.items():  # short drip tails that fall with the letters
            d = drops[x]
            if d > 1 and yb + d < lava:
                tail = min(d, 6 * k)
                cv.rect(x, yb + d - tail, 1, tail, (170, 50, 0))
        xs, ys, c = self._title_px
        d = drops[xs]
        ny = (ys + d).astype(int)
        keep = ny < lava
        level = np.minimum(8, (8 * d / (lava - ys)).astype(int))[:, None] / 8  # tint quantized to 8 levels
        col = (c + (np.array((255, 70, 0)) - c) * level).astype(np.uint8)
        cv.a[ny[keep], xs[keep]] = col[keep]

    def _draw_rise(self, cv, t):
        """RESURRECTION rises out of the lava with skeleton birdo on each side."""
        if t < RISE_AT:
            return
        w, h, sp, k = self.w, self.h, self.split, self.k
        s = 3 * k
        tw = text_width(TITLE2, s)
        tx = (w - tw) // 2
        ty = sp + (h - sp - 2 * k - 5 * s) // 2
        y = ty + max(0.0, 1 - (t - RISE_AT)) * (h - ty)
        rnd = random.Random(int(t * 10))
        for i, ch in enumerate(TITLE2):
            draw_text(cv, tx + i * 4 * s, y, ch, FLAME[rnd.randint(0, 2)], s, shadow=(90, 0, 0))
        for bx, ph in ((tx - 14 * k, 0), (tx + tw + 9 * k, 2)):
            cv.sprite(bx, y + 4 * k + math.sin(t * 5 + ph) * 2 * k, SKEL_FLAP if int(t * 8 + ph) % 2 else SKEL,
                      SKEL_PAL, scale=k)

    def draw(self, cv):
        if self.state == "intro":
            return self._draw_intro(cv)
        p, h = self.p, self.h
        cv.a[:self.split] = self._background(p["theme"])
        if self.split < h:
            self._draw_underworld(cv)

        th = THEMES[p["theme"]]
        if th["deco"] == "rain":
            for i in range(60):
                x = (i * 53 - self.t * 220) % self.w
                y = (i * 17 + self.t * 90) % h
                cv.px(x, y, (150, 160, 200))
        for q in self.particles:
            cv.rect(q[0], q[1], 3, 1, (200, 210, 230))

        dark = "dark" in p["flags"]
        for pipe in self.pipes:
            f = 1.0
            if dark:
                f = max(0.07, 1 - max(0, abs(pipe.x - self.bx) - 30) / 110)
            col = th["pipe"]
            body, hi, lo = dim(col, f), dim(mix(col, (255, 255, 255), 0.4), f), dim(col, 0.55 * f)
            c = pipe.center(self.t, self.split)
            top = round(c - pipe.gap / 2)
            bot = top + pipe.gap
            x, k, pw = pipe.x, self.k, self.pipe_w
            for y0, y1 in ((0, top), (bot, self.split)):
                cv.rect(x, y0, pw, y1 - y0, body)
                cv.rect(x + k, y0, k, y1 - y0, hi)
                cv.rect(x + pw - k, y0, k, y1 - y0, lo)
            cv.rect(x - k, top - 2 * k, pw + 2 * k, 2 * k, hi)
            cv.rect(x - k, bot, pw + 2 * k, 2 * k, hi)

        for cx, cy, taken in self.coin_list:
            if not taken:
                f = max(0.1, 1 - max(0, abs(cx - self.bx) - 30) / 110) if dark else 1
                wdt = 2 if int(self.t * 6 + cx) % 3 else 1
                cv.rect(cx + (2 - wdt) * self.k / 2, cy, wdt * self.k, 2 * self.k, dim((255, 210, 0), f))

        flip = self._gdir() < 0
        flapping = self.vy * self._gdir() < 0 and int(self.timer * 12) % 2
        skel = self.y + 2 * self.k > self.split
        spr, pal = (SKEL_FLAP if flapping else SKEL, SKEL_PAL) if skel else (BIRD_FLAP if flapping else BIRD, BIRD_PAL)
        if self.state == "dead" and int(self.timer * 10) % 2:
            pal = {c: (255, 60, 60) for c in "ywkolrg"}
        cv.sprite(self.bx, self.y, spr, pal, flip, self.k)

        for text, x, y, ttl, col in self.popups:
            draw_text(cv, x, y, text, col, shadow=(0, 0, 0))

        self._hud(cv)
        if self.state == "dead" and self.timer < 0.12:
            cv.clear((120, 0, 0))

    def _draw_underworld(self, cv, columns=True):
        h, sp, k, w = self.h, self.split, self.k, self.w
        if not hasattr(self, "_under_bg"):
            rows = [mix((25, 0, 0), (90, 18, 0), (y - sp) / max(1, h - sp - 1)) for y in range(sp, h)]
            self._under_bg = np.repeat(np.array(rows, np.uint8)[:, None], w, axis=1)
        cv.a[sp:] = self._under_bg
        t = self.t + self.timer * (self.state != "play")
        for i in range(40):  # rising embers
            x = (i * 97 + math.sin(t + i) * 6) % w
            y = h - 1 - ((i * 13 + t * (6 + i % 5)) % (h - sp))
            cv.px(x, y, FLAME[1 + i % 3])
        for x in range(0, w, 4 * k):  # lava lake, shimmering
            cv.rect(x, h - 2 * k, 4 * k, 2 * k, FLAME[int(t * 3 + x * 0.37) % 3 + 1])
        if not columns:
            return
        utop, ubot = self._hole()
        rnd = random.Random(int(t * 12))
        for pipe in self.pipes:
            x, pw = pipe.x, self.pipe_w
            cv.rect(x, ubot, pw, h - ubot, (120, 25, 0))  # magma pillar
            cv.rect(x + k, ubot, k, h - ubot, (200, 60, 0))
            cv.rect(x, sp, pw, utop - sp, (60, 10, 5))  # hanging rock
            for col in range(0, pw, k):  # flames: bright at the hole edge, darker inside; tips flicker
                depth = rnd.randint(3, 6)
                for j in range(depth):
                    f = FLAME[min(len(FLAME) - 1, j * len(FLAME) // depth)]
                    cv.rect(x + col, ubot + j * k, k, k, f)
                    cv.rect(x + col, utop - (j + 1) * k, k, k, f)
                if rnd.random() < 0.4:  # cosmetic tip licking into the hole
                    cv.rect(x + col, ubot - k, k, k, FLAME[1])
                if rnd.random() < 0.4:
                    cv.rect(x + col, utop, k, k, FLAME[1])

    def _hud(self, cv):
        w = self.w
        for i in range(self.lives):
            cv.sprite(2 + i * 6, 1, HEART, {"r": (255, 50, 70)})
        x = 4 + max(self.lives, 1) * 6
        draw_text(cv, x, 1, f"S{self.stage + 1}  LEVEL {self.level % LEVELS_PER_STAGE + 1}/{LEVELS_PER_STAGE}",
                  (255, 255, 255), shadow=(0, 0, 0))
        s = f"SCORE {self.score}  BEST {max(self.best, self.score)}"
        draw_text(cv, w - text_width(s) - 2, 1, s, (255, 255, 255), shadow=(0, 0, 0))

        lines = None
        if self.state == "ready":
            lines = (self.banner[0] + "  " + self.banner[1], "FLAP TO START")
        elif self.state == "paused":
            lines = ("PAUSED", "")
        elif self.state == "over":
            lines = ("GAME OVER", f"SCORE {self.score}  BEST {self.best}  - FLAP TO RESTART")
        if lines:
            big, small = lines
            k = self.k
            bw = text_width(big, 2 * k)
            sw = text_width(small, k)
            bx = w - max(bw, sw) - 10 * k  # right side of the top display
            by = (self.split - 22 * k) // 2 + k
            cv.rect(bx - 6 * k, by, max(bw, sw) + 12 * k, 22 * k, (0, 0, 0))
            draw_text(cv, bx, by + 2 * k, big, (255, 230, 80), 2 * k)
            draw_text(cv, bx, by + 15 * k, small, (200, 220, 255), k)

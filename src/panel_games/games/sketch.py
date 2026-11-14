"""Panel Sketch: a drawing and guessing game. The server picks who draws; everyone else types guesses.

Each player draws once per cycle. The drawing area takes the left 560 pixels; the right end holds the timer,
the word hint and the leaderboard. Faster correct guesses earn more points, and the drawer is paid when people
guess their picture.
"""

import random
import re

import cv2
import numpy as np

from .base import FONT, PLAYER_COLORS, Lobby, center_text, draw_podium, text
from .pad import DOWN, UP
from .words import WORDS

CW, CH = 560, 128
PX = CW
BG = (14, 14, 28)
PANEL_BG = (24, 22, 46)
PALETTE = [(255, 255, 255), (255, 60, 60), (255, 150, 30), (255, 230, 40), (60, 220, 90), (40, 210, 255), (70, 110, 255), (240, 80, 220)]
ERASER = 8
TURN_TIME, INTRO_TIME, REVEAL_TIME, OVER_TIME = 75.0, 3.0, 6.0, 16.0
GUESS_GAP = 0.4
CYCLES = [1, 2, 3]


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def matches(guess: str, word: str) -> bool:
    g, w = norm(guess), norm(word)
    return bool(g) and (g == w or g.rstrip("s") == w.rstrip("s"))


def edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def is_close(guess: str, word: str) -> bool:
    g, w = norm(guess), norm(word)
    return len(w) >= 5 and bool(g) and edit_distance(g, w) <= 1


class Sketch:
    id = "sketch"
    title = "Panel Sketch"
    description = "Drawing and guessing for everyone. One player draws a secret word on their phone, the rest guess. Faster guesses score more."
    max_players = 8
    size = (768, 128)
    layout = "sketch"
    controls = "Drawer: draw with your finger. Everyone else: type your guesses"

    def __init__(self, width: int, height: int, seed=None, words=None):
        if width < 768 or height < 128:
            raise ValueError("Panel Sketch needs a 768x128 display")
        self.width, self.height = width, height
        self.rng = random.Random(seed)
        self.words = list(words if words is not None else WORDS)
        self.t = 0.0
        self.lobby = Lobby()
        self.cycles_i = 0
        self.state = "lobby"
        self.joined: set[int] = set()
        self.prev: dict[int, int] = {}
        self.canvas = np.zeros((CH, CW, 3), np.uint8)
        self._reset_game()

    def _reset_game(self):
        self.canvas[:] = BG
        self.roster: list[int] = []
        self.order: list[int] = []
        self.scores: dict[int, int] = {}
        self.turn_i = -1
        self.drawer: int | None = None
        self.word, self.used = "", set()
        self.solved: dict[int, int] = {}
        self.feedback: dict[int, list] = {}
        self.last_guess: dict[int, float] = {}
        self.revealed: set[int] = set()
        self.remaining, self.timer, self.early = 0.0, 0.0, None
        self.gains: dict[int, int] = {}
        self.ranking: list = []

    # ---- information shared with phones

    def hint_chars(self) -> list[str]:
        return [" " if ch == " " else (ch.upper() if i in self.revealed else "_") for i, ch in enumerate(self.word)]

    def info(self) -> dict:
        playing = self.state in ("intro", "draw", "reveal")
        return {"state": self.state, "host": self.lobby.host, "drawer": self.drawer if playing else None,
                "remaining": int(self.remaining + 0.99) if self.state == "draw" else None,
                "hint": "".join(self.hint_chars()) if self.state in ("draw", "intro") else None,
                "turn": self.turn_i + 1, "turns": len(self.order), "cycles": CYCLES[self.cycles_i]}

    def private_info(self, slot: int) -> dict:
        active = self.state in ("intro", "draw")
        role = ("drawer" if slot == self.drawer else "guesser" if slot in self.roster else "wait") if active else "wait"
        out = {"role": role, "score": self.scores.get(slot, 0), "solved": slot in self.solved,
               "feedback": list(self.feedback.get(slot, []))[-8:], "canvas": [CW, CH]}
        if role == "drawer" or self.state == "reveal":
            out["word"] = self.word
        return out

    # ---- messages from phones

    def on_message(self, slot: int, data: dict):
        kind = data.get("t")
        if kind == "draw" and self.state == "draw" and slot == self.drawer:
            self._apply_ops(data.get("ops"))
        elif kind == "guess" and self.state == "draw":
            self._guess(slot, data.get("text"))

    def _apply_ops(self, ops):
        if not isinstance(ops, list):
            return
        for op in ops[:40]:
            if not isinstance(op, dict):
                continue
            if op.get("clear"):
                self.canvas[:] = BG
                continue
            pts, c, s = op.get("p"), op.get("c"), op.get("s")
            if not (isinstance(pts, list) and isinstance(c, int) and isinstance(s, int)) or isinstance(c, bool) or isinstance(s, bool):
                continue
            color = BG if c >= ERASER else PALETTE[max(0, c)]
            s = max(1, min(16, s))
            clean = [(int(max(0, min(CW - 1, p[0]))), int(max(0, min(CH - 1, p[1])))) for p in pts[:300]
                     if isinstance(p, (list, tuple)) and len(p) == 2
                     and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in p)]
            if len(clean) == 1:
                cv2.circle(self.canvas, clean[0], max(1, s // 2), color, -1)
            for a, b in zip(clean, clean[1:]):
                cv2.line(self.canvas, a, b, color, s)
            if s > 3:
                for p in clean[:1] + clean[-1:]:
                    cv2.circle(self.canvas, p, s // 2, color, -1)

    def _note(self, slot: int, guess: str, status: str):
        self.feedback.setdefault(slot, []).append([guess, status])
        del self.feedback[slot][:-12]

    def _guess(self, slot: int, raw):
        if slot not in self.roster or slot == self.drawer or slot in self.solved or slot not in self.joined:
            return
        if not isinstance(raw, str):
            return
        guess = re.sub(r"\s+", " ", raw).strip()[:40]
        if not guess or self.t - self.last_guess.get(slot, -9) < GUESS_GAP:
            return
        self.last_guess[slot] = self.t
        if matches(guess, self.word):
            frac = max(0.0, self.remaining) / TURN_TIME
            pts = (100 + int(400 * frac)) // 10 * 10 + (50, 30, 10, 0)[min(len(self.solved), 3)]
            self.solved[slot] = pts
            self.scores[slot] = self.scores.get(slot, 0) + pts
            self.gains[slot] = pts
            if self.drawer is not None:
                bonus = pts // 4
                self.scores[self.drawer] = self.scores.get(self.drawer, 0) + bonus
                self.gains[self.drawer] = self.gains.get(self.drawer, 0) + bonus
            self._note(slot, guess, "correct")
        else:
            self._note(slot, guess, "close" if is_close(guess, self.word) else "wrong")

    # ---- turns

    def _start_game(self, roster):
        self._reset_game()
        self.roster = sorted(roster)
        self.scores = {s: 0 for s in self.roster}
        self.order = self.roster * CYCLES[self.cycles_i]
        self._next_turn()

    def _next_turn(self):
        while True:
            self.turn_i += 1
            if self.turn_i >= len(self.order):
                ordered = sorted(self.roster, key=lambda s: (-self.scores[s], s))
                self.ranking = [(s, self.scores[s]) for s in ordered]
                self.state, self.timer, self.drawer = "over", OVER_TIME, None
                return
            if self.order[self.turn_i] in self.joined:
                break
        self.drawer = self.order[self.turn_i]
        fresh = [w for w in self.words if w not in self.used] or self.words
        self.word = self.rng.choice(fresh)
        self.used.add(self.word)
        self.canvas[:] = BG
        self.solved, self.gains, self.revealed, self.early = {}, {}, set(), None
        self.feedback = {}
        self.remaining, self.timer = TURN_TIME, INTRO_TIME
        self.state = "intro"

    def _end_turn(self):
        self.state, self.timer = "reveal", REVEAL_TIME

    def _maybe_hint(self):
        letters = [i for i, ch in enumerate(self.word) if ch != " " and i not in self.revealed]
        due = sum(1 for frac in (0.5, 0.25) if self.remaining <= TURN_TIME * frac)
        if len(self.word) >= 4 and len(self.revealed) < due and letters:
            self.revealed.add(self.rng.choice(letters))

    # ---- update

    def update(self, dt: float, pads: list):
        self.t += dt
        joined = {i for i, k in enumerate(pads) if k is not None}
        keys = {i: (pads[i] or 0) for i in joined}
        pressed = {i: k & ~self.prev.get(i, 0) for i, k in keys.items()}
        self.prev = keys
        self.joined = joined
        if self.state == "lobby":
            host = self.lobby.host
            if host is not None and host in pressed:
                if pressed[host] & UP:
                    self.cycles_i = min(len(CYCLES) - 1, self.cycles_i + 1)
                if pressed[host] & DOWN:
                    self.cycles_i = max(0, self.cycles_i - 1)
            if self.lobby.update(joined, pads):
                self._start_game(joined)
        elif self.state == "intro":
            self.timer -= dt
            if self.timer <= 0:
                self.state = "draw"
        elif self.state == "draw":
            self.remaining -= dt
            self._maybe_hint()
            guessers = [s for s in self.roster if s != self.drawer and s in joined]
            everyone = not guessers or all(s in self.solved for s in guessers)
            drawer_gone = self.drawer not in joined
            if (everyone or drawer_gone) and self.early is None:
                self.early = 0.5 if drawer_gone else 2.5
            if self.early is not None:
                self.early -= dt
            if self.remaining <= 0 or (self.early is not None and self.early <= 0):
                self._end_turn()
        elif self.state == "reveal":
            self.timer -= dt
            if self.timer <= 0:
                self._next_turn()
        else:
            self.timer -= dt
            if self.timer <= 0:
                self.state = "lobby"
                self.lobby.reset()
                self._reset_game()

    # ---- drawing

    @staticmethod
    def _center_in(canvas, s, x0, x1, y, scale, color, thickness=1):
        (tw, _), _ = cv2.getTextSize(s, FONT, scale, thickness)
        text(canvas, s, x0 + (x1 - x0 - tw) // 2, y, scale, color, thickness)

    def _panel(self, canvas):
        canvas[:, PX:] = PANEL_BG
        canvas[:, PX : PX + 2] = (90, 90, 130)
        x0, x1 = PX + 4, self.width - 4
        playing = self.state in ("intro", "draw", "reveal")
        if playing and self.drawer is not None:
            text(canvas, f"TURN {self.turn_i + 1}/{len(self.order)}", PX + 8, 11, 0.38, (180, 180, 210))
            label = f"P{self.drawer + 1} DRAWS"
            (tw, _), _ = cv2.getTextSize(label, FONT, 0.4, 1)
            text(canvas, label, self.width - 8 - tw, 11, 0.4, PLAYER_COLORS[self.drawer])
        if self.state == "draw":
            rem = max(0, int(self.remaining + 0.99))
            color = (255, 90, 90) if rem <= 10 else (255, 255, 255)
            self._center_in(canvas, f"{rem // 60}:{rem % 60:02d}", x0, x1, 40, 1.0, color, 2)
            frac = max(0.0, self.remaining) / TURN_TIME
            canvas[46:50, PX + 10 : PX + 10 + int((x1 - x0 - 12) * frac)] = (255, int(70 + 150 * frac), 60) if frac < 0.3 else (90, 200, 255)
        if self.state in ("draw", "intro"):
            chars = self.hint_chars()
            n = max(1, len(chars))
            w = min(13, (x1 - x0 - 8) // n)
            start = x0 + (x1 - x0 - w * n) // 2 + 4
            for i, ch in enumerate(chars):
                x = start + i * w
                if ch == "_":
                    canvas[70:72, x : x + max(2, w - 3)] = (230, 230, 240)
                elif ch != " ":
                    text(canvas, ch, x, 68, 0.5, (255, 230, 90))
            text(canvas, f"{len(self.word.replace(' ', ''))} letters", x0 + 6, 82, 0.34, (150, 150, 190))
        board_top = 90 if self.state in ("draw", "intro") else 20
        order = sorted(self.roster or sorted(self.joined), key=lambda s: -self.scores.get(s, 0))
        cols = 2 if len(order) > 4 else 1
        rows = -(-len(order) // cols) or 1
        row_h = min(13, (126 - board_top) // rows)
        col_w = (x1 - x0) // cols
        for i, s in enumerate(order):
            x, y = x0 + 6 + (i // rows) * col_w, board_top + (i % rows) * row_h + row_h - 3
            canvas[y - 6 : y, x : x + 6] = PLAYER_COLORS[s]
            gain = self.gains.get(s, 0)
            text(canvas, f"P{s + 1} {self.scores.get(s, 0)}", x + 9, y, 0.36, PLAYER_COLORS[s])
            mark_x = x + col_w - 22
            if s == self.drawer and self.state in ("intro", "draw"):
                cv2.line(canvas, (mark_x, y), (mark_x + 6, y - 6), (255, 230, 90), 2)
            elif s in self.solved or (self.state == "reveal" and gain):
                cv2.line(canvas, (mark_x, y - 3), (mark_x + 2, y), (90, 255, 120), 2)
                cv2.line(canvas, (mark_x + 2, y), (mark_x + 7, y - 7), (90, 255, 120), 2)
            if self.state == "reveal" and gain:
                text(canvas, f"+{gain}", x + col_w - 52, y, 0.32, (120, 255, 120))

    def render(self, canvas: np.ndarray):
        if self.state == "over":
            return draw_podium(canvas, self.ranking, self.t)
        canvas[:] = PANEL_BG
        canvas[:, :CW] = self.canvas
        if self.state == "lobby":
            canvas[:, :CW] = BG
            self._center_in(canvas, "PANEL SKETCH", 0, CW, 36, 1.2, (255, 230, 0), 2)
            self._center_in(canvas, "One player draws a secret word, everyone else types guesses.", 0, CW, 62, 0.5, (255, 255, 255))
            self._center_in(canvas, "The faster you guess, the more you score.", 0, CW, 80, 0.5, (255, 255, 255))
            (tw, _), _ = cv2.getTextSize(self.lobby.message, FONT, 0.8, 1)
            text(canvas, self.lobby.message, (CW - tw) // 2, 108, 0.8, (120, 255, 120) if self.joined else (255, 255, 255))
            canvas[:, PX : PX + 2] = (90, 90, 130)
            self._center_in(canvas, "EVERYONE DRAWS", PX + 4, self.width - 4, 22, 0.45, (200, 200, 230))
            self._center_in(canvas, f"{CYCLES[self.cycles_i]} TIME{'S' if CYCLES[self.cycles_i] > 1 else ''}", PX + 4, self.width - 4, 46, 0.8, (120, 220, 255))
            self._center_in(canvas, "(UP / DOWN)", PX + 4, self.width - 4, 62, 0.38, (150, 150, 190))
            for i, s in enumerate(sorted(self.joined)):
                canvas[80:92, PX + 10 + i * 22 : PX + 28 + i * 22] = PLAYER_COLORS[s]
            return
        self._panel(canvas)
        if self.state == "intro":
            canvas[44:84, :CW] = canvas[44:84, :CW] // 4
            self._center_in(canvas, f"PLAYER {self.drawer + 1} IS DRAWING", 0, CW, 72, 1.0, PLAYER_COLORS[self.drawer], 2)
        elif self.state == "reveal":
            canvas[34:94, :CW] = canvas[34:94, :CW] // 4
            self._center_in(canvas, "THE WORD WAS", 0, CW, 54, 0.5, (200, 200, 230))
            self._center_in(canvas, self.word.upper(), 0, CW, 82, 1.1, (255, 230, 90), 2)

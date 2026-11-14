"""Panel Trivia: everyone answers on their phone, faster correct answers score more.

Questions come from src/panel_games/data/trivia.json (see scripts/fetch_trivia.py). The host picks a deck and
the number of questions in the lobby with the controller arrows.
"""

import json
import random
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from .base import FONT, PLAYER_COLORS, Lobby, center_text, draw_podium, text
from .pad import A, B, DOWN, LEFT, RIGHT, UP, X, Y

DATA = Path(__file__).resolve().parent.parent / "data" / "trivia.json"
DECKS = [("mixed", "MIXED"), ("nerd", "NERD"), ("engineering", "ENGINEERING"), ("movies", "MOVIES & TV"),
         ("scifi", "SCI-FI"), ("fantasy", "FANTASY")]
COUNTS = [5, 10, 15, 20, 30]
QUESTION_TIME, REVEAL_TIME, OVER_TIME = 20.0, 6.0, 16.0
ANSWER_BITS = ((A, 0), (B, 1), (X, 2), (Y, 3))
BOX_COLORS = [(220, 50, 50), (40, 100, 240), (240, 190, 20), (40, 170, 80)]
LETTERS = "ABCD"

# Used only if the downloaded data file is missing.
_BUILTIN = [
    {"q": "What does CPU stand for?", "correct": "Central Processing Unit",
     "wrong": ["Computer Personal Unit", "Central Program Utility", "Core Power Unit"], "decks": ["nerd", "engineering"]},
    {"q": "Which planet is known as the Red Planet?", "correct": "Mars", "wrong": ["Venus", "Jupiter", "Mercury"], "decks": ["scifi"]},
    {"q": "What is the SI unit of electrical resistance?", "correct": "Ohm", "wrong": ["Volt", "Watt", "Farad"], "decks": ["engineering"]},
    {"q": "Who wrote The Lord of the Rings?", "correct": "J. R. R. Tolkien", "wrong": ["C. S. Lewis", "George R. R. Martin", "Terry Pratchett"], "decks": ["fantasy"]},
    {"q": "Which binary number equals 5?", "correct": "101", "wrong": ["110", "011", "111"], "decks": ["nerd", "engineering"]},
    {"q": "Which film features the quote 'May the Force be with you'?", "correct": "Star Wars", "wrong": ["Star Trek", "Alien", "Dune"], "decks": ["movies", "scifi"]},
    {"q": "What is the chemical symbol for gold?", "correct": "Au", "wrong": ["Ag", "Gd", "Go"], "decks": ["engineering"]},
    {"q": "In computing, what does HTTP stand for?", "correct": "HyperText Transfer Protocol",
     "wrong": ["High Transfer Text Process", "HyperText Transmission Program", "Host Terminal Transfer Protocol"], "decks": ["nerd"]},
]


@lru_cache(maxsize=1)
def load_questions() -> tuple:
    try:
        return tuple(json.loads(DATA.read_text())["questions"])
    except (OSError, ValueError, KeyError):
        return tuple(_BUILTIN)


def deck_questions(deck: str, source=None) -> list:
    pool = source if source is not None else load_questions()
    return list(pool) if deck == "mixed" else [q for q in pool if deck in q.get("decks", [])]


def wrap(s: str, max_w: int, scale: float) -> list[str]:
    lines, cur = [], ""
    for word in s.split():
        trial = f"{cur} {word}".strip()
        if cur and cv2.getTextSize(trial, FONT, scale, 1)[0][0] > max_w:
            lines.append(cur)
            cur = word
        else:
            cur = trial
    return lines + ([cur] if cur else [])


def fit(s: str, w: int, h: int, scales=(0.9, 0.8, 0.7, 0.6, 0.5, 0.45, 0.4, 0.35)):
    """Largest text scale that wraps s into a w x h box. Returns (lines, scale, line_height)."""
    for sc in scales:
        lh = int(24 * sc) + 3
        lines = wrap(s, w, sc)
        if len(lines) * lh <= h and all(cv2.getTextSize(l, FONT, sc, 1)[0][0] <= w for l in lines):
            return lines, sc, lh
    sc = scales[-1]
    lh = int(24 * sc) + 3
    lines = wrap(s, w, sc)[: max(1, h // lh)]
    lines[-1] = lines[-1][: max(0, len(lines[-1]) - 3)] + "..."
    return lines, sc, lh


class Trivia:
    id = "trivia"
    title = "Panel Trivia"
    description = "Nerd, engineering, movie, sci-fi and fantasy trivia. Answer on your phone; faster correct answers score more."
    max_players = 8
    size = (768, 128)
    layout = "quiz"
    controls = "Tap A, B, C or D. The host picks the deck with the arrows and presses Start"

    def __init__(self, width: int, height: int, seed=None, questions=None):
        if width < 768 or height < 128:
            raise ValueError("Panel Trivia needs a 768x128 display")
        self.width, self.height = width, height
        self.rng = random.Random(seed)
        self.source = questions
        self.t = 0.0
        self.lobby = Lobby()
        self.deck_i, self.count_i = 0, 1
        self.state = "lobby"
        self.joined: set[int] = set()
        self.host_prev = 0
        self.prev: dict[int, int] = {}
        self._reset_game()

    def _reset_game(self):
        self.roster: list[int] = []
        self.scores: dict[int, int] = {}
        self.gained: dict[int, int] = {}
        self.answers: dict[int, tuple[int, float]] = {}
        self.questions: list = []
        self.qi, self.remaining, self.timer, self.elapsed = 0, 0.0, 0.0, 0.0
        self.options: list[str] = []
        self.correct = 0
        self.ranking: list = []

    def info(self) -> dict:
        return {"state": self.state, "host": self.lobby.host}

    # ---- helpers

    def deck_choices(self):
        pool = self.source if self.source is not None else load_questions()
        return [(d, name) for d, name in DECKS if len(deck_questions(d, pool)) >= 5]

    def _pool(self):
        deck = self.deck_choices()[self.deck_i % len(self.deck_choices())][0]
        return deck_questions(deck, self.source)

    def _start_game(self, roster):
        pool = self._pool()
        n = min(COUNTS[self.count_i], len(pool))
        self.roster = sorted(roster)
        self.scores = {s: 0 for s in self.roster}
        self.questions = self.rng.sample(pool, n)
        self.qi = -1
        self._next_question()

    def _next_question(self):
        self.qi += 1
        if self.qi >= len(self.questions):
            ordered = sorted(self.roster, key=lambda s: (-self.scores[s], s))
            self.ranking = [(s, self.scores[s]) for s in ordered]
            self.state, self.timer = "over", OVER_TIME
            return
        q = self.questions[self.qi]
        opts = [(q["correct"], True)] + [(w, False) for w in q["wrong"][:3]]
        self.rng.shuffle(opts)
        self.options = [o for o, _ in opts]
        self.correct = next(i for i, (_, ok) in enumerate(opts) if ok)
        self.answers, self.gained = {}, {s: 0 for s in self.roster}
        self.remaining, self.elapsed = QUESTION_TIME, 0.0
        self.state = "question"

    def _reveal(self):
        for slot, (idx, left) in self.answers.items():
            if idx == self.correct:
                pts = 500 + int(500 * left / QUESTION_TIME)
                self.gained[slot] = pts // 10 * 10
                self.scores[slot] += self.gained[slot]
        self.state, self.timer = "reveal", REVEAL_TIME

    # ---- update

    def update(self, dt: float, pads: list):
        self.t += dt
        joined = {i for i, k in enumerate(pads) if k is not None}
        keys = {i: (pads[i] or 0) for i in joined}
        pressed = {i: k & ~self.prev.get(i, 0) for i, k in keys.items()}
        self.prev = keys
        if self.state == "lobby":
            self.joined = joined
            host = self.lobby.host
            if host is not None and host in pressed:
                self._lobby_options(pressed[host])
            if self.lobby.update(joined, pads) and self.deck_choices():
                self._start_game(joined)
        elif self.state == "question":
            self.elapsed += dt
            self.remaining = max(0.0, self.remaining - dt)
            for slot in self.roster:
                if slot in joined and slot not in self.answers:
                    for bit, idx in ANSWER_BITS:
                        if pressed.get(slot, 0) & bit:
                            self.answers[slot] = (idx, self.remaining)
                            break
            present = [s for s in self.roster if s in joined]
            if self.remaining <= 0 or (present and self.elapsed > 1.0 and all(s in self.answers for s in present)):
                self._reveal()
        elif self.state == "reveal":
            self.timer -= dt
            if self.timer <= 0:
                self._next_question()
        else:
            self.timer -= dt
            if self.timer <= 0:
                self.state, self.joined = "lobby", joined
                self.lobby.reset()
                self._reset_game()

    def _lobby_options(self, pressed: int):
        n_decks = len(self.deck_choices())
        if pressed & LEFT:
            self.deck_i = (self.deck_i - 1) % n_decks
        if pressed & RIGHT:
            self.deck_i = (self.deck_i + 1) % n_decks
        if pressed & UP:
            self.count_i = min(len(COUNTS) - 1, self.count_i + 1)
        if pressed & DOWN:
            self.count_i = max(0, self.count_i - 1)

    # ---- drawing

    def _answer_boxes(self, canvas, reveal: bool):
        for i, opt in enumerate(self.options):
            x0, y0, w, h = 8 + i * 190, 68, 180, 46
            dim = reveal and i != self.correct
            color = tuple(c // 4 for c in BOX_COLORS[i]) if dim else BOX_COLORS[i]
            canvas[y0 : y0 + h, x0 : x0 + w] = color
            fg = (20, 20, 20) if i == 2 and not dim else (255, 255, 255) if not dim else (120, 120, 120)
            cv2.putText(canvas, LETTERS[i], (x0 + 6, y0 + 30), FONT, 1.0, fg, 2, cv2.LINE_AA)
            lines, sc, lh = fit(opt, w - 42, h - 4, (0.5, 0.45, 0.4, 0.35))
            top = y0 + (h - len(lines) * lh) // 2
            for j, line in enumerate(lines):
                text(canvas, line, x0 + 36, top + int(16 * sc) + j * lh, sc, fg)
            if reveal:
                for slot, (idx, _) in self.answers.items():
                    if idx == i:
                        k = sum(1 for s, (ix, _) in self.answers.items() if ix == i and s < slot)
                        canvas[y0 + h - 8 : y0 + h - 2, x0 + w - 10 - k * 8 : x0 + w - 4 - k * 8] = PLAYER_COLORS[slot]
            if reveal and i == self.correct:
                canvas[y0 : y0 + 2, x0 : x0 + w] = 255
                canvas[y0 + h - 2 : y0 + h, x0 : x0 + w] = 255

    def _chips(self, canvas):
        for i, slot in enumerate(self.roster):
            x, y = 8 + i * 12, 118
            if slot in self.answers:
                canvas[y : y + 8, x : x + 8] = PLAYER_COLORS[slot]
            else:
                canvas[y : y + 8, x : x + 8] = tuple(c // 5 for c in PLAYER_COLORS[slot])

    def render(self, canvas: np.ndarray):
        canvas[:] = (8, 10, 24)
        if self.state == "lobby":
            choices = self.deck_choices()
            deck, name = choices[self.deck_i % len(choices)]
            n_avail = len(deck_questions(deck, self.source if self.source is not None else load_questions()))
            center_text(canvas, "PANEL TRIVIA", 28, 1.0, (255, 230, 0), box=False)
            center_text(canvas, f"<  {name}  >   {n_avail} questions", 58, 0.8, (120, 220, 255), box=False)
            center_text(canvas, f"{min(COUNTS[self.count_i], n_avail)} QUESTIONS  (UP / DOWN)", 80, 0.5, box=False)
            center_text(canvas, self.lobby.message, 108, 0.8, (120, 255, 120) if self.joined else (255, 255, 255), box=False)
            for i, s in enumerate(sorted(self.joined)):
                canvas[118:126, 8 + i * 12 : 16 + i * 12] = PLAYER_COLORS[s]
            return
        if self.state == "over":
            return draw_podium(canvas, self.ranking, self.t)
        reveal = self.state == "reveal"
        if not reveal:
            frac = self.remaining / QUESTION_TIME
            canvas[0:5, : int(self.width * frac)] = (int(255 * (1 - frac)), int(220 * frac) + 35, 60)
            lines, sc, lh = fit(self.questions[self.qi]["q"], 752, 54)
            for j, line in enumerate(lines):
                text(canvas, line, 8, 8 + int(20 * sc) + j * lh, sc, (255, 255, 255))
            self._chips(canvas)
            text(canvas, f"Q{self.qi + 1}/{len(self.questions)}   {int(self.remaining + 0.99)}s", 640, 126, 0.4, (200, 200, 200))
        else:
            self._scoreboard(canvas)
        self._answer_boxes(canvas, reveal)

    def _scoreboard(self, canvas):
        n = max(1, len(self.roster))
        col_w = min(94, 752 // n)
        top = max(self.scores.values(), default=1) or 1
        for i, slot in enumerate(sorted(self.roster, key=lambda s: -self.scores[s])):
            x = 8 + i * col_w
            color = PLAYER_COLORS[slot]
            gained = self.gained.get(slot, 0)
            text(canvas, f"P{slot + 1}", x, 18, 0.5, color)
            text(canvas, f"+{gained}" if gained else "+0", x + 28, 18, 0.45, (120, 255, 120) if gained else (130, 130, 130))
            text(canvas, str(self.scores[slot]), x, 42, 0.7, (255, 255, 255))
            canvas[50:58, x : x + max(2, int((col_w - 12) * self.scores[slot] / top))] = color

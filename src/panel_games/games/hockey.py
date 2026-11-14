"""Panel Hockey: air hockey across the whole display for up to four players, with the computer filling an empty side.

Teams are by slot: players 1 and 3 defend the left goal, players 2 and 4 the right. Each player drags a mallet inside
their team's half by touching a pad on their phone (absolute position, sent as messages).
"""

import math
import random

import cv2
import numpy as np

from .base import FONT, PLAYER_COLORS, Lobby, center_text
from .pad import DOWN, UP

W, H, MID = 768, 128, 384
R_PADDLE, R_PUCK = 9, 5
GOAL_TOP, GOAL_BOTTOM = 40, 88
PADDLE_SPEED, AI_SPEED = 1100.0, 470.0
MAX_PUCK, FRICTION, BOUNCE = 800.0, 0.3, 0.92
SUBSTEPS = 4
WINS = [3, 5, 7, 10]
SERVE_TIME, GOAL_TIME, OVER_TIME = 1.5, 1.8, 7.0
TEAM_COLORS = ((0, 200, 255), (255, 140, 0))
TEAMS = ("LEFT", "RIGHT")
BG, LINE = (10, 18, 44), (70, 90, 150)


def team_of(slot: int) -> int:
    return slot % 2


def half_bounds(team: int) -> tuple[float, float]:
    return (R_PADDLE, MID - R_PADDLE - 2) if team == 0 else (MID + R_PADDLE + 2, W - R_PADDLE)


class Hockey:
    id = "hockey"
    title = "Panel Hockey"
    description = "Air hockey for up to four players: slide your mallet with a finger and knock the puck into the other goal. The computer plays an empty side."
    max_players = 4
    size = (768, 128)
    layout = "air"
    controls = "Drag your finger on the pad to move your mallet in your half. Host picks the winning score with the arrows"

    def __init__(self, width: int, height: int, seed=None):
        if width < W or height < H:
            raise ValueError("Panel Hockey needs a 768x128 display")
        self.width, self.height = width, height
        self.rng = random.Random(seed)
        self.t = 0.0
        self.lobby = Lobby()
        self.win_i = 2
        self.state = "lobby"
        self.joined: set[int] = set()
        self.prev: dict[int, int] = {}
        self.target: dict[int, tuple[float, float]] = {}
        self._reset_game()

    def _reset_game(self):
        self.scores = [0, 0]
        self.winner: int | None = None
        self.paddles: dict[int, list[float]] = {}  # key -> x, y, vx, vy. Slots are humans, 100 + team is the computer
        self.timer = 0.0
        self.last_scorer: int | None = None
        self.stall = 0.0
        self._put_puck(self.rng.choice((0, 1)))

    def _put_puck(self, toward: int, speed: float = 0.0):
        self.px, self.py = MID + 0.0, H / 2
        ang = self.rng.uniform(-0.6, 0.6)
        d = -1 if toward == 0 else 1
        self.pvx, self.pvy = d * speed * math.cos(ang), speed * math.sin(ang)

    @property
    def target_score(self) -> int:
        return WINS[self.win_i]

    # ---- information shared with phones

    def info(self) -> dict:
        return {"state": self.state, "host": self.lobby.host, "scores": list(self.scores), "to_win": self.target_score,
                "winner": self.winner if self.state == "over" else None}

    def private_info(self, slot: int) -> dict:
        return {"team": team_of(slot)}

    def on_message(self, slot: int, data: dict):
        if data.get("t") != "move" or slot not in self.joined:
            return
        u, v = data.get("x"), data.get("y")
        if not all(isinstance(a, (int, float)) and not isinstance(a, bool) and math.isfinite(a) for a in (u, v)):
            return
        lo, hi = half_bounds(team_of(slot))
        self.target[slot] = (lo + max(0.0, min(1.0, u)) * (hi - lo), R_PADDLE + max(0.0, min(1.0, v)) * (H - 2 * R_PADDLE))

    # ---- paddles

    def _sync_paddles(self):
        humans = {team: [s for s in sorted(self.joined) if team_of(s) == team] for team in (0, 1)}
        wanted = {s for t in (0, 1) for s in humans[t]} | {100 + t for t in (0, 1) if not humans[t]}
        for key in list(self.paddles):
            if key not in wanted:
                del self.paddles[key]
                self.target.pop(key, None)
        for team in (0, 1):
            keys = humans[team] or [100 + team]
            for i, key in enumerate(keys):
                if key not in self.paddles:
                    lo, hi = half_bounds(team)
                    x = lo + (hi - lo) * (0.15 if team == 0 else 0.85)
                    y = H * (0.5 if len(keys) == 1 else 0.3 + 0.4 * (i % 2))
                    self.paddles[key] = [x, y, 0.0, 0.0]
                    if key < 100:
                        self.target[key] = (x, y)

    def _ai_target(self, team: int, p: list[float]) -> tuple[float, float]:
        d = 1 if team == 0 else -1
        home = (40.0 if team == 0 else W - 40.0, H / 2 + (self.py - H / 2) * 0.45)
        if (self.px < MID) != (team == 0) or self.state != "play":
            return home
        gx, gy = (W if team == 0 else 0), H / 2
        dist = math.hypot(gx - self.px, gy - self.py) or 1.0
        ux, uy = (gx - self.px) / dist, (gy - self.py) / dist
        gap = R_PADDLE + R_PUCK
        if (self.px - p[0]) * d < 0:  # puck is between us and our own goal: swing round it
            side = 1 if p[1] >= self.py else -1
            return self.px - d * (gap + 6), self.py + side * (gap + 12)
        behind = (self.px - ux * (gap + 5), self.py - uy * (gap + 5))
        if math.hypot(p[0] - behind[0], p[1] - behind[1]) < 7:
            return self.px + ux * 40, self.py + uy * 40
        return behind

    def _move_paddles(self, dts: float):
        for key, p in self.paddles.items():
            team = team_of(key) if key < 100 else key - 100
            if key >= 100:
                tx, ty = self._ai_target(team, p)
                speed = AI_SPEED
            else:
                tx, ty = self.target.get(key, (p[0], p[1]))
                speed = PADDLE_SPEED
            lo, hi = half_bounds(team)
            tx, ty = max(lo, min(hi, tx)), max(R_PADDLE, min(H - R_PADDLE, ty))
            dx, dy = tx - p[0], ty - p[1]
            dist = math.hypot(dx, dy)
            step = min(dist, speed * dts)
            nx, ny = (p[0] + dx / dist * step, p[1] + dy / dist * step) if dist > 1e-9 else (p[0], p[1])
            p[2], p[3] = (nx - p[0]) / dts, (ny - p[1]) / dts
            p[0], p[1] = nx, ny

    # ---- puck

    def _step_puck(self, dts: float):
        self.px += self.pvx * dts
        self.py += self.pvy * dts
        for p in self.paddles.values():
            dx, dy = self.px - p[0], self.py - p[1]
            dist = math.hypot(dx, dy)
            reach = R_PADDLE + R_PUCK
            if dist < reach:
                nx, ny = (dx / dist, dy / dist) if dist > 1e-9 else (1.0, 0.0)
                self.px, self.py = p[0] + nx * reach, p[1] + ny * reach
                vn = (self.pvx - p[2]) * nx + (self.pvy - p[3]) * ny
                if vn < 0:
                    self.pvx -= (1 + BOUNCE) * vn * nx
                    self.pvy -= (1 + BOUNCE) * vn * ny
        if self.py < R_PUCK:
            self.py, self.pvy = R_PUCK, abs(self.pvy) * BOUNCE
        elif self.py > H - R_PUCK:
            self.py, self.pvy = H - R_PUCK, -abs(self.pvy) * BOUNCE
        in_mouth = GOAL_TOP + R_PUCK <= self.py <= GOAL_BOTTOM - R_PUCK
        if self.px < R_PUCK and not (in_mouth and self.pvx < 0):
            self.px, self.pvx = R_PUCK, abs(self.pvx) * BOUNCE
        elif self.px > W - R_PUCK and not (in_mouth and self.pvx > 0):
            self.px, self.pvx = W - R_PUCK, -abs(self.pvx) * BOUNCE
        speed = math.hypot(self.pvx, self.pvy)
        if speed > MAX_PUCK:
            self.pvx, self.pvy = self.pvx * MAX_PUCK / speed, self.pvy * MAX_PUCK / speed
        decay = max(0.0, 1.0 - FRICTION * dts)
        self.pvx, self.pvy = self.pvx * decay, self.pvy * decay

    def _goal(self, scorer: int):
        self.scores[scorer] += 1
        self.last_scorer = scorer
        if self.scores[scorer] >= self.target_score:
            self.state, self.timer, self.winner = "over", OVER_TIME, scorer
        else:
            self.state, self.timer = "goal", GOAL_TIME
        self.pvx = self.pvy = 0.0

    def _unstick(self, dt: float):
        if math.hypot(self.pvx, self.pvy) < 8:
            self.stall += dt
            if self.stall > 3.0:
                self.pvx += (MID - self.px) * 0.8
                self.pvy += (H / 2 - self.py) * 0.8
                self.stall = 0.0
        else:
            self.stall = 0.0

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
                self.win_i = max(0, min(len(WINS) - 1, self.win_i + bool(pressed[host] & UP) - bool(pressed[host] & DOWN)))
            if self.lobby.update(joined, pads):
                self._reset_game()
                self._sync_paddles()
                self.state, self.timer = "serve", SERVE_TIME
            return
        if not joined:
            self.state = "lobby"
            self.lobby.reset()
            self._reset_game()
            return
        self._sync_paddles()
        for _ in range(SUBSTEPS):
            self._move_paddles(dt / SUBSTEPS)
            if self.state == "play":
                self._step_puck(dt / SUBSTEPS)
                if self.px < -R_PUCK:
                    return self._goal(1)
                if self.px > W + R_PUCK:
                    return self._goal(0)
        if self.state == "play":
            self._unstick(dt)
        elif self.state in ("serve", "goal", "over"):
            self.timer -= dt
            if self.timer <= 0:
                if self.state == "over":
                    self.state = "lobby"
                    self.lobby.reset()
                    self._reset_game()
                else:
                    conceder = self.rng.choice((0, 1)) if self.last_scorer is None else 1 - self.last_scorer
                    self._put_puck(conceder, 90.0)
                    self.state = "play"

    # ---- drawing

    def _rink(self, canvas):
        canvas[:] = BG
        cv2.rectangle(canvas, (0, 0), (W - 1, H - 1), LINE, 2)
        canvas[:, MID - 1 : MID + 1] = LINE
        cv2.circle(canvas, (MID, H // 2), 26, LINE, 1)
        for team, x in ((0, 0), (1, W - 4)):
            canvas[GOAL_TOP:GOAL_BOTTOM, x : x + 4] = tuple(c // 2 + 30 for c in TEAM_COLORS[team])
            cv2.line(canvas, (x, GOAL_TOP), (x + 3, GOAL_TOP), TEAM_COLORS[team], 2)
            cv2.line(canvas, (x, GOAL_BOTTOM), (x + 3, GOAL_BOTTOM), TEAM_COLORS[team], 2)

    def _render_lobby(self, canvas):
        self._rink(canvas)
        center_text(canvas, "PANEL HOCKEY", 44, 1.3, (255, 230, 0), 2, box=False)
        center_text(canvas, "Slide your mallet with your finger. Knock the puck into the other goal.", 66, 0.5, (255, 255, 255), box=False)
        center_text(canvas, f"FIRST TO {self.target_score}  (host: UP / DOWN)", 84, 0.5, (120, 220, 255), box=False)
        center_text(canvas, self.lobby.message, 112, 0.7, (120, 255, 120) if self.joined else (255, 255, 255), box=False)
        for i, s in enumerate(sorted(self.joined)):
            cv2.circle(canvas, (60 + i * 22, 104 - 8), 7, TEAM_COLORS[team_of(s)], -1)

    def render(self, canvas: np.ndarray):
        if self.state == "lobby":
            return self._render_lobby(canvas)
        self._rink(canvas)
        for team, x in ((0, MID - 70), (1, MID + 40)):
            cv2.putText(canvas, str(self.scores[team]), (x, 44), FONT, 1.4, tuple(c // 2 for c in TEAM_COLORS[team]), 2, cv2.LINE_AA)
        for key, p in self.paddles.items():
            team = team_of(key) if key < 100 else key - 100
            c = (int(p[0]), int(p[1]))
            cv2.circle(canvas, c, R_PADDLE, TEAM_COLORS[team], -1, cv2.LINE_AA)
            cv2.circle(canvas, c, R_PADDLE - 4, PLAYER_COLORS[key] if key < 100 else (60, 60, 60), -1, cv2.LINE_AA)
            cv2.circle(canvas, c, 2, (255, 255, 255), -1)
        cv2.circle(canvas, (int(self.px), int(self.py)), R_PUCK, (255, 255, 255), -1, cv2.LINE_AA)
        if self.state == "serve":
            center_text(canvas, "GET READY", 68, 0.8, (255, 230, 0), 2, box=False)
        elif self.state == "goal":
            center_text(canvas, f"{TEAMS[self.last_scorer]} SCORES", 72, 1.1, TEAM_COLORS[self.last_scorer], 2, box=False)
        elif self.state == "over":
            canvas[34:96] = canvas[34:96] // 3
            side = TEAMS[self.winner]
            humans = any(team_of(s) == self.winner for s in self.joined)
            center_text(canvas, f"{side} WINS" if humans else "COMPUTER WINS", 70, 1.2, TEAM_COLORS[self.winner], 2, box=False)
            center_text(canvas, f"{self.scores[0]} - {self.scores[1]}", 92, 0.6, (255, 255, 255), box=False)

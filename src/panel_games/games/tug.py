"""Panel Tug of War: two teams tap their phones as fast as they can to drag a rope across the display.

Taps arrive as messages, not button presses, so rapid tapping is never lost between frames. Each team's pull is
its taps divided by its size, so uneven teams stay fair. Drag the flag to the far line to win a round at once,
otherwise whoever holds the flag on their side when the timer ends wins it.
"""

import math
import random

import cv2
import numpy as np

from .base import FONT, PLAYER_COLORS, Lobby, center_text, text
from .pad import DOWN, LEFT, RIGHT, UP

TEAMS = ("RED", "BLUE")
TEAM_COLORS = ((255, 70, 70), (70, 140, 255))
SECONDS = [10, 15, 20, 30]
BEST_OF = [1, 3, 5]
READY_TIME, END_TIME, OVER_TIME, OVERTIME = 3.0, 4.0, 14.0, 5.0
PULL = 0.04  # rope travel per tap per team member, where the whole rope length is 2.0
TAP_RATE, TAP_BURST = 12.0, 6.0  # most taps per second a player can land, as a bot or API user would try
MID, SPAN = 384, 240
ROPE_Y, GROUND_Y = 76, 104


def _width(s: str, scale: float, thickness: int = 1) -> int:
    return cv2.getTextSize(s, FONT, scale, thickness)[0][0]


class Tug:
    id = "tug"
    title = "Tug of War"
    description = "Two teams tap their phones as fast as they can to drag the rope to their side. Best of several rounds."
    max_players = 8
    size = (768, 128)
    layout = "tug"
    controls = "Tap the big button as fast as you can. The host picks round length and best-of with the arrows"

    def __init__(self, width: int, height: int, seed=None):
        if width < 768 or height < 128:
            raise ValueError("Tug of War needs a 768x128 display")
        self.width, self.height = width, height
        self.rng = random.Random(seed)
        self.t = 0.0
        self.lobby = Lobby()
        self.secs_i, self.best_i = 2, 1
        self.state = "lobby"
        self.joined: set[int] = set()
        self.prev: dict[int, int] = {}
        self.need_flash = 0.0
        self._reset_game()

    def _reset_game(self):
        self.team: dict[int, int] = {}
        self.wins = [0, 0]
        self.round_i = 0
        self.total_taps: dict[int, int] = {}
        self.winner: int | None = None
        self._new_round()

    def _new_round(self):
        self.pos = 0.0
        self.remaining = float(SECONDS[self.secs_i])
        self.overtime = False
        self.round_taps = {s: 0 for s in self.team}
        self.credit = {s: TAP_BURST for s in self.team}
        self.pending = {s: 0 for s in self.team}
        self.last_tap: dict[int, float] = {}
        self.timer = 0.0

    @property
    def need(self) -> int:
        return BEST_OF[self.best_i] // 2 + 1

    def members(self, team: int) -> list[int]:
        return sorted(s for s, t in self.team.items() if t == team and s in self.joined)

    # ---- information shared with phones

    def info(self) -> dict:
        return {"state": self.state, "host": self.lobby.host, "pos": round(self.pos, 2), "wins": list(self.wins),
                "need": self.need, "round": self.round_i + 1, "secs": SECONDS[self.secs_i], "best_of": BEST_OF[self.best_i],
                "remaining": int(self.remaining + 0.99) if self.state == "pull" else None,
                "countdown": int(self.timer + 0.99) if self.state == "ready" else None,
                "overtime": self.overtime, "winner": self.winner if self.state in ("round_end", "over") else None}

    def private_info(self, slot: int) -> dict:
        return {"team": None if self.state == "lobby" else self.team.get(slot),
                "taps": self.round_taps.get(slot, 0), "total": self.total_taps.get(slot, 0)}

    def on_message(self, slot: int, data: dict):
        n = data.get("n")
        if data.get("t") != "tap" or self.state != "pull" or slot not in self.team or slot not in self.joined:
            return
        if isinstance(n, bool) or not isinstance(n, int) or n < 1:
            return
        take = min(n, 8, int(self.credit.get(slot, 0)))
        if take > 0:
            self.credit[slot] -= take
            self.pending[slot] = self.pending.get(slot, 0) + take

    # ---- flow

    def _start_game(self, joined):
        self._reset_game()
        order = sorted(joined)
        self.rng.shuffle(order)
        self.team = {s: i % 2 for i, s in enumerate(order)}
        self.total_taps = {s: 0 for s in order}
        self._begin_round()

    def _begin_round(self):
        self._new_round()
        self.state, self.timer = "ready", READY_TIME

    def _end_round(self, winner: int):
        self.wins[winner] += 1
        self.winner = winner
        self.state, self.timer = "round_end", END_TIME

    def update(self, dt: float, pads: list):
        self.t += dt
        self.need_flash = max(0.0, self.need_flash - dt)
        joined = {i for i, k in enumerate(pads) if k is not None}
        keys = {i: (pads[i] or 0) for i in joined}
        pressed = {i: k & ~self.prev.get(i, 0) for i, k in keys.items()}
        self.prev = keys
        self.joined = joined
        if self.state == "lobby":
            host = self.lobby.host
            if host is not None and host in pressed:
                p = pressed[host]
                self.secs_i = max(0, min(len(SECONDS) - 1, self.secs_i + bool(p & UP) - bool(p & DOWN)))
                self.best_i = max(0, min(len(BEST_OF) - 1, self.best_i + bool(p & RIGHT) - bool(p & LEFT)))
            if self.lobby.update(joined, pads):
                if len(joined) >= 2:
                    self._start_game(joined)
                else:
                    self.need_flash = 2.5
            return
        if not joined:
            self.state = "lobby"
            self.lobby.reset()
            self._reset_game()
            return
        if self.state == "ready":
            self._check_forfeit()
            self.timer -= dt
            if self.timer <= 0 and self.state == "ready":
                self.state = "pull"
        elif self.state == "pull":
            self._pull(dt)
        elif self.state == "round_end":
            self.timer -= dt
            if self.timer <= 0:
                if max(self.wins) >= self.need:
                    self.state, self.timer = "over", OVER_TIME
                else:
                    self.round_i += 1
                    self._begin_round()
        else:
            self.timer -= dt
            if self.timer <= 0:
                self.state = "lobby"
                self.lobby.reset()
                self._reset_game()

    def _check_forfeit(self):
        for team in (0, 1):
            if not self.members(team):
                self._end_round(1 - team)
                return

    def _pull(self, dt: float):
        self._check_forfeit()
        if self.state != "pull":
            return
        force = [0.0, 0.0]
        for s, team in self.team.items():
            n = self.pending.get(s, 0)
            self.pending[s] = 0
            self.credit[s] = min(TAP_BURST, self.credit.get(s, 0) + dt * TAP_RATE)
            if n and s in self.joined:
                force[team] += n / len(self.members(team))
                self.round_taps[s] += n
                self.total_taps[s] += n
                self.last_tap[s] = self.t
        self.pos = max(-1.0, min(1.0, self.pos + (force[1] - force[0]) * PULL))
        if abs(self.pos) >= 1.0:
            return self._end_round(0 if self.pos < 0 else 1)
        self.remaining -= dt
        if self.remaining <= 0:
            if abs(self.pos) < 0.01:
                self.remaining, self.overtime = OVERTIME, True
            else:
                self._end_round(0 if self.pos < 0 else 1)

    # ---- drawing

    def _figure(self, canvas, x: int, team: int, slot: int):
        dir_ = -1 if team == 0 else 1  # the side this player is pulling toward
        bob = 3 if self.t - self.last_tap.get(slot, -9.0) < 0.12 else 0
        color = TEAM_COLORS[team]
        for foot in (-6, 6):
            cv2.line(canvas, (x, 88), (x + foot, GROUND_Y), color, 2)
        shoulder = (x + dir_ * 8, 70 - bob)
        cv2.line(canvas, (x, 88), shoulder, color, 3)
        cv2.line(canvas, shoulder, (x - dir_ * 10, ROPE_Y), color, 2)
        cv2.circle(canvas, (x + dir_ * 11, 63 - bob), 5, PLAYER_COLORS[slot], -1)
        label = f"P{slot + 1}: {self.round_taps.get(slot, 0) if self.state != 'over' else self.total_taps.get(slot, 0)}"
        text(canvas, label, max(0, min(self.width - 40, x - _width(label, 0.33) // 2)), 120, 0.33, color)

    def _scene(self, canvas):
        canvas[:] = (16, 20, 44)
        canvas[GROUND_Y:] = (34, 60, 34)
        canvas[GROUND_Y : GROUND_Y + 2] = (70, 110, 60)
        for team, sign in ((0, -1), (1, 1)):
            x = MID + sign * SPAN
            canvas[30:GROUND_Y, x - 1 : x + 1] = tuple(c // 2 for c in TEAM_COLORS[team])
        for y in range(30, GROUND_Y, 8):
            canvas[y : y + 4, MID - 1 : MID + 1] = (200, 200, 200)
        cv2.line(canvas, (0, ROPE_Y), (self.width - 1, ROPE_Y), (190, 150, 90), 3)
        fx = int(MID + self.pos * SPAN)
        for team in (0, 1):
            for i, s in enumerate(self.members(team)):
                self._figure(canvas, fx + (-1 if team == 0 else 1) * (40 + i * 30), team, s)
        cv2.line(canvas, (fx, ROPE_Y - 20), (fx, ROPE_Y + 3), (240, 240, 240), 2)
        lead = TEAM_COLORS[0] if self.pos < -0.01 else TEAM_COLORS[1] if self.pos > 0.01 else (255, 230, 90)
        cv2.fillPoly(canvas, [np.array([(fx, ROPE_Y - 20), (fx, ROPE_Y - 10), (fx + 12, ROPE_Y - 15)], np.int32)], lead)

    def _hud(self, canvas):
        text(canvas, TEAMS[0], 8, 17, 0.6, TEAM_COLORS[0], 2)
        label_w = _width(TEAMS[1], 0.6, 2)
        text(canvas, TEAMS[1], self.width - 8 - label_w, 17, 0.6, TEAM_COLORS[1], 2)
        for team in (0, 1):
            for k in range(self.need):
                x = 70 + k * 16 if team == 0 else self.width - 76 - label_w - k * 16
                cv2.circle(canvas, (x, 12), 5, TEAM_COLORS[team], -1 if k < self.wins[team] else 1)
        if self.state == "pull":
            rem = max(0, int(self.remaining + 0.99))
            color = (255, 90, 90) if rem <= 5 else (255, 255, 255)
            s = f"{rem}"
            text(canvas, s, (self.width - _width(s, 1.0, 2)) // 2, 24, 1.0, color, 2)
            if self.overtime:
                center_text(canvas, "OVERTIME", 44, 0.4, (255, 200, 60), box=False)

    def _render_lobby(self, canvas):
        canvas[:] = (16, 20, 44)
        canvas[:, : self.width // 2, 0] = np.minimum(canvas[:, : self.width // 2, 0] + 18, 255)
        center_text(canvas, "TUG OF WAR", 38, 1.3, (255, 230, 0), 2, box=False)
        center_text(canvas, "Two teams. Tap your phone as fast as you can.", 62, 0.55, (255, 255, 255), box=False)
        center_text(canvas, f"BEST OF {BEST_OF[self.best_i]}   {SECONDS[self.secs_i]} SECONDS A ROUND", 82, 0.5, (120, 220, 255), box=False)
        center_text(canvas, "host: LEFT/RIGHT rounds, UP/DOWN seconds", 94, 0.34, (150, 150, 190), box=False)
        msg = "NEED AT LEAST 2 PLAYERS" if self.need_flash > 0 else self.lobby.message
        center_text(canvas, msg, 116, 0.7, (255, 120, 120) if self.need_flash > 0 else (120, 255, 120) if self.joined else (255, 255, 255), box=False)
        for i, s in enumerate(sorted(self.joined)):
            canvas[100:108, 12 + i * 14 : 22 + i * 14] = PLAYER_COLORS[s]

    def _render_over(self, canvas):
        canvas[:] = (10, 8, 30)
        for i in range(70):
            x, y = (i * 97 + int(self.t * 10) * (i % 3)) % self.width, int(self.t * 45 + i * 37) % self.height
            canvas[y : y + 2, x : x + 2] = TEAM_COLORS[self.winner] if i % 2 else (255, 230, 0)
        center_text(canvas, f"TEAM {TEAMS[self.winner]} WINS", 36, 1.2, TEAM_COLORS[self.winner], 2, box=False)
        center_text(canvas, f"{self.wins[0]} - {self.wins[1]}", 60, 0.7, (255, 255, 255), box=False)
        for team, x0 in ((0, 40), (1, self.width - 220)):
            for i, s in enumerate(self.members(team)):
                text(canvas, f"P{s + 1}  {self.total_taps.get(s, 0)} taps", x0, 80 + i * 12, 0.4, TEAM_COLORS[team])
        if self.total_taps:
            mvp = max(self.total_taps, key=lambda s: self.total_taps[s])
            center_text(canvas, f"MOST TAPS: P{mvp + 1} ({self.total_taps[mvp]})", 108, 0.5, (255, 230, 90), box=False)

    def render(self, canvas: np.ndarray):
        if self.state == "lobby":
            return self._render_lobby(canvas)
        if self.state == "over":
            return self._render_over(canvas)
        self._scene(canvas)
        self._hud(canvas)
        if self.state == "ready":
            canvas[34:96] = canvas[34:96] // 3
            center_text(canvas, "GET READY", 52, 0.6, (255, 255, 255), box=False)
            center_text(canvas, str(max(1, int(self.timer + 0.99))), 90, 1.6, (255, 230, 0), 3, box=False)
        elif self.state == "pull" and self.remaining > SECONDS[self.secs_i] - 1.0:
            center_text(canvas, "TAP TAP TAP!", 56, 1.0, (255, 230, 0), 2, box=False)
        elif self.state == "round_end":
            canvas[34:96] = canvas[34:96] // 3
            center_text(canvas, f"{TEAMS[self.winner]} WINS THE ROUND", 72, 1.0, TEAM_COLORS[self.winner], 2, box=False)

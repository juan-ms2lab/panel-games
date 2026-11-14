"""Pong for two players; the computer plays any empty side."""

import random

import cv2
import numpy as np

from .pad import DOWN, UP

PW, PH, BALL = 4, 24, 5
PADDLE_SPEED, AI_SPEED = 170.0, 105.0
SERVE, MAX_SPEED = 200.0, 520.0
WIN_SCORE = 7
COLORS = [(0, 200, 255), (255, 140, 0)]


class Pong:
    id = "pong"
    title = "Pong"
    description = "Two players, one ball. First to 7 wins. The computer plays any empty side."
    max_players = 2
    size = (768, 128)
    controls = "Up / Down to move your paddle"

    def __init__(self, width: int, height: int, seed=None):
        self.rng = random.Random(seed)
        self.aim = [0.0, 0.0]  # where on its paddle each computer player tries to meet the ball
        self.width, self.height = width, height
        self.scores = [0, 0]
        self.paddles = [(height - PH) / 2.0] * 2
        self.joined = [False, False]
        self.state, self.timer, self.winner = "play", 0.0, None
        self.serve(1)

    def serve(self, direction: int):
        self.bx, self.by = self.width / 2, self.height / 2
        self.vx, self.vy = SERVE * direction, 60.0 * (1 if self.scores[0] + self.scores[1] & 1 else -1)

    def _paddle_x(self, side):
        return 8 if side == 0 else self.width - 8 - PW

    def update(self, dt: float, pads: list):
        self.joined = [k is not None for k in pads] + [False] * (2 - len(pads))
        if self.state != "play":
            self.timer -= dt
            if self.timer <= 0:
                self.scores, self.state = [0, 0], "play"
                self.serve(1)
            return
        for side in range(2):
            keys = pads[side] if side < len(pads) else None
            if keys is not None:
                move = (1 if keys & DOWN else 0) - (1 if keys & UP else 0)
                self.paddles[side] += move * PADDLE_SPEED * dt
            else:
                diff = (self.by + BALL / 2) - (self.paddles[side] + PH / 2) + self.aim[side]
                self.paddles[side] += max(-AI_SPEED * dt, min(AI_SPEED * dt, diff))
            self.paddles[side] = max(0.0, min(self.height - PH, self.paddles[side]))
        self.bx += self.vx * dt
        self.by += self.vy * dt
        if self.by < 0 or self.by > self.height - BALL:
            self.by = max(0.0, min(self.height - BALL, self.by))
            self.vy = -self.vy
        for side in range(2):
            px = self._paddle_x(side)
            toward = self.vx < 0 if side == 0 else self.vx > 0
            if toward and px - BALL < self.bx < px + PW and self.paddles[side] - BALL < self.by < self.paddles[side] + PH:
                offset = (self.by + BALL / 2 - self.paddles[side] - PH / 2) / (PH / 2)
                speed = min(MAX_SPEED, abs(self.vx) * 1.07)
                self.vx = speed if side == 0 else -speed
                self.vy = offset * 200.0
                self.bx = px + PW if side == 0 else px - BALL
                self.aim[side] = self.rng.uniform(-0.9, 0.9) * PH / 2
        if self.bx < -BALL or self.bx > self.width:
            scorer = 1 if self.bx < 0 else 0
            self.scores[scorer] += 1
            if self.scores[scorer] >= WIN_SCORE:
                self.state, self.timer, self.winner = "over", 4.0, scorer
            self.serve(-1 if scorer else 1)

    def render(self, canvas: np.ndarray):
        canvas[:] = 0
        for y in range(0, self.height, 8):
            canvas[y : y + 4, self.width // 2 - 1 : self.width // 2 + 1] = 70
        for side in range(2):
            x = self._paddle_x(side)
            y = int(self.paddles[side])
            canvas[y : y + PH, x : x + PW] = COLORS[side] if self.joined[side] else (110, 110, 110)
        canvas[int(self.by) : int(self.by) + BALL, int(self.bx) : int(self.bx) + BALL] = 255
        for side in range(2):
            cv2.putText(canvas, str(self.scores[side]), (self.width // 2 + (-70 if side == 0 else 40), 44),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.4, COLORS[side], 2, cv2.LINE_AA)
        if self.state == "over":
            text = f"PLAYER {self.winner + 1} WINS" if self.joined[self.winner] else "COMPUTER WINS"
            (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
            cv2.putText(canvas, text, ((self.width - tw) // 2, 76), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 230, 0), 2, cv2.LINE_AA)

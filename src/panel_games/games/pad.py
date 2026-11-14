"""Shared controller state: player slots, join tokens and button masks sent from phones."""

import secrets
import threading
import time

# Button bits sent by the phone controller.
LEFT, RIGHT, UP, DOWN, A, B, START, X, Y = 1, 2, 4, 8, 16, 32, 64, 128, 256

SLOT_TIMEOUT = 20.0  # seconds without a heartbeat before a slot is freed
KEY_TIMEOUT = 1.0    # held buttons are released if updates stop (phone lost signal)


class Pad:
    def __init__(self):
        self._lock = threading.Lock()
        self.game: str | None = None
        self._slots: list[dict] = []
        self._tokens: dict[str, int] = {}
        self._inbox: list[tuple[int, dict]] = []
        self.last_input = time.monotonic()

    def start(self, game: str, max_players: int):
        with self._lock:
            self.game = game
            self._slots = [{"token": None, "last": 0.0, "keys": 0, "keys_at": 0.0} for _ in range(max_players)]
            self._tokens = {}
            self._inbox = []
            self.last_input = time.monotonic()

    def stop(self):
        with self._lock:
            self.game, self._slots, self._tokens, self._inbox = None, [], {}, []

    def _expire(self, now: float):
        for i, s in enumerate(self._slots):
            if s["token"] and now - s["last"] > SLOT_TIMEOUT:
                self._tokens.pop(s["token"], None)
                s.update(token=None, keys=0)

    def join(self, owner: str | None = None) -> tuple[int, str]:
        """Take a free slot. A repeat join by the same owner returns its existing slot."""
        with self._lock:
            if not self.game:
                raise ValueError("no game is running")
            now = time.monotonic()
            self._expire(now)
            if owner:
                for i, s in enumerate(self._slots):
                    if s["token"] and s.get("owner") == owner:
                        s["last"] = now
                        return i, s["token"]
            for i, s in enumerate(self._slots):
                if s["token"] is None:
                    token = secrets.token_urlsafe(12)
                    s.update(token=token, owner=owner, last=now, keys=0, keys_at=now)
                    self._tokens[token] = i
                    self.last_input = now
                    return i, token
            raise ValueError("the game is full")

    def set_keys(self, token: str, keys: int):
        if isinstance(keys, bool) or not isinstance(keys, int) or not 0 <= keys <= 511:
            raise ValueError("keys must be a whole number 0-511")
        with self._lock:
            i = self._tokens.get(token)
            if i is None:
                raise ValueError("unknown player, join again")
            s, now = self._slots[i], time.monotonic()
            if keys or keys != s["keys"]:
                self.last_input = now
            s.update(keys=keys, keys_at=now, last=now)

    def slot_of(self, token: str) -> int:
        with self._lock:
            i = self._tokens.get(token)
        if i is None:
            raise ValueError("unknown player, join again")
        return i

    def put_message(self, token: str, data) -> None:
        """Queue a free-form message (drawing strokes, typed guesses) from a player for the running game."""
        if not isinstance(data, dict):
            raise ValueError("message must be an object")
        with self._lock:
            i = self._tokens.get(token)
            if i is None:
                raise ValueError("unknown player, join again")
            if len(self._inbox) >= 400:
                raise ValueError("too many messages, slow down")
            now = time.monotonic()
            self._inbox.append((i, data))
            self._slots[i]["last"] = now
            self.last_input = now

    def drain(self) -> list[tuple[int, dict]]:
        with self._lock:
            out, self._inbox = self._inbox, []
        return out

    def leave(self, token: str):
        with self._lock:
            i = self._tokens.pop(token, None)
            if i is not None:
                self._slots[i].update(token=None, keys=0)

    def snapshot(self) -> list[int | None]:
        """Per slot: None if nobody joined, else the buttons currently held."""
        with self._lock:
            now = time.monotonic()
            self._expire(now)
            return [None if s["token"] is None else (s["keys"] if now - s["keys_at"] < KEY_TIMEOUT else 0)
                    for s in self._slots]

    def idle_seconds(self) -> float:
        return time.monotonic() - self.last_input

    def status(self) -> dict | None:
        with self._lock:
            if not self.game:
                return None
            return {"id": self.game, "joined": sum(1 for s in self._slots if s["token"]),
                    "max_players": len(self._slots)}


PAD = Pad()

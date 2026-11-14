"""Runs one game at a time on a fixed-rate loop thread and pushes its frames to every output."""

import threading
import time

import numpy as np

from . import games
from .games.pad import PAD

STEP = 1 / 60
IDLE_SECONDS = 120  # a game with no button presses for this long ends and the outputs blank


class Host:
    def __init__(self, outputs, target=(768, 128), fps: float = 60):
        self.outputs = list(outputs)
        self.target = tuple(target)
        self.frame_time = 1 / fps
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._worker = None
        self.game = None
        self.error = None

    def catalog(self):
        return games.catalog(self.target)

    def start(self, game_id: str):
        cls = games.REGISTRY.get(game_id)
        if cls is None:
            raise ValueError(f"game must be one of {sorted(games.REGISTRY)}")
        size = games.pick_size(cls, *self.target)
        if size is None:
            raise ValueError(f"{cls.title} needs {' or '.join(f'{w}x{h}' for w, h in games.sizes(cls))}; "
                             f"the output is {self.target[0]}x{self.target[1]}")
        with self._lock:
            self._halt()
            self.error = None
            game = cls(*size)
            PAD.start(game_id, game.max_players)
            self.game = game
            stop = self._stop = threading.Event()
            self._worker = threading.Thread(target=self._run, args=(game, size, stop), daemon=True)
            self._worker.start()

    def stop(self):
        with self._lock:
            self._halt()

    def _halt(self):
        self._stop.set()
        w, self._worker = self._worker, None
        if w and w is not threading.current_thread():
            w.join()

    def _run(self, game, size, stop):
        canvas = np.zeros((size[1], size[0], 3), np.uint8)
        try:
            last, acc, shown = time.monotonic(), 0.0, 0.0
            while not stop.is_set():
                now = time.monotonic()
                acc = min(acc + now - last, 0.25)
                last = now
                if hasattr(game, "on_message"):
                    for slot, data in PAD.drain():
                        game.on_message(slot, data)
                while acc >= STEP:
                    game.update(STEP, PAD.snapshot())
                    acc -= STEP
                if now - shown >= self.frame_time - 0.002:
                    shown = now
                    game.render(canvas)
                    for o in self.outputs:
                        o.show(canvas)
                if PAD.idle_seconds() > IDLE_SECONDS:
                    return
                stop.wait(max(0.0, STEP - (time.monotonic() - now)))
        except Exception as e:
            self.error = f"{type(e).__name__}: {e}"
        finally:
            if self.game is game:
                self.game = None
                PAD.stop()
            for o in self.outputs:
                o.clear()

    def status(self):
        s = PAD.status()
        game = self.game
        if s and game is not None:
            s["title"] = type(game).title
            if hasattr(game, "info"):
                s.update(game.info())  # e.g. {"state": "lobby", "host": 0}
        return s

    def private_state(self, token: str) -> dict:
        """Public game info plus whatever only this player may see (the word to draw, their own feedback)."""
        slot = PAD.slot_of(token)
        out = dict(self.status() or {})
        out["slot"] = slot
        game = self.game
        if game is not None and hasattr(game, "private_info"):
            out.update(game.private_info(slot))
        return out

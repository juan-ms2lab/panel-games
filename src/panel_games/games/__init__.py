"""Games played from phone controllers. Each game is a class with update(dt, pads) and render(canvas).

Every game declares the canvas it renders at with a `size` class attribute: one (w, h) tuple, or a list of
them in order of preference. The host builds the game at the first size that fits the output target.
"""

from .bomber import Bombers
from .chomp import Chomp
from .flap import Flap
from .elbirdo import ElBirdo
from .hockey import Hockey
from .hop import Hop
from .pad import PAD
from .platformer import Platformer
from .pong import Pong
from .sketch import Sketch
from .sweep import Sweep
from .trivia import Trivia
from .tug import Tug

REGISTRY = {g.id: g for g in (Platformer, Hop, Flap, ElBirdo, Chomp, Bombers, Sweep, Trivia, Sketch, Tug, Hockey, Pong)}


def sizes(game) -> list[tuple[int, int]]:
    s = game.size
    return [tuple(s)] if isinstance(s[0], int) else [tuple(x) for x in s]


def pick_size(game, target_w: int, target_h: int) -> tuple[int, int] | None:
    """The first declared size that fits inside the target, or None."""
    return next(((w, h) for w, h in sizes(game) if w <= target_w and h <= target_h), None)


def catalog(target=(768, 128)) -> list[dict]:
    out = []
    for g in REGISTRY.values():
        size = pick_size(g, *target)
        out.append({"id": g.id, "title": g.title, "description": g.description, "max_players": g.max_players,
                    "controls": g.controls, "layout": getattr(g, "layout", "dpad"),
                    "sizes": [f"{w}x{h}" for w, h in sizes(g)], "available": size is not None,
                    "size": f"{size[0]}x{size[1]}" if size else None})
    return out

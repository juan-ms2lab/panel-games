"""Games played from phone controllers. Each game is a class with update(dt, pads) and render(canvas).

Every game declares the canvas it renders at with a `size` class attribute: one (w, h) tuple, or a list of
them in order of preference. The host builds the game at the first size that fits the output target.
A game may also declare `bezel` settings (see bezel.py). Both can be overridden per game in the config file:

    [default.bezel]          # every game
    aspect = "16:9"
    [resurrection]                # one game, by id
    size = [768, 64]
    [resurrection.bezel]
    colors = ["#141022", "#000000"]
"""

from .. import bezel as bz
from .bomber import Bombers
from .chomp import Chomp
from .flap import Flap
from .resurrection import Resurrection
from .hockey import Hockey
from .hop import Hop
from .pad import PAD
from .platformer import Platformer
from .pong import Pong
from .sketch import Sketch
from .sweep import Sweep
from .trivia import Trivia
from .tug import Tug

REGISTRY = {g.id: g for g in (Platformer, Hop, Flap, Resurrection, Chomp, Bombers, Sweep, Trivia, Sketch, Tug, Hockey, Pong)}


def sizes(game, conf=None) -> list[tuple[int, int]]:
    s = (conf or {}).get(game.id, {}).get("size") or game.size
    return [tuple(s)] if isinstance(s[0], int) else [tuple(x) for x in s]


def pick_size(game, target_w: int, target_h: int, conf=None) -> tuple[int, int] | None:
    """The first declared (or configured) size that fits inside the target, or None."""
    return next(((w, h) for w, h in sizes(game, conf) if w <= target_w and h <= target_h), None)


def bezel(game, conf=None) -> dict:
    """Bezel settings: defaults < config [default.bezel] < the game's own < config [<id>.bezel]."""
    conf = conf or {}
    return bz.config(conf.get("default", {}).get("bezel"), getattr(game, "bezel", None),
                     conf.get(game.id, {}).get("bezel"))


def check_config(conf: dict):
    """Raise ValueError on an unknown game id, setting or bad bezel value."""
    for gid, sect in conf.items():
        if gid != "default" and gid not in REGISTRY:
            raise ValueError(f"config: unknown game [{gid}]; games are {sorted(REGISTRY)}")
        extra = set(sect) - ({"bezel"} if gid == "default" else {"bezel", "size"})
        if extra:
            raise ValueError(f"config [{gid}]: unknown setting(s) {sorted(extra)}")
        if "size" in sect:
            sz = sect["size"]
            ok = isinstance(sz, list) and sz and (all(isinstance(v, int) for v in sz) and len(sz) == 2 or all(
                isinstance(p, list) and len(p) == 2 and all(isinstance(v, int) for v in p) for p in sz))
            if not ok:
                raise ValueError(f"config [{gid}]: size must be [w, h] or [[w, h], ...]")
    for g in REGISTRY.values():
        bezel(g, conf)


def catalog(target=(768, 128), conf=None) -> list[dict]:
    out = []
    for g in REGISTRY.values():
        size = pick_size(g, *target, conf)
        out.append({"id": g.id, "title": g.title, "description": g.description, "max_players": g.max_players,
                    "controls": g.controls, "layout": getattr(g, "layout", "dpad"),
                    "sizes": [f"{w}x{h}" for w, h in sizes(g, conf)], "available": size is not None,
                    "size": f"{size[0]}x{size[1]}" if size else None})
    return out

"""Bezel: the graphic around and between a game's picture, so the screen output looks like the LED strips.

The bezel is generated, not drawn, so it fits any size: a vertical gradient between two colors with a repeating
pattern lightened on top. Settings (all optional; see DEFAULT):

- strip_height: cut the game canvas into strips of this many rows (the physical strip height) and put
  `gap * strip_height` rows of bezel between them. 0 = one piece.
- aspect: pad the screen frame with bezel to this width:height ratio ("16:9"), for letterboxing on a TV.
- ddp: also paint the bezel in the unused (letterbox) area of the DDP target instead of black. Strip gaps are
  never sent by DDP: the real strips already have them.
- colors: gradient top and bottom, "#rrggbb". pattern: diagonal, checker, dots or none. pattern_size: pixels
  per pattern cell. pattern_light: how much brighter (0-255) the pattern cells are.
"""

import json
from functools import lru_cache

import numpy as np

DEFAULT = {"strip_height": 0, "gap": 0.2, "aspect": None, "ddp": False,
           "colors": ["#202028", "#08080c"], "pattern": "diagonal", "pattern_size": 8, "pattern_light": 10}
PATTERNS = {"diagonal", "checker", "dots", "none"}


def config(*layers) -> dict:
    """Merge bezel settings, later layers win; raises ValueError on a bad value."""
    cfg = dict(DEFAULT)
    for layer in layers:
        cfg.update(layer or {})
    unknown = set(cfg) - set(DEFAULT)
    if unknown:
        raise ValueError(f"unknown bezel setting(s): {', '.join(sorted(unknown))}")
    if cfg["pattern"] not in PATTERNS:
        raise ValueError(f"bezel pattern must be one of {sorted(PATTERNS)}")
    if len(cfg["colors"]) != 2:
        raise ValueError("bezel colors must be two '#rrggbb' strings")
    [_rgb(c) for c in cfg["colors"]]
    _aspect(cfg["aspect"])
    return cfg


def _rgb(s: str):
    s = str(s).lstrip("#")
    if len(s) != 6:
        raise ValueError(f"bad color {s!r}, expected '#rrggbb'")
    return [int(s[i : i + 2], 16) for i in (0, 2, 4)]


def _aspect(a):
    if not a:
        return None
    w, _, h = str(a).partition(":")
    if not (w.isdigit() and h.isdigit() and int(w) and int(h)):
        raise ValueError(f"bad aspect {a!r}, expected e.g. '16:9'")
    return int(w) / int(h)


def fill(w: int, h: int, cfg: dict) -> np.ndarray:
    """A (h, w, 3) bezel image. Cached: the same size and settings give the same (read-only) array."""
    return _fill(w, h, json.dumps(cfg, sort_keys=True))


@lru_cache(maxsize=32)
def _fill(w, h, key):
    cfg = json.loads(key)
    top, bottom = (np.array(_rgb(c), float) for c in cfg["colors"])
    t = np.linspace(0, 1, h)[:, None, None] if h > 1 else np.zeros((1, 1, 1))
    img = np.broadcast_to(top + (bottom - top) * t, (h, w, 3)).copy()
    s = max(1, int(cfg["pattern_size"]))
    y, x = np.mgrid[0:h, 0:w]
    mask = {"diagonal": ((x + y) // s) % 2 == 0,
            "checker": (x // s + y // s) % 2 == 0,
            "dots": (x % s == s // 2) & (y % s == s // 2),
            "none": np.zeros((h, w), bool)}[cfg["pattern"]]
    img[mask] += cfg["pattern_light"]
    out = np.clip(img, 0, 255).astype(np.uint8)
    out.flags.writeable = False
    return out


def compose(frame: np.ndarray, cfg: dict) -> np.ndarray:
    """The game frame with strip gaps and aspect letterboxing in bezel. Unchanged when neither applies."""
    h, w = frame.shape[:2]
    sh = int(cfg["strip_height"])
    aspect = _aspect(cfg["aspect"])
    if not (0 < sh < h) and not aspect:
        return frame
    if not 0 < sh < h:
        sh = h
    gap = max(1, round(sh * cfg["gap"])) if sh < h else 0
    n = -(-h // sh)
    ch = h + gap * (n - 1)  # content height with the gaps
    ow, oh = w, ch
    if aspect:
        if ow / oh > aspect:
            oh = round(ow / aspect)
        else:
            ow = round(oh * aspect)
    out = fill(ow, oh, cfg).copy()
    x0, y0 = (ow - w) // 2, (oh - ch) // 2
    for i in range(n):
        strip = frame[i * sh : (i + 1) * sh]
        y = y0 + i * (sh + gap)
        out[y : y + len(strip), x0 : x0 + w] = strip
    return out


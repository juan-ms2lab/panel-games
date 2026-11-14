"""Pixel-art helpers: build sprites from text rows and draw them with transparency."""

import numpy as np


def make(rows: list[str], palette: dict[str, tuple]) -> np.ndarray:
    """Build an (h, w, 4) RGBA array. Characters missing from the palette are transparent."""
    if len({len(r) for r in rows}) != 1:
        raise ValueError(f"sprite rows differ in length: {[len(r) for r in rows]}")
    out = np.zeros((len(rows), len(rows[0]), 4), np.uint8)
    for y, row in enumerate(rows):
        for x, ch in enumerate(row):
            if ch in palette:
                out[y, x, :3] = palette[ch]
                out[y, x, 3] = 255
    return out


def blit(canvas: np.ndarray, spr: np.ndarray, x: int, y: int, flip_x: bool = False, flip_y: bool = False):
    """Draw spr onto canvas with its top-left at (x, y), clipped to the canvas."""
    if flip_x:
        spr = spr[:, ::-1]
    if flip_y:
        spr = spr[::-1]
    h, w = spr.shape[:2]
    ch, cw = canvas.shape[:2]
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, cw), min(y + h, ch)
    if x0 >= x1 or y0 >= y1:
        return
    part = spr[y0 - y : y1 - y, x0 - x : x1 - x]
    region = canvas[y0:y1, x0:x1]
    region[:] = np.where(part[:, :, 3:4] > 0, part[:, :, :3], region)

import numpy as np
import pytest

from panel_games.games import REGISTRY, pick_size, pad
from panel_games.games import elbirdo as fp

DT = 1 / 60


def bot(g):
    """Flap when below the next gap (sky only); good enough to pass some pipes."""
    nxt = next((p for p in g.pipes if p.x + g.pipe_w > g.bx), None)
    target = nxt.center(g.t, g.split) if nxt else g.split / 2
    return pad.A if g.vy > 0 and g.y + 2 * g.k > target else 0


@pytest.mark.parametrize("h", [32, 64, 128])
def test_elbirdo_plays_and_renders(h):
    g = fp.ElBirdo(768, h, seed=3)
    cv = np.zeros((h, 768, 3), np.uint8)
    seen, prev = set(), 0
    for i in range(60 * 40):
        seen.add(g.state)
        if g.state in ("intro", "ready", "over"):
            keys = pad.A if i % 20 == 0 else 0
        else:
            keys = bot(g)
        g.update(DT, [keys if keys != prev else 0])  # release between flaps so each is a new press
        prev = keys if keys != prev else 0
        if i % 15 == 0:
            g.render(cv)
            assert cv.any()
    assert {"intro", "ready", "play"} <= seen
    assert g.score > 0 or g.level > 0


def test_intro_and_underworld_render_at_64():
    g = fp.ElBirdo(768, 64, seed=1)
    cv = np.zeros((64, 768, 3), np.uint8)
    for _ in range(int(6 * 60)):  # through the melt and the rise of RESURRECTION
        g.update(DT, [0])
    g.render(cv)
    assert cv[32:].any() and g.split == 32
    g.update(DT, [pad.A])  # intro -> ready
    g.update(DT, [0])
    g.update(DT, [pad.A])  # ready -> play
    assert g.state == "play"
    g.render(cv)
    under = cv[32:62].reshape(-1, 3)
    assert (under[:, 0] > under[:, 2]).mean() > 0.9  # red underworld with fire columns
    utop, ubot = g._hole()
    assert 32 < utop < ubot < 64
    g.y = 45.0  # drop into the underworld: skeleton birdo
    g.update(DT, [0])
    assert g.was_under


def test_single_display_has_no_underworld():
    g = fp.ElBirdo(768, 32)
    assert g.split == 32 and g.k == 1


def test_pause_resume_and_stage_select():
    g = fp.ElBirdo(768, 64)
    fp.ElBirdo.record["unlocked"] = 3
    g.update(DT, [pad.RIGHT])
    assert g.state == "ready" and g.stage == 1
    g.update(DT, [0])
    g.update(DT, [pad.A])
    assert g.state == "play"
    g.update(DT, [pad.START])
    assert g.state == "paused"
    y = g.y
    g.update(DT, [0])
    assert g.y == y
    g.update(DT, [pad.START])
    assert g.state == "play"
    g.update(DT, [None])  # player left
    assert g.state == "paused"
    fp.ElBirdo.record.update(best=0, unlocked=1)


def test_falling_kills_and_loses_a_life():
    g = fp.ElBirdo(768, 32)
    g.update(DT, [pad.A])
    g.update(DT, [0])
    g.update(DT, [pad.A])
    for _ in range(60 * 4):
        g.update(DT, [0])
    assert g.lives == fp.START_LIVES - 1


def test_declared_sizes():
    assert REGISTRY["elbirdo"] is fp.ElBirdo and REGISTRY["flap"].title == "Panel Flap"
    assert pick_size(fp.ElBirdo, 768, 64) == (768, 64)
    assert pick_size(fp.ElBirdo, 768, 32) == (768, 32)
    assert pick_size(fp.ElBirdo, 1536, 256) == (768, 128)
    assert pick_size(REGISTRY["platformer"], 768, 64) is None

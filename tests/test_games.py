import numpy as np
import pytest

from panel_games.games import REGISTRY, catalog, pad
from panel_games.games import platformer as pf
from panel_games.games.pong import Pong
from panel_games.games.sprites import blit, make

DT = 1 / 60


def bot_keys(g, h):
    """A simple player: run right, jump walls, bottomless pits and nearby enemies, hold jump while rising."""
    keys = pad.RIGHT | pad.B
    feet = int((h.y + h.h - 1) // pf.T)
    front = int((h.x + h.w + 3) // pf.T)
    wall = g._solid(front, feet) or g._solid(front, feet - 1)
    pit = h.on_ground and not any(g._solid(front, r) for r in range(feet + 1, pf.LH))
    foe = any(8 < e.x - h.x < 34 and abs(e.y - h.y) < 14 and e.squish <= 0 for e in g.enemies)
    if h.on_ground and (wall or pit or foe):
        keys |= pad.A
    if not h.on_ground and h.vy < 0:
        keys |= pad.A
    return keys


def play(g, seconds, keys_for=bot_keys, slots=(0,)):
    for _ in range(int(seconds / DT)):
        pads = [None] * g.max_players
        for s in slots:
            h = g.heroes.get(s)
            pads[s] = keys_for(g, h) if h and h.alive else 0
        g.update(DT, pads)
        if g.state != "play":
            break


def test_level_shape():
    lvl = pf.build_level()
    assert len(lvl) == pf.LH and all(len(r) == pf.LW for r in lvl)
    flat = "".join("".join(r) for r in lvl)
    assert flat.count("S") == 1 and flat.count("F") > 5 and flat.count("e") >= 5 and flat.count("?") >= 4
    assert all(lvl[y][x] == " " for y in range(0, 2) for x in range(pf.LW))  # HUD rows stay clear


def test_hero_stands_and_jumps():
    g = pf.Platformer(768, 128)
    g.update(DT, [0, None, None, None])
    for _ in range(30):
        g.update(DT, [0, None, None, None])
    h = g.heroes[0]
    assert h.on_ground and abs(h.y - (14 * pf.T - h.h)) < 0.5
    g.update(DT, [pad.A, None, None, None])
    assert h.vy < 0 and not h.on_ground
    top = h.y
    for _ in range(40):
        g.update(DT, [pad.A, None, None, None])
        top = min(top, h.y)
    assert 14 * pf.T - h.h - top > 24  # clears a 3-tile obstacle


def test_cannot_walk_through_pipe():
    g = pf.Platformer(768, 128)
    g.update(DT, [0, None, None, None])
    for _ in range(60 * 8):
        g.update(DT, [pad.RIGHT, None, None, None])  # walking, never jumping
    assert g.heroes[0].x < 18 * pf.T  # stopped by the first pipe


def test_bot_completes_the_level():
    g = pf.Platformer(768, 128)
    play(g, 120)
    assert g.state == "won" and g.winner == 0


def test_coin_box_pays_once():
    g = pf.Platformer(768, 128)
    g.update(DT, [0, None, None, None])
    h = g.heroes[0]
    h.x, h.y, h.vy = 11 * pf.T + 1, 11 * pf.T + 2, -200.0  # just under the first coin box
    score = h.score
    for _ in range(10):
        g.update(DT, [pad.A, None, None, None])
    assert g.grid[10][11] == "U" and h.score == score + 10


def test_stomping_defeats_enemy_and_side_hit_kills():
    g = pf.Platformer(768, 128)
    g.update(DT, [0, None, None, None])
    h, e = g.heroes[0], g.enemies[0]
    h.invuln = 0
    h.x, h.y, h.vy = e.x, e.y - h.h - 1, 120.0
    g.update(DT, [0, None, None, None])
    assert e.squish > 0 and h.vy < 0 and h.alive
    e2 = g.enemies[1]
    h.x, h.y, h.vy, h.invuln = e2.x - 3, e2.y - 4, 0.0, 0.0
    g.update(DT, [0, None, None, None])
    assert not h.alive and h.lives == 2


def test_everyone_out_of_lives_ends_round_and_reset_restores():
    g = pf.Platformer(768, 128)
    g.update(DT, [0, None, None, None])
    h = g.heroes[0]
    h.lives = 1
    h.y = 200
    g.update(DT, [0, None, None, None])
    assert g.state == "lost"
    for _ in range(int(pf.RESET_DELAY / DT) + 2):
        g.update(DT, [0, None, None, None])
    assert g.state == "play" and g.heroes[0].lives == 3


def test_players_join_and_leave():
    g = pf.Platformer(768, 128)
    g.update(DT, [0, 0, None, None])
    assert set(g.heroes) == {0, 1}
    g.update(DT, [0, None, None, None])
    assert set(g.heroes) == {0}


def test_render_draws_and_hud():
    g = pf.Platformer(768, 128)
    g.update(DT, [0, 0, None, None])
    canvas = np.zeros((128, 768, 3), np.uint8)
    g.render(canvas)
    assert canvas[100, 24:32].any() and tuple(canvas[60, 400]) == pf.SKY
    assert canvas[0:12, 0:200].any()  # HUD text


def test_render_every_frame_while_playing():
    g = pf.Platformer(768, 128)
    g.update(DT, [0, 0, None, None])
    canvas = np.zeros((128, 768, 3), np.uint8)
    seen = set()
    for _ in range(int(14 / DT)):
        pads = [bot_keys(g, g.heroes[0]) if g.heroes[0].alive else 0, pad.LEFT | pad.A, None, None]
        g.update(DT, pads)
        g.render(canvas)  # covers stand, run, jump, death and squash drawing
        seen.add(g.heroes[0].on_ground)
        if g.state != "play":
            break
    assert seen == {True, False}


def test_all_game_sprites_are_rectangular():
    for rows in (pf._HERO_STAND, pf._HERO_RUN, pf._HERO_JUMP, *pf._ENEMY, pf._COIN, pf._GROUND_TOP, pf._DIRT,
                 pf._BRICK, pf._BOX, pf._USED):
        assert len({len(r) for r in rows}) == 1
    with pytest.raises(ValueError):
        make(["ab", "abc"], {})


def test_small_display_is_rejected():
    with pytest.raises(ValueError):
        pf.Platformer(64, 32)


def test_pong_scores_and_ai():
    g = Pong(768, 128, seed=1)
    for _ in range(60 * 30):
        g.update(DT, [None, None])
        if g.state != "play":
            break
    assert sum(g.scores) > 0  # computer vs computer still rallies and scores
    canvas = np.zeros((128, 768, 3), np.uint8)
    g.render(canvas)
    assert canvas.any()


def test_sprite_blit_clips_and_makes_transparency():
    spr = make(["ab", "b."], {"a": (1, 2, 3), "b": (4, 5, 6)})
    canvas = np.zeros((3, 3, 3), np.uint8)
    blit(canvas, spr, 2, 2)  # mostly off canvas
    assert canvas[2, 2].tolist() == [1, 2, 3]
    blit(canvas, spr, -5, -5)  # fully off canvas
    assert catalog() and {"platformer", "pong"} <= set(REGISTRY)


def test_pad_slots_tokens_and_stale_keys(monkeypatch):
    p = pad.Pad()
    with pytest.raises(ValueError):
        p.join()
    p.start("pong", 2)
    s0, t0 = p.join()
    s1, t1 = p.join()
    assert (s0, s1) == (0, 1)
    with pytest.raises(ValueError):
        p.join()  # full
    p.set_keys(t1, pad.UP | pad.A)
    assert p.snapshot() == [0, pad.UP | pad.A]
    with pytest.raises(ValueError):
        p.set_keys("nope", 1)
    with pytest.raises(ValueError):
        p.set_keys(t0, 999)
    p.leave(t1)
    assert p.snapshot() == [0, None]
    assert p.join()[0] == 1  # freed slot is reused
    now = pad.time.monotonic()
    monkeypatch.setattr(pad.time, "monotonic", lambda: now + pad.KEY_TIMEOUT + 0.1)
    assert p.snapshot()[0] == 0  # held keys go stale
    monkeypatch.setattr(pad.time, "monotonic", lambda: now + pad.SLOT_TIMEOUT + 1)
    assert p.snapshot() == [None, None]  # silent players are dropped

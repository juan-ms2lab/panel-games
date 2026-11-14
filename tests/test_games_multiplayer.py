import numpy as np
import pytest

from panel_games.games import REGISTRY, chomp as ch, flap as fl, pad
from panel_games.games.base import Lobby

DT = 1 / 60
NONE4 = [None, None, None, None]


def pads_for(**keys):
    out = [None] * max(4, 1 + max((int(s[1:]) for s in keys), default=0))
    for slot, k in keys.items():
        out[int(slot[1:])] = k
    return out


def canvas():
    return np.zeros((128, 768, 3), np.uint8)


def begin(g, slots, host=None):
    """Join the given slots, then have the host press START."""
    host = min(slots) if host is None else host
    joined = pads_for(**{f"s{s}": 0 for s in slots})
    for _ in range(3):
        g.update(DT, joined)
    g.update(DT, pads_for(**{f"s{s}": (pad.START if s == host else 0) for s in slots}))
    g.update(DT, joined)
    return g


# ---- Panel Flap

def flap_bot(g, b):
    """Flap whenever falling below the next gap's centre."""
    nxt = next((p for p in g.pipes if p[0] + fl.PIPE_W > b.x - 4), None)
    target = nxt[1] if nxt else fl.SH / 2
    return pad.A if b.vy > 0 and b.y > target + 4 else 0


def start_flap(slots=(0,), seed=3):
    g = begin(fl.Flap(768, 128, seed=seed), slots)
    assert g.state == "play"
    return g


def test_lobby_waits_for_the_host_to_press_start():
    g = fl.Flap(768, 128, seed=1)
    for _ in range(60):
        g.update(DT, NONE4)
    assert g.state == "lobby" and g.lobby.host is None
    both = pads_for(s0=0, s1=0)
    for _ in range(int(30 / DT)):  # no countdown: waiting forever is fine
        g.update(DT, both)
    assert g.state == "lobby" and g.lobby.host == 0
    g.update(DT, pads_for(s0=0, s1=pad.START))  # a guest pressing START does nothing
    g.update(DT, both)
    assert g.state == "lobby"
    g.update(DT, pads_for(s0=pad.START, s1=0))
    assert g.state == "play" and set(g.birds) == {0, 1}


def test_lobby_host_passes_to_next_player_when_host_leaves():
    lobby = Lobby()
    lobby.update({1, 2}, [None, 0, 0])
    assert lobby.host == 1
    lobby.update({0, 1, 2}, [0, 0, 0])
    assert lobby.host == 1  # first to join stays host, even though slot 0 is lower
    lobby.update({0, 2}, [0, None, 0])
    assert lobby.host == 2  # next in line is whoever joined next, not the lowest slot number
    assert lobby.update({0, 2}, [0, None, pad.START]) is True
    assert lobby.update({0, 2}, [0, None, pad.START]) is False  # held START fires once


def test_flap_idle_bird_falls_and_dies():
    g = start_flap()
    for _ in range(int(3 / DT)):
        g.update(DT, pads_for(s0=0))
        if g.state != "play":
            break
    assert g.state == "over" and g.result["solo"] and g.result["score"] == 0


def test_flap_bot_travels_across_multiple_screens():
    g = start_flap()
    for _ in range(int(60 / DT)):
        b = g.birds[0]
        g.update(DT, pads_for(s0=flap_bot(g, b) if b.alive else 0))
        if g.state == "over" or g.page >= 2:
            break
    assert g.page >= 2, f"bot died on screen {g.page + 1} (state {g.state})"
    assert g.birds[0].score >= 2 * fl.PER_PAGE


def test_flap_scroll_pauses_physics_and_rebases_birds():
    g = start_flap()
    g.state, g.cam0 = "scroll", g.cam
    b = g.birds[0]
    b.x0, b.x, b.y = b.x, g.cam + fl.TRIGGER, 60.0
    g.scroll_t = 0.0
    y_before = b.y
    for _ in range(int(fl.SCROLL_TIME / DT) // 2):
        g.update(DT, pads_for(s0=0))
    assert g.state == "scroll" and abs(b.y - y_before) < 1e-6  # no gravity while the screen scrolls
    for _ in range(int(fl.SCROLL_TIME / DT)):
        g.update(DT, pads_for(s0=0))
        if g.state != "scroll":
            break
    assert g.state == "play" and g.page == 1
    assert b.x - g.cam == pytest.approx(b.sx, abs=0.01)  # back at the left edge of the new screen


def test_flap_last_bird_standing_wins():
    g = start_flap(slots=(0, 1))
    for _ in range(int(20 / DT)):
        b1 = g.birds[1]
        g.update(DT, pads_for(s0=0, s1=flap_bot(g, b1) if b1.alive else 0))  # bird 0 never flaps
        if g.state == "over":
            break
    assert g.state == "over" and g.result["winner"] == 1 and not g.result["solo"]
    for _ in range(int(fl.RESULT_TIME / DT) + 2):
        g.update(DT, pads_for(s0=0, s1=0))
    assert g.state == "lobby"


def test_flap_leaving_player_counts_as_dead():
    g = start_flap(slots=(0, 1))
    g.update(DT, pads_for(s0=0))  # slot 1 vanished
    assert not g.birds[1].alive
    assert g.state == "over" and g.result["winner"] == 0


def test_flap_renders_every_state():
    g = fl.Flap(768, 128, seed=2)
    c = canvas()
    g.update(DT, pads_for(s0=0))
    g.render(c)
    begin(g, (0,))
    for _ in range(int(40 / DT)):
        b = g.birds.get(0)
        g.update(DT, pads_for(s0=flap_bot(g, b) if b and b.alive else 0))
        g.render(c)  # lobby, play, scroll and over
    assert c.any()


# ---- Panel Chomp

def reachable(g):
    start = ch.HOME
    seen, stack = {start}, [start]
    while stack:
        x, y = stack.pop()
        for dx, dy in ch.DIRS:
            n = (x + dx, y + dy)
            if n not in seen and g._open(*n):
                seen.add(n)
                stack.append(n)
    return seen


def start_chomp(slots=(0,), seed=5):
    g = begin(ch.Chomp(768, 128, seed=seed), slots)
    assert g.state == "play"
    return g


def test_chomp_maze_is_fully_connected_and_loops():
    for seed in range(5):
        g = ch.Chomp(768, 128, seed=seed)
        open_tiles = {(x, y) for y in range(ch.GH) for x in range(ch.GW) if not g.grid[y][x]}
        assert reachable(g) == open_tiles
        assert all(not g.grid[y][x] for x, y in ch.STARTS + ch.PELLETS + [ch.HOME])
        edges = sum(1 for (x, y) in open_tiles for d in ((1, 0), (0, 1)) if (x + d[0], y + d[1]) in open_tiles)
        assert edges > len(open_tiles) - 1  # extra openings make loops, not just a tree
        assert 400 < len(g.dots) < 900


def test_chomp_movement_stays_in_corridors_and_eats():
    g = start_chomp()
    c = g.chompers[0]
    rng = np.random.default_rng(0)
    for i in range(60 * 40):
        keys = [pad.LEFT, pad.RIGHT, pad.UP, pad.DOWN][int(rng.integers(0, 4))] if i % 20 == 0 else None
        g.update(DT, pads_for(s0=keys if keys is not None else 0))
        x, y = c.pos()
        assert g._open(int(round(x)), int(round(y)))
        if g.state != "play":
            break
    assert c.score >= 0


def test_chomp_player_eats_dots_and_pellet_scares_wisps():
    g = start_chomp()
    c = g.chompers[0]
    c.tx, c.ty = 2, 1  # next to the corner pellet at (1, 1)
    g.grid[1][1] = g.grid[1][2] = 0
    g.update(DT, pads_for(s0=pad.LEFT))
    for _ in range(60):
        g.update(DT, pads_for(s0=pad.LEFT))
    assert (1, 1) not in g.pellets
    assert any(w.state == "scared" for w in g.wisps) or g.play_t < min(w.release for w in g.wisps)
    for w in g.wisps:
        w.state = "scared"
    g.scare = ch.SCARE_TIME
    w = g.wisps[0]
    w.tx, w.ty, w.dx, w.dy, w.p = c.tx, c.ty, 0, 0, 0.0
    score = c.score
    g._collide(w)
    assert w.state == "eyes" and c.score == score + 200 and c.alive


def test_chomp_wisp_catches_player():
    g = start_chomp()
    c, w = g.chompers[0], ch.Wisp(0)
    w.state, w.tx, w.ty = "chase", c.tx, c.ty
    g._collide(w)
    assert not c.alive


def test_chomp_all_caught_last_standing_wins():
    g = start_chomp(slots=(0, 1))
    g.chompers[0].died_at = 0
    g._die(g.chompers[0])
    g.t += 5
    g._die(g.chompers[1])
    g._check_end()
    assert g.state == "over" and g.result["mode"] == "last"
    assert [s for s, _ in g.result["ranking"]] == [1, 0]  # died later = better


def test_chomp_clear_goes_to_podium_by_score_with_survivors_first():
    g = start_chomp(slots=(0, 1, 2, 3))
    for slot, score in ((0, 300), (1, 900), (2, 500), (3, 100)):
        g.chompers[slot].score = score
    g.dots.clear()
    g.pellets.clear()
    g._check_end()
    assert g.result["mode"] == "podium"
    assert g.result["ranking"] == [(1, 900), (2, 500), (0, 300), (3, 100)]
    # A player who was caught ranks below every survivor even with a higher score.
    g2 = start_chomp(slots=(0, 1, 2))
    for slot, score in ((0, 100), (1, 200), (2, 5000)):
        g2.chompers[slot].score = score
    g2._die(g2.chompers[2])
    g2.dots.clear()
    g2.pellets.clear()
    g2._check_end()
    assert [s for s, _ in g2.result["ranking"]] == [1, 0, 2]


def test_chomp_solo_game_over_and_new_round():
    g = start_chomp()
    g._die(g.chompers[0])
    g._check_end()
    assert g.result["solo"] and g.result["mode"] == "last"
    for _ in range(int(ch.RESULT_TIME_LAST / DT) + 2):
        g.update(DT, pads_for(s0=0))
    assert g.state == "lobby" and len(g.dots) > 400


def test_chomp_leaving_player_is_out():
    g = start_chomp(slots=(0, 1))
    g.update(DT, pads_for(s0=0))
    assert not g.chompers[1].alive and g.chompers[0].alive


def test_chomp_renders_every_state():
    g = ch.Chomp(768, 128, seed=7)
    c = canvas()
    g.render(c)  # empty lobby
    g.update(DT, pads_for(s0=0, s1=0))
    g.render(c)  # lobby with joiners
    begin(g, (0, 1))
    for i in range(int(20 / DT)):
        g.update(DT, pads_for(s0=pad.RIGHT if i % 90 < 45 else pad.DOWN, s1=pad.LEFT))
        g.render(c)
        if g.state == "over":
            break
    g.state, g.result = "over", {"mode": "podium", "solo": False, "ranking": [(2, 900), (0, 500), (3, 100)]}
    g.render(c)  # podium
    assert c.any()
    g.result = {"mode": "last", "solo": False, "ranking": [(1, 10), (0, 5)]}
    g.render(c)


def test_chomp_rejects_small_display():
    with pytest.raises(ValueError):
        ch.Chomp(64, 32)


def test_registry_has_all_games():
    assert {"platformer", "flap", "chomp", "pong", "bomber", "sweep", "trivia"} <= set(REGISTRY)

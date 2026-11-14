import random

import numpy as np
import pytest

from panel_games.games import hop, pad
from test_games_multiplayer import DT, begin, canvas, pads_for

STEP = 10 * DT  # one hop and its recovery take exactly ten frames
ACTIONS = [(1, 0), (0, -1), (0, 1), (0, 0), (-1, 0)]
KEY = {(1, 0): pad.RIGHT, (-1, 0): pad.LEFT, (0, -1): pad.UP, (0, 1): pad.DOWN, (0, 0): 0}


def start_hop(slots=(0,), seed=1):
    g = begin(hop.Hop(768, 128, seed=seed), slots)
    assert g.state == "play"
    for h in g.hoppers.values():
        h.invuln = 0.0
    return g


def tick(g, frames=1, **keys):
    for _ in range(frames):
        g.update(DT, pads_for(**keys) if keys else pads_for(s0=0))


# ---- level design

def test_level_layout_and_difficulty_curve():
    lanes = hop.build_level(random.Random(1))
    assert len(lanes) == hop.GW
    assert [l.kind for l in lanes[:3]] == ["safe"] * 3
    assert all(l.kind == "pond" for l in lanes[hop.POND_START:]) and lanes[hop.POND_START - 1].kind == "safe"
    assert sum(1 for l in lanes if l.biome == "city" and l.kind == "road") == 24  # a huge multilane highway

    def avg(biome, attr):
        ls = [l for l in lanes if l.biome == biome and l.kind in ("road", "water")]
        if attr == "cover":
            return sum(sum(o[1] for o in l.objs) / hop.P for l in ls) / len(ls)
        return sum(l.speed for l in ls) / len(ls)

    roads = [avg(b, "speed") for b in ("country", "suburb", "city")]
    assert roads == sorted(roads) and roads[-1] > 1.8 * roads[0]  # roads get faster
    cover = [avg(b, "cover") for b in ("country", "suburb", "city")]
    assert cover == sorted(cover)  # and busier
    rivers = [avg(b, "speed") for b in ("stream", "lava", "canal")]
    assert rivers == sorted(rivers)
    rcov = [avg(b, "cover") for b in ("stream", "lava", "canal")]
    assert rcov[0] > rcov[1] and rcov[0] > rcov[2]  # and offer fewer platforms than the gentle stream
    assert [i for i, l in enumerate(lanes) if l.kind == "rail"]  # the desert has railway lines


def test_lane_objects_loop_and_never_overlap():
    for seed in range(5):
        for lane in hop.build_level(random.Random(seed)):
            iv = sorted((o, ln) for o, ln, _ in lane.objs)
            for i, (o, ln) in enumerate(iv):
                nxt_o = iv[(i + 1) % len(iv)][0] + (hop.P if i + 1 == len(iv) else 0)
                assert o + ln <= nxt_o + 1e-9, (lane.biome, iv)
    lane = hop.Lane("road", "city", 1, 5.0, [(3.0, 2, "car")])
    y_a = [a for a, b, *_ in lane.intervals(0.0)]
    y_b = [a for a, b, *_ in lane.intervals(hop.P / 5.0)]  # one full loop later
    assert y_a == pytest.approx(y_b)


def test_vehicles_flash_a_warning_before_entering_the_screen():
    lane = hop.Lane("road", "city", 1, 4.0, [(float(hop.MARGIN) - 4.0, 2, "car")])  # fully above the screen
    assert lane.entering(0.0) == "top"
    assert not lane.hits(0.0, 1.0, 0.0)  # nothing on screen yet
    assert lane.hits(0.0, 1.0, 0.0, offscreen=True) is False
    down = hop.Lane("road", "city", -1, 4.0, [(float(hop.GH + hop.MARGIN) + 0.5, 2, "car")])
    assert down.entering(0.0) == "bottom"
    assert hop.Lane("road", "city", 1, 4.0, [(float(hop.MARGIN) + 5, 2, "car")]).entering(0.0) is None


def test_turtles_dive_and_platform_disappears():
    lane = hop.Lane("water", "stream", 1, 0.0, [(10.0, 3, "turtle")])
    yc = 10.0 - hop.MARGIN + 1.0
    assert lane.platform(yc, 0.0) is not None
    assert hop.turtle_state(10.0, 2.6) == "down"  # diving is staggered by the turtles' belt offset
    assert lane.platform(yc, 2.6) is None


# ---- movement and rules

def test_hopping_moves_one_lane_and_respects_bounds():
    g = start_hop()
    h = g.hoppers[0]
    tick(g, 20, s0=pad.RIGHT)
    assert h.x >= 1.0 and h.max_col >= 1
    h.x, h.y, h.hop = 0.0, 5.0, None
    tick(g, 20, s0=pad.LEFT)
    assert h.x == 0.0  # cannot leave the left edge
    h.y = 0.0
    tick(g, 20, s0=pad.UP)
    assert h.y == 0.0


def test_hold_repeats_hops_and_a_tap_makes_one():
    g = start_hop()
    g.lanes = [hop.Lane("safe", "median") for _ in range(hop.GW)]
    h = g.hoppers[0]
    tick(g, int(1.0 / DT), s0=pad.RIGHT)
    assert h.max_col >= 5
    g2 = start_hop(seed=2)
    g2.lanes = [hop.Lane("safe", "median") for _ in range(hop.GW)]
    h2 = g2.hoppers[0]
    tick(g2, 2, s0=pad.RIGHT)
    tick(g2, 40, s0=0)
    assert h2.max_col == 1


def test_traffic_kills_costs_a_life_and_respawns_at_checkpoint():
    g = start_hop()
    h = g.hoppers[0]
    h.cp = 9
    lane = g.lanes[5]
    lane.objs, lane.speed, lane.dir = [(float(hop.MARGIN + 5), 2, "car")], 0.0, 1  # parked across rows 5-7
    h.x, h.y, h.hop = 5.0, 5.5, None
    tick(g, 2)
    assert not h.alive and h.lives == hop.LIVES - 1 and h.cause == "squash"
    tick(g, int(1.5 / DT))
    assert h.alive and h.x == 9.0 and h.invuln > 0  # back at the last safe lane it reached


def test_water_needs_a_platform_and_carries_you():
    g = start_hop()
    h = g.hoppers[0]
    stream = 10
    lane = g.lanes[stream]
    lane.objs, lane.speed, lane.dir = [(8.0, 4, "log")], 1.5, 1
    g.play_t = 0.0
    log_y = 8.0 - hop.MARGIN
    h.x, h.y, h.hop = float(stream), log_y + 1.0, None
    y0 = h.y
    tick(g, 30)
    assert h.alive and h.y > y0 + 0.5  # carried down with the log
    h.x, h.y, h.hop = float(stream), log_y + 20.0, None  # nothing there
    tick(g, 2)
    assert not h.alive and h.cause in ("drown", "swept")
    h2 = g.hoppers[0]
    h2.lives, h2.alive = 3, True
    h2.x, h2.y, h2.hop = 9.0, log_y + 1.0, None
    lane.speed = 0.0
    tick(g, 20, s0=pad.RIGHT)  # hop from the bank onto the log
    assert h2.alive and int(round(h2.x)) >= stream


def test_landing_in_open_water_drowns():
    g = start_hop()
    h = g.hoppers[0]
    g.lanes[10].objs = []
    h.x, h.y, h.hop = 9.0, 7.0, None
    tick(g, 20, s0=pad.RIGHT)
    assert not h.alive and h.cause == "drown"


def test_train_warning_then_train_kills():
    g = start_hop()
    h = g.hoppers[0]
    rail = next(i for i, l in enumerate(g.lanes) if l.kind == "rail")
    g.lanes[rail].trains = [5.0]
    g.play_t = 3.0
    h.x, h.y, h.hop = float(rail), 7.0, None
    assert g.lanes[rail].warning(3.0) is False
    g.play_t = 4.0
    assert g.lanes[rail].warning(g.play_t)
    tick(g, 20)
    assert h.alive  # warned, but nothing yet
    for _ in range(int(2.5 / DT)):
        tick(g, 1)
        if not h.alive:
            break
    assert not h.alive and h.cause == "squash"


def test_finishing_ranks_players_and_starts_last_call():
    g = start_hop(slots=(0, 1))
    a, b = g.hoppers[0], g.hoppers[1]
    for h in (a, b):
        h.x, h.y, h.hop, h.max_col = float(hop.POND_START - 1), float(2 + 3 * h.slot), None, hop.POND_START - 1
    tick_pads = lambda **k: g.update(DT, pads_for(**k))
    for _ in range(20):
        tick_pads(s0=pad.RIGHT, s1=0)
    assert a.finished and a.place == 1 and not b.finished
    assert g.deadline <= g.play_t + hop.LAST_CALL + 1e-6
    for _ in range(20):
        tick_pads(s0=0, s1=pad.RIGHT)
    assert b.place == 2 and g.state == "over"
    assert g.result["finished"] == [0, 1] and g.result["ranking"][0][0] == 0
    assert g.result["ranking"][0][1] > g.result["ranking"][1][1] >= hop.POND_START * 10


def test_everyone_out_of_lives_ends_the_round_and_furthest_ranks_first():
    g = start_hop(slots=(0, 1))
    a, b = g.hoppers[0], g.hoppers[1]
    a.max_col, b.max_col = 20, 45
    for h in (a, b):
        h.lives = 1
        g._die(h, "squash")
    tick_pads = lambda: g.update(DT, pads_for(s0=0, s1=0))
    tick_pads()
    assert g.state == "over" and [s for s, _ in g.result["ranking"]] == [1, 0]
    for _ in range(int((hop.RESULT_TIME + 1) / DT)):
        tick_pads()
    assert g.state == "lobby"


def test_time_limit_ends_the_round():
    g = start_hop()
    g.play_t = hop.TIME_LIMIT
    tick(g, 1)
    assert g.state == "over"


def test_hop_renders_every_state():
    g = hop.Hop(768, 128, seed=3)
    c = canvas()
    g.render(c)
    begin(g, (0, 1, 2))
    for i in range(240):
        g.update(DT, [pad.RIGHT if i % 20 < 10 else pad.DOWN, pad.UP, 0, None])
        g.render(c)
    g.play_t = 100.0
    g.render(c)
    g.state, g.result = "over", {"solo": False, "finished": [1], "ranking": [(1, 3000), (0, 500), (2, 100)]}
    g.render(c)
    g.result["solo"] = True
    g.render(c)
    assert c.any()
    with pytest.raises(ValueError):
        hop.Hop(64, 32)


# ---- the level can be crossed

def plan(g, col, y, t0, depth):
    """Search `depth` hops ahead. Prefers (1) surviving the horizon, (2) ending on safe ground, (3) progress."""
    memo = {}

    def best(col, y, k):
        key = (col, round(y * 4), k)
        if key in memo:
            return memo[key]
        if k == depth:
            res = (k, g.lanes[col].kind in ("safe", "pond"), col, None)
        else:
            res = None
            for a in ACTIONS:
                nxt = step(g, col, y, t0 + k * STEP, a)
                if not nxt:
                    continue
                sub = best(nxt[0], nxt[1], k + 1)
                cand = (sub[0], sub[1], sub[2] + (0.01 if k == 0 and a == (1, 0) else 0.0), a)
                if res is None or cand[:3] > res[:3]:
                    res = cand
            if res is None:
                res = (k, False, col, None)  # cornered: survived k hops
        memo[key] = res
        return res

    return best(col, y, 0)


def bot_action(g, h, depth=14):
    return plan(g, int(round(h.x)), h.y, g.play_t, depth)[3] or (0, 0)


def ok_at(g, col, y, t, margin=0.15):
    """(safe, carry velocity) with a margin, one frame ahead, because the game steps time before testing hits."""
    lane = g.lanes[col]
    t += DT
    if lane.kind in ("safe", "pond"):
        return True, 0.0
    if lane.kind == "water":
        vs = [lane.platform(y + f, t) for f in (0.2, 0.5, 0.8)]
        return all(v is not None for v in vs), vs[1] or 0.0
    return not lane.hits(y - margin, y + 1 + margin, t, offscreen=True), 0.0


def step(g, col, y, t, a):
    dx, dy = a
    t2 = t + STEP
    if (dx, dy) == (0, 0):
        ok, v = ok_at(g, col, y, t)
        y2 = y + v * STEP
        ok2, _ = ok_at(g, col, y2, t2)
        return (col, y2, t2) if ok and ok2 and -0.3 <= y2 <= hop.GH - 0.7 else None
    col2, y2 = col + dx, y + dy
    if not (0 <= col2 < hop.GW and 0 <= y2 <= hop.GH - 1):
        return None
    for frac in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):  # the body overlaps both lanes while hopping
        x, ym, tm = col + dx * frac, y + dy * frac, t + hop.HOP_T * frac
        for c in range(int((x + 0.15) // 1), int((x + 0.85) // 1) + 1):
            if 0 <= c < hop.GW and g.lanes[c].kind in ("road", "rail") and not ok_at(g, c, ym, tm)[0]:
                return None
    ok, _ = ok_at(g, col2, y2, t + hop.HOP_T)
    if not ok:
        return None
    if g.lanes[col2].kind != "water":
        y2 = float(round(y2))
    return col2, y2, t2


import os

SEEDS = [1, 2, 3, 4, 5] if os.environ.get("LED_SLOW_TESTS") else [5]  # each seed takes 15-25 s


@pytest.mark.parametrize("seed", SEEDS)
def test_a_lookahead_bot_can_cross_every_biome(seed):
    g = start_hop(seed=seed)
    h = g.hoppers[0]
    for _ in range(int(hop.TIME_LIMIT / DT)):
        if g.state != "play":
            break
        if h.hop is None and h.alive and h.cool <= DT + 1e-9:
            g.update(DT, pads_for(s0=KEY[bot_action(g, h)]))
        else:
            g.update(DT, pads_for(s0=0))
    assert h.finished, f"seed {seed}: reached column {h.max_col} with {h.lives} lives left"

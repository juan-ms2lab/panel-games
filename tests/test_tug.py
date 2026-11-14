import numpy as np

from panel_games.games import REGISTRY, pad, tug
from test_games_multiplayer import DT, begin, canvas, pads_for


def new(slots=(0, 1, 2, 3), seed=1):
    g = tug.Tug(768, 128, seed=seed)
    begin(g, slots)
    assert g.state == "ready"
    return g


def advance(g, seconds, slots=(0, 1, 2, 3)):
    for _ in range(int(seconds / DT)):
        g.update(DT, pads_for(**{f"s{s}": 0 for s in slots}))


def to_pull(g, slots=(0, 1, 2, 3)):
    advance(g, tug.READY_TIME + 0.1, slots)
    assert g.state == "pull"


def tap_team(g, team, rate=6, seconds=1.0, slots=(0, 1, 2, 3)):
    """Every member of the team taps `rate` times a second."""
    members = g.members(team)
    for frame in range(int(seconds / DT)):
        if frame % int(round(60 / rate)) == 0:
            for s in members:
                g.on_message(s, {"t": "tap", "n": 1})
        g.update(DT, pads_for(**{f"s{s}": 0 for s in slots}))


def test_teams_are_balanced_and_cover_everyone():
    for slots in ((0, 1), (0, 1, 2), (0, 1, 2, 3, 4, 5, 6, 7)):
        g = new(slots)
        assert sorted(g.team) == sorted(slots)
        assert abs(len(g.members(0)) - len(g.members(1))) <= 1


def test_start_needs_two_players():
    g = tug.Tug(768, 128, seed=1)
    begin(g, (0,))
    assert g.state == "lobby" and g.need_flash > 0


def test_taps_only_count_during_the_pull_and_for_team_members():
    g = new()
    g.on_message(0, {"t": "tap", "n": 3})
    assert g.round_taps[0] == 0
    to_pull(g)
    g.on_message(7, {"t": "tap", "n": 3})
    g.on_message(0, {"t": "tap", "n": 2})
    advance(g, 0.05)
    assert g.round_taps[0] == 2 and 7 not in g.round_taps


def test_malformed_and_oversized_tap_messages_are_ignored_or_capped():
    g = new()
    to_pull(g)
    for bad in ({"t": "tap"}, {"t": "tap", "n": "9"}, {"t": "tap", "n": -3}, {"t": "tap", "n": True}, {"t": "tap", "n": 1.5}, {"t": "x"}):
        g.on_message(0, bad)
    advance(g, 0.05)
    assert g.round_taps[0] == 0
    g.on_message(0, {"t": "tap", "n": 1000})
    advance(g, 0.05)
    assert g.round_taps[0] <= tug.TAP_BURST


def test_tap_rate_is_limited():
    g = new()
    to_pull(g)
    for _ in range(600):  # ten seconds of frantic scripting
        g.on_message(0, {"t": "tap", "n": 8})
        g.update(DT, pads_for(s0=0, s1=0, s2=0, s3=0))
    assert g.round_taps[0] <= tug.TAP_BURST + tug.TAP_RATE * 10 + 2


def test_more_taps_move_the_rope_toward_that_team():
    g = new()
    to_pull(g)
    tap_team(g, 1, seconds=1.0)
    assert g.pos > 0
    p = g.pos
    tap_team(g, 0, rate=12, seconds=1.0)
    assert g.pos < p


def test_even_pulling_leaves_the_rope_where_it_is():
    g = new()
    to_pull(g)
    for _ in range(120):
        for s in g.team:
            g.on_message(s, {"t": "tap", "n": 1}) if _ % 10 == 0 else None
        g.update(DT, pads_for(s0=0, s1=0, s2=0, s3=0))
    assert abs(g.pos) < 0.02


def test_uneven_teams_are_normalised_by_size():
    g = new((0, 1, 2))  # 2 against 1
    to_pull(g, (0, 1, 2))
    big, small = (0, 1) if len(g.members(0)) > len(g.members(1)) else (1, 0)
    for frame in range(120):
        if frame % 10 == 0:
            for s in g.members(big) + g.members(small):
                g.on_message(s, {"t": "tap", "n": 1})
        g.update(DT, pads_for(s0=0, s1=0, s2=0))
    assert abs(g.pos) < 0.02


def test_dragging_the_flag_to_the_end_wins_the_round_at_once():
    g = new()
    to_pull(g)
    for _ in range(40):
        tap_team(g, 0, rate=12, seconds=0.5)
        if g.state != "pull":
            break
    assert g.state == "round_end" and g.winner == 0 and g.wins == [1, 0]


def test_timer_decides_the_round_and_a_dead_heat_goes_to_overtime():
    g = new()
    to_pull(g)
    tap_team(g, 1, rate=3, seconds=1.0)
    advance(g, tug.SECONDS[g.secs_i] + 0.2)
    assert g.state == "round_end" and g.winner == 1
    h = new()
    to_pull(h)
    advance(h, tug.SECONDS[h.secs_i] + 0.2)
    assert h.state == "pull" and h.overtime


def test_best_of_series_ends_with_a_winner_then_returns_to_lobby():
    g = new()
    for rnd in range(2):
        to_pull(g)
        for _ in range(60):
            tap_team(g, 0, rate=12, seconds=0.5)
            if g.state != "pull":
                break
        assert g.wins[0] == rnd + 1
        advance(g, tug.END_TIME + 0.1)
        if rnd == 0:
            assert g.state == "ready"
    assert g.state == "over" and g.winner == 0
    assert sum(g.total_taps.values()) > 0
    advance(g, tug.OVER_TIME + 0.1)
    assert g.state == "lobby"


def test_a_team_with_nobody_left_forfeits_the_round():
    g = new()
    to_pull(g)
    gone = set(g.members(1))
    left = tuple(s for s in (0, 1, 2, 3) if s not in gone)
    g.update(DT, pads_for(**{f"s{s}": 0 for s in left}))
    assert g.state == "round_end" and g.winner == 0


def test_host_picks_round_length_and_best_of_with_the_arrows():
    g = tug.Tug(768, 128, seed=1)
    up = pads_for(s0=pad.UP, s1=0)
    idle = pads_for(s0=0, s1=0)
    for _ in range(3):
        g.update(DT, idle)
    before = g.secs_i
    g.update(DT, up)
    g.update(DT, idle)
    assert g.secs_i == before + 1
    g.update(DT, pads_for(s0=pad.LEFT, s1=0))
    assert tug.BEST_OF[g.best_i] == 1


def test_late_joiner_is_a_spectator_and_gets_no_team_info_leak():
    g = new((0, 1, 2))
    to_pull(g, (0, 1, 2))
    g.update(DT, pads_for(s0=0, s1=0, s2=0, s3=0))
    g.on_message(3, {"t": "tap", "n": 1})
    assert 3 not in g.round_taps and g.private_info(3)["team"] is None
    assert g.private_info(0)["team"] in (0, 1)


def test_renders_every_state():
    g = tug.Tug(768, 128, seed=1)
    c = canvas()
    g.render(c)
    assert c.any()
    begin(g, (0, 1, 2, 3, 4, 5, 6, 7))
    seen = set()
    for frame in range(20000):
        if g.state == "pull" and frame % 7 == 0:
            for s in g.members(0):
                g.on_message(s, {"t": "tap", "n": 1})
        g.update(DT, pads_for(**{f"s{s}": 0 for s in range(8)}))
        if g.state not in seen:
            seen.add(g.state)
            g.render(c)
            assert c.any() and c.shape == (128, 768, 3)
        if g.state == "lobby":
            break
    assert {"ready", "pull", "round_end", "over"} <= seen


def test_tug_is_registered_and_rejects_small_displays():
    assert "tug" in REGISTRY
    try:
        tug.Tug(64, 32)
    except ValueError:
        return
    raise AssertionError("expected ValueError")

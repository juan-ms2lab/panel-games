import math

import pytest

from panel_games.games import REGISTRY, hockey as hk, pad
from test_games_multiplayer import DT, begin, canvas, pads_for


def new(slots=(0,), seed=1):
    g = hk.Hockey(768, 128, seed=seed)
    begin(g, slots)
    assert g.state == "serve"
    return g


def run(g, seconds, slots=(0,), each=None):
    pads = pads_for(**{f"s{s}": 0 for s in slots})
    for _ in range(int(seconds / DT)):
        if each:
            each(g)
        g.update(DT, pads)


def to_play(g, slots=(0,)):
    run(g, hk.SERVE_TIME + 0.1, slots)
    assert g.state == "play"


def aim(g, slot, x, y):
    g.on_message(slot, {"t": "move", "x": x, "y": y})


def test_teams_follow_slots_and_empty_sides_get_a_computer():
    g = new((0,))
    assert set(g.paddles) == {0, 101}
    g = new((0, 1, 2, 3))
    assert set(g.paddles) == {0, 1, 2, 3}
    assert hk.team_of(0) == hk.team_of(2) == 0 and hk.team_of(1) == hk.team_of(3) == 1


def test_mallets_stay_in_their_own_half_whatever_the_phone_sends():
    g = new((0, 1))
    to_play(g, (0, 1))
    for u, v in ((5, 5), (-5, -5), (1e9, 1e9), (0, 0), (1, 1)):
        aim(g, 0, u, v)
        aim(g, 1, u, v)
        run(g, 0.5, (0, 1))
        p0, p1 = g.paddles[0], g.paddles[1]
        assert hk.R_PADDLE <= p0[0] <= hk.MID - hk.R_PADDLE and hk.MID + hk.R_PADDLE <= p1[0] <= hk.W - hk.R_PADDLE
        assert hk.R_PADDLE <= p0[1] <= hk.H - hk.R_PADDLE


def test_mallet_speed_is_limited():
    g = new((0,))
    to_play(g)
    aim(g, 0, 0.0, 0.0)
    run(g, 1.0)
    aim(g, 0, 1.0, 1.0)
    before = tuple(g.paddles[0][:2])
    run(g, DT)
    moved = math.hypot(g.paddles[0][0] - before[0], g.paddles[0][1] - before[1])
    assert moved <= hk.PADDLE_SPEED * DT + 1e-6


@pytest.mark.parametrize("bad", [{"t": "move"}, {"t": "move", "x": "0.5", "y": 0.5}, {"t": "move", "x": True, "y": 0.5},
                                 {"t": "move", "x": float("nan"), "y": 0.5}, {"t": "move", "x": float("inf"), "y": 0.5},
                                 {"t": "x", "x": 0.5, "y": 0.5}, {"t": "move", "x": None, "y": 0.2}])
def test_malformed_moves_are_ignored(bad):
    g = new((0,))
    to_play(g)
    before = g.target[0]
    g.on_message(0, bad)
    assert g.target[0] == before


def test_players_who_have_not_joined_cannot_move_anything():
    g = new((0,))
    g.on_message(3, {"t": "move", "x": 0.5, "y": 0.5})
    assert 3 not in g.target


def test_a_swinging_mallet_hits_the_puck_away():
    g = new((0,))
    to_play(g)
    g.paddles[0][:] = [200.0, 64.0, 0.0, 0.0]
    g.target[0] = (200.0, 64.0)
    g.paddles[101][:] = [700.0, 20.0, 0.0, 0.0]
    g.px, g.py, g.pvx, g.pvy = 214.0, 64.0, 0.0, 0.0
    aim(g, 0, 0.9, 0.5)  # swing right through the puck
    run(g, 0.1)
    assert g.pvx > 150


def test_puck_bounces_off_walls_and_stays_on_the_table():
    g = new((0,))
    to_play(g)
    g.px, g.py, g.pvx, g.pvy = 300.0, 20.0, 0.0, -400.0
    run(g, 0.2)
    assert g.pvy > 0 and g.py >= hk.R_PUCK
    g.px, g.py, g.pvx, g.pvy = 20.0, 10.0, -300.0, 0.0  # left wall above the goal mouth
    run(g, 0.2)
    assert g.state == "play" and g.pvx > 0


def test_puck_in_the_mouth_scores_for_the_other_team():
    g = new((0,))
    to_play(g)
    g.paddles[0][:] = [200.0, 120.0, 0.0, 0.0]
    g.target[0] = (200.0, 120.0)
    g.px, g.py, g.pvx, g.pvy = 40.0, 64.0, -500.0, 0.0
    run(g, 0.3)
    assert g.scores == [0, 1] and g.state == "goal" and g.last_scorer == 1
    run(g, hk.GOAL_TIME + 0.1)
    assert g.state == "play" and abs(g.px - hk.MID) < 40 and g.pvx < 0  # served toward the team that conceded


def test_reaching_the_target_score_ends_the_match_then_returns_to_the_lobby():
    g = new((0,))
    to_play(g)
    g.scores = [0, g.target_score - 1]
    g.paddles[0][:] = [200.0, 120.0, 0.0, 0.0]
    g.target[0] = (200.0, 120.0)
    g.px, g.py, g.pvx, g.pvy = 40.0, 64.0, -500.0, 0.0
    run(g, 0.3)
    assert g.state == "over" and g.winner == 1
    run(g, hk.OVER_TIME + 0.1)
    assert g.state == "lobby" and g.scores == [0, 0]


def test_the_computer_scores_on_an_idle_player_and_the_puck_never_escapes():
    g = new((0,))
    worst = 0.0
    for _ in range(int(240 / DT)):
        g.update(DT, pads_for(s0=0))
        assert -20 <= g.px <= hk.W + 20 and 0 <= g.py <= hk.H  # a goal frame may overshoot the line slightly
        worst = max(worst, math.hypot(g.pvx, g.pvy))
        if g.state == "lobby":
            break
    assert worst <= hk.MAX_PUCK + 1e-6
    assert g.scores[1] > 0 or g.state == "lobby"


def test_a_chasing_player_can_play_a_whole_match_against_the_computer():
    g = new((0,), seed=3)

    def chase(game):
        mx, my = game.paddles[0][:2]
        tx, ty = (game.px + 40, game.py) if mx < game.px - 12 and abs(my - game.py) < 8 else (game.px - 24, game.py)
        aim(game, 0, (tx - hk.R_PADDLE) / (hk.MID - 2 * hk.R_PADDLE), ty / hk.H)

    seen = set()
    for _ in range(int(900 / DT)):
        chase(g)
        g.update(DT, pads_for(s0=0))
        seen.add(g.state)
        if "over" in seen and g.state == "lobby":
            break
    assert {"play", "goal", "over", "lobby"} <= seen


def test_four_players_play_two_against_two():
    g = new((0, 1, 2, 3))
    to_play(g, (0, 1, 2, 3))
    run(g, 20, (0, 1, 2, 3))
    assert g.state in ("play", "goal", "over") and len(g.paddles) == 4


def test_late_joiners_and_leavers_change_the_mallets():
    g = new((0,))
    to_play(g)
    assert 101 in g.paddles
    g.update(DT, pads_for(s0=0, s1=0))
    assert 1 in g.paddles and 101 not in g.paddles
    g.update(DT, pads_for(s0=0, s1=None))
    assert 1 not in g.paddles and 101 in g.paddles


def test_host_picks_the_winning_score_and_start_needs_the_host():
    g = hk.Hockey(768, 128, seed=1)
    idle = pads_for(s0=0)
    for _ in range(3):
        g.update(DT, idle)
    before = g.win_i
    g.update(DT, pads_for(s0=pad.UP))
    g.update(DT, idle)
    assert g.win_i == before + 1 and g.state == "lobby"


def test_a_stalled_puck_gets_nudged():
    g = new((0,))
    to_play(g)
    g.paddles[0][:] = [100.0, 100.0, 0.0, 0.0]
    g.target[0] = (100.0, 100.0)
    g.paddles[101][:] = [700.0, 100.0, 0.0, 0.0]
    g.px, g.py, g.pvx, g.pvy = 30.0, 10.0, 0.0, 0.0
    run(g, 3.3)
    assert math.hypot(g.pvx, g.pvy) > 5 or abs(g.px - 30.0) > 1


def test_renders_every_state_and_is_registered():
    c = canvas()
    g = hk.Hockey(768, 128, seed=2)
    g.render(c)
    assert c.any()
    begin(g, (0, 1))
    seen = set()
    placed = False
    for _ in range(int(120 / DT)):
        g.update(DT, pads_for(s0=0, s1=0))
        if g.state not in seen:
            seen.add(g.state)
            g.render(c)
            assert c.any()
        if g.state == "play" and not placed:
            placed = True
            g.px, g.py, g.pvx, g.pvy = 40.0, 64.0, -600.0, 0.0
        if g.state == "goal":
            g.scores[1] = g.target_score - 1
    assert {"serve", "play", "goal"} <= seen
    g.state, g.winner = "over", 1
    g.render(c)
    assert c.any()
    assert "hockey" in REGISTRY
    with pytest.raises(ValueError):
        hk.Hockey(64, 32)

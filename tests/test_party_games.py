import random

import numpy as np
import pytest

from panel_games.games import bomber as bm
from panel_games.games import pad, sweep as sw, trivia as tr
from panel_games.games.base import Lobby
from test_games_multiplayer import DT, NONE4, begin, canvas, pads_for

QS = [{"q": f"Question number {i}?", "correct": f"right{i}", "wrong": [f"w{i}a", f"w{i}b", f"w{i}c"],
       "decks": ["nerd", "scifi"] if i % 2 else ["nerd"]} for i in range(30)]


# ---- Panel Trivia

def test_trivia_dataset_is_usable():
    qs = tr.load_questions()
    assert len(qs) > 1000
    assert all(len(q["wrong"]) == 3 and q["correct"] and q["decks"] is not None for q in qs)
    names = {d for d, _ in tr.DECKS if len(tr.deck_questions(d)) >= 5}
    assert names == {d for d, _ in tr.DECKS}  # every deck, including sci-fi and fantasy, has questions


def test_trivia_text_fits_the_screen():
    rng = random.Random(0)
    for q in rng.sample(tr.load_questions(), 200):
        lines, sc, lh = tr.fit(q["q"], 752, 54)
        assert lines and len(lines) * lh <= 54 + lh
        for opt in [q["correct"], *q["wrong"]]:
            lines, sc, lh = tr.fit(opt, 138, 42, (0.5, 0.45, 0.4, 0.35))
            assert lines and len(lines) * lh <= 42 + lh


def start_trivia(slots=(0, 1), count_i=0, seed=1):
    g = tr.Trivia(768, 128, seed=seed, questions=QS)
    g.count_i = count_i
    begin(g, slots)
    assert g.state == "question"
    return g


def answer(g, slot, idx, others=(0, 1)):
    bit = tr.ANSWER_BITS[idx][0]
    pads = pads_for(**{f"s{s}": (bit if s == slot else 0) for s in others})
    g.update(DT, pads)
    g.update(DT, pads_for(**{f"s{s}": 0 for s in others}))


def test_trivia_host_picks_deck_and_count_in_lobby():
    g = tr.Trivia(768, 128, seed=1, questions=QS)
    joined = pads_for(s0=0, s1=0)
    g.update(DT, joined)
    g.update(DT, pads_for(s0=pad.RIGHT, s1=pad.LEFT))  # only the host's arrows count
    g.update(DT, joined)
    assert g.deck_i == 1
    g.update(DT, pads_for(s0=pad.UP, s1=0))
    assert g.count_i == 2


def test_trivia_scoring_rewards_correct_and_fast():
    g = start_trivia(count_i=0)
    right, wrong = g.correct, (g.correct + 1) % 4
    for _ in range(int(2 / DT)):  # player 0 waits two seconds
        g.update(DT, pads_for(s0=0, s1=0))
    answer(g, 0, right)
    answer(g, 1, wrong)
    assert g.state == "reveal"  # everyone answered, so no need to wait out the clock
    assert 900 <= g.gained[0] <= 950 and g.gained[1] == 0 and g.scores == {0: g.gained[0], 1: 0}


def test_trivia_first_answer_is_final_and_early_answer_scores_more():
    a, b = start_trivia(), start_trivia(seed=1)
    right = a.correct
    for _ in range(int(1.2 / DT)):
        a.update(DT, pads_for(s0=0, s1=0))
        b.update(DT, pads_for(s0=0, s1=0))
    for _ in range(int(8 / DT)):
        b.update(DT, pads_for(s0=0, s1=0))
    answer(a, 0, right)
    answer(b, 0, right)
    wrong = (right + 1) % 4
    answer(a, 0, wrong)  # changing your mind is ignored
    assert a.answers[0][0] == right
    assert a.scores[0] > 0 or a.state == "question"
    answer(a, 1, right)
    answer(b, 1, right)
    assert a.gained[0] > b.gained[0]


def test_trivia_timeout_then_next_question_then_podium():
    g = start_trivia(slots=(0, 1, 2), count_i=0)  # 5 questions
    seen = 0
    for _ in range(int(400 / DT)):
        if g.state == "question" and g.remaining > 19 and g.qi % 2 == 0:
            answer(g, 0, g.correct, others=(0, 1, 2))
        g.update(DT, pads_for(s0=0, s1=0, s2=0))
        seen += g.state == "reveal"
        if g.state == "over":
            break
    assert g.state == "over" and seen > 0
    assert [s for s, _ in g.ranking][0] == 0 and g.ranking[0][1] > g.ranking[1][1]
    for _ in range(int((tr.OVER_TIME + 1) / DT)):
        g.update(DT, pads_for(s0=0, s1=0, s2=0))
    assert g.state == "lobby" and g.scores == {}


def test_trivia_renders_every_state_with_real_questions():
    g = tr.Trivia(768, 128, seed=3)
    c = canvas()
    g.render(c)
    begin(g, (0, 1, 2, 3, 4, 5, 6, 7))
    for _ in range(40):
        g.render(c)
        g.update(DT, pads_for(**{f"s{i}": 0 for i in range(8)}))
        for slot in range(8):
            if slot % 3 and g.state == "question":
                answer(g, slot, slot % 4, others=tuple(range(8)))
        if g.state == "question":
            g.remaining = 0.1
        if g.state in ("reveal", "over"):
            g.render(c)
    assert c.any()


# ---- Panel Bomber

def start_bomber(slots=(0,), seed=2):
    g = begin(bm.Bombers(768, 128, seed=seed), slots)
    assert g.state == "play"
    return g


def run(g, frames, **keys):
    slots = [int(k[1:]) for k in keys] or [0]
    for _ in range(frames):
        pads = [None] * 4
        for s in slots:
            pads[s] = keys.get(f"s{s}", 0)
        g.update(DT, pads)


def test_bomber_arena_layout():
    g = bm.Bombers(768, 128, seed=1)
    assert all(g.grid[y][x] == 1 for y in range(bm.GH) for x in (0, bm.GW - 1))
    assert all(g.grid[y][x] == 1 for y in range(0, bm.GH, 2) for x in range(0, bm.GW, 2))
    for sx, sy in bm.SPAWNS:
        dx, dy = (1 if sx < bm.GW // 2 else -1), (1 if sy < bm.GH // 2 else -1)
        assert all(g.grid[y][x] == 0 for x, y in ((sx, sy), (sx + dx, sy), (sx, sy + dy)))
    soft = sum(row.count(2) for row in g.grid)
    open_tiles = sum(row.count(0) for row in g.grid) + soft
    assert 0.4 < soft / open_tiles < 0.7
    seen, stack = {(1, 1)}, [(1, 1)]
    while stack:
        x, y = stack.pop()
        for dx, dy in bm.DIRS:
            n = (x + dx, y + dy)
            if n not in seen and g.grid[n[1]][n[0]] != 1:
                seen.add(n)
                stack.append(n)
    assert all(s in seen for s in bm.SPAWNS)


def test_bomber_moves_and_is_stopped_by_crates_and_walls():
    g = start_bomber()
    p = g.players[0]
    g.grid[1][3] = 2  # a crate two tiles east
    run(g, 120, s0=pad.RIGHT)
    assert p.tile() == (2, 1) and not g._blocked(p, (1, 1))
    run(g, 60, s0=pad.UP)  # wall above
    assert p.tile() == (2, 1)
    run(g, 120, s0=pad.LEFT)
    assert p.tile() == (1, 1)


def test_bomber_bomb_destroys_crate_and_kills_in_range_but_owner_can_escape():
    g = start_bomber(slots=(0, 1))
    g.grid[1][4] = 2
    for y in range(2, 8):  # a clear corridor down for the owner to escape along
        g.grid[y][1] = 0
    g.grid[1][2] = g.grid[1][3] = 0
    p0, p1 = g.players[0], g.players[1]
    p1.tx, p1.ty = 3, 1  # standing in the blast line
    g.update(DT, [pad.A, 0, None, None])
    assert (1, 1) in g.bombs and g.bombs[(1, 1)].owner == 0
    for _ in range(int(0.6 / DT)):  # owner walks off the bomb tile (allowed) and down the corridor
        g.update(DT, [pad.DOWN, 0, None, None])
    assert p0.tile() != (1, 1)
    run(g, int(2.0 / DT), s0=pad.DOWN, s1=0)
    assert not p1.alive and p0.alive
    assert g.state == "over" and g.result["winner"] == 0


def test_bomber_chain_reaction_and_bomb_limit():
    g = start_bomber()
    p = g.players[0]
    g.update(DT, [pad.A, None, None, None])
    g.update(DT, [0, None, None, None])
    g.update(DT, [pad.A, None, None, None])  # second drop on the same tile or beyond the limit: ignored
    assert len(g.bombs) == 1
    b2 = bm.Bomb((3, 1), p)
    g.bombs[(3, 1)] = b2
    g.bombs[(1, 1)].left = 0.01
    g.update(DT, [0, None, None, None])
    assert not g.bombs and (3, 1) in g.flames  # the blast lit the neighbouring bomb


def test_bomber_powerups_apply_once():
    g = start_bomber()
    p = g.players[0]
    g.powerups[(2, 1)] = "range"
    g.powerups[(1, 2)] = "speed"
    run(g, 90, s0=pad.RIGHT)
    assert p.range == 3 and (2, 1) not in g.powerups
    run(g, 90, s0=pad.LEFT)
    run(g, 90, s0=pad.DOWN)
    assert p.speed > bm.BASE_SPEED


def test_bomber_sudden_death_closes_in_and_kills():
    g = start_bomber(slots=(0, 1))
    g.play_t = bm.SUDDEN_DEATH_AT
    for _ in range(int(3 / DT)):
        g.update(DT, [0, 0, None, None])
    assert g.grid[0][5] == 1 and g.sd_i > 10
    g.grid[1][1] = 0
    assert g.spiral[0] == (0, 0) and len(set(g.spiral)) == bm.GW * bm.GH
    while g.state == "play" and g.sd_i < len(g.spiral):
        g.update(DT, [0, 0, None, None])
    assert g.state == "over"


def test_bomber_solo_and_draw_and_leaving():
    g = start_bomber()
    g._die(g.players[0])
    g.update(DT, [0, None, None, None])
    assert g.state == "over" and g.result["solo"]
    g2 = start_bomber(slots=(0, 1))
    g2.update(DT, [0, None, None, None])  # slot 1 left
    assert g2.state == "over" and g2.result["winner"] == 0
    g3 = start_bomber(slots=(0, 1))
    g3._die(g3.players[0])
    g3._die(g3.players[1])
    g3._finish([])
    assert g3.result["winner"] is None


def test_bomber_renders_every_state():
    g = bm.Bombers(768, 128, seed=4)
    c = canvas()
    g.render(c)
    begin(g, (0, 1, 2, 3))
    for i in range(300):
        g.update(DT, [pad.A if i % 50 == 0 else pad.RIGHT, pad.LEFT, pad.DOWN, pad.UP])
        g.render(c)
    g.play_t = bm.SUDDEN_DEATH_AT
    g.update(DT, [0, 0, 0, 0])
    g.render(c)
    g.state, g.result = "over", {"winner": None, "solo": False, "ranking": [0]}
    g.render(c)
    assert c.any()


# ---- Panel Sweep

def start_sweep(slots=(0, 1), seed=6):
    g = begin(sw.Sweep(768, 128, seed=seed), slots)
    assert g.state == "play"
    return g


def press(g, slot, keys, slots=(0, 1)):
    pads = [None] * 6
    for s in slots:
        pads[s] = keys if s == slot else 0
    g.update(DT, pads)
    g.update(DT, [0 if s in slots else None for s in range(6)])


def test_sweep_first_reveal_is_safe_and_places_all_mines():
    g = start_sweep()
    c = g.cursors[0]
    c.x, c.y = 20, 3
    press(g, 0, pad.A)
    assert sum(sum(row) for row in g.mine) == sw.MINES
    assert not any(g.mine[y][x] for y in range(2, 5) for x in range(19, 22))
    assert g.shown[3][20] and c.score >= 1 and c.alive


def test_sweep_flood_fill_counts_and_scoring():
    g = start_sweep()
    c = g.cursors[0]
    c.x, c.y = 10, 3
    press(g, 0, pad.A)
    revealed = sum(sum(row) for row in g.shown)
    assert c.score == revealed == sw.GW * sw.GH - sw.MINES - g.safe_left
    for y in range(sw.GH):
        for x in range(sw.GW):
            if g.shown[y][x] and g.count(x, y) == 0:  # an opened zero never leaves a safe neighbour hidden
                assert all(g.shown[ny][nx] or g.mine[ny][nx] or g.flag[ny][nx] is not None
                           for nx, ny in g._neighbours(x, y))


def test_sweep_mine_costs_a_life_stuns_and_eliminates():
    g = start_sweep()
    g.cursors[0].x, g.cursors[0].y = 10, 3
    press(g, 0, pad.A)
    mx, my = next((x, y) for y in range(sw.GH) for x in range(sw.GW) if g.mine[y][x])
    c = g.cursors[0]
    c.x, c.y = mx, my
    press(g, 0, pad.A)
    assert c.lives == sw.LIVES - 1 and c.stun > 0 and g.boom[my][mx]
    x_before = c.x
    g.update(DT, [pad.RIGHT, 0, None, None, None, None])
    assert c.x == x_before  # stunned players cannot move
    run_frames = int((sw.STUN + 0.2) / DT)
    for _ in range(run_frames):
        g.update(DT, [0, 0, None, None, None, None])
    mx2, my2 = next((x, y) for y in range(sw.GH) for x in range(sw.GW) if g.mine[y][x] and not g.boom[y][x])
    c.x, c.y = mx2, my2
    press(g, 0, pad.A)
    assert not c.alive and g.cursors[1].alive and g.state == "play"


def test_sweep_flags_chording_and_cursor_repeat():
    g = start_sweep()
    c = g.cursors[0]
    c.x, c.y = 10, 3
    press(g, 0, pad.A)
    press(g, 0, pad.B)
    assert g.flag[3][10] is None or g.shown[3][10]  # revealed squares cannot be flagged
    hx, hy = next((x, y) for y in range(sw.GH) for x in range(sw.GW) if not g.shown[y][x])
    c.x, c.y = hx, hy
    press(g, 0, pad.B)
    assert g.flag[hy][hx] == 0
    press(g, 0, pad.B)
    assert g.flag[hy][hx] is None
    c.x, c.y, c.hold, c.prev = 5, 3, 0.0, 0
    for _ in range(int(0.7 / DT)):  # holding RIGHT: instant step, a pause, then repeats
        g.update(DT, [pad.RIGHT, 0, None, None, None, None])
    assert 5 + 4 <= c.x <= 5 + 9
    # chord: a number whose flags match opens the rest of its neighbours
    for y in range(sw.GH):
        for x in range(sw.GW):
            if g.shown[y][x] and g.count(x, y) == 1:
                around = g._neighbours(x, y)
                mines = [(nx, ny) for nx, ny in around if g.mine[ny][nx]]
                if len(mines) == 1 and all(not g.shown[ny][nx] for nx, ny in around if not g.mine[ny][nx]):
                    g.flag[mines[0][1]][mines[0][0]] = 1
                    before = c.score
                    c.x, c.y = x, y
                    press(g, 0, pad.A)
                    assert c.score > before and c.alive
                    return


def test_sweep_clearing_the_board_gives_podium_with_flag_bonuses():
    g = start_sweep(slots=(0, 1, 2))
    g.cursors[0].x, g.cursors[0].y = 10, 3
    press(g, 0, pad.A, slots=(0, 1, 2))
    for y in range(sw.GH):
        for x in range(sw.GW):
            if not g.mine[y][x]:
                g.shown[y][x] = True
    g.safe_left = 0
    g.cursors[0].score, g.cursors[1].score, g.cursors[2].score = 100, 40, 60
    mine_cell = next((x, y) for y in range(sw.GH) for x in range(sw.GW) if g.mine[y][x])
    safe_cell = next((x, y) for y in range(sw.GH) for x in range(sw.GW) if not g.mine[y][x])
    g.flag[mine_cell[1]][mine_cell[0]] = 1  # correct flag: +10 for player 1
    g.flag[safe_cell[1]][safe_cell[0]] = 2  # wrong flag: -10 for player 2
    g._check_end()
    assert g.state == "over" and g.result["mode"] == "podium"
    assert g.result["ranking"] == [(0, 100), (1, 50), (2, 50)] or g.result["ranking"] == [(0, 100), (2, 50), (1, 50)]


def test_sweep_everyone_out_last_standing_wins():
    g = start_sweep()
    g.cursors[0].alive, g.cursors[0].died_at = False, 1.0
    g.cursors[1].alive, g.cursors[1].died_at = False, 9.0
    g._check_end()
    assert g.result["mode"] == "last" and g.result["ranking"][0][0] == 1


def test_sweep_renders_every_state():
    g = sw.Sweep(768, 128, seed=8)
    c = canvas()
    g.render(c)
    begin(g, (0, 1, 2))
    g.cursors[0].x, g.cursors[0].y = 12, 3
    press(g, 0, pad.A, slots=(0, 1, 2))
    press(g, 1, pad.B, slots=(0, 1, 2))
    g.render(c)
    g.state, g.result = "over", {"mode": "last", "solo": False, "ranking": [(1, 5), (0, 3)]}
    g.render(c)
    g.result = {"mode": "podium", "solo": False, "ranking": [(1, 5), (0, 3), (2, 1)]}
    g.render(c)
    assert c.any()


def test_new_games_reject_small_displays():
    for cls in (bm.Bombers, sw.Sweep, tr.Trivia):
        with pytest.raises(ValueError):
            cls(64, 32)

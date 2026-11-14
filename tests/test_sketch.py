import json
import time

import numpy as np
import pytest

from panel_games.games import pad, sketch as sk
from test_web import call, served
from test_games_multiplayer import DT, begin, canvas, pads_for

WORDS = ["robot", "dragon", "rocket", "bridge", "castle", "laptop", "magnet", "volcano"]


def new(slots=(0, 1, 2), seed=1, **kw):
    g = sk.Sketch(768, 128, seed=seed, words=WORDS, **kw)
    begin(g, slots)
    assert g.state == "intro"
    return g


def advance(g, seconds, slots=(0, 1, 2)):
    for _ in range(int(seconds / DT)):
        g.update(DT, pads_for(**{f"s{s}": 0 for s in slots}))


def to_draw(g, slots=(0, 1, 2)):
    advance(g, sk.INTRO_TIME + 0.1, slots)
    assert g.state == "draw"


def guess(g, slot, text_):
    g.on_message(slot, {"t": "guess", "text": text_})


# ---- matching

@pytest.mark.parametrize("a,b,ok", [
    ("Robot", "robot", True), ("  rObOt ", "robot", True), ("robots", "robot", True), ("circuitboard", "circuit board", True),
    ("circuit-board", "circuit board", True), ("3D printer", "3d printer", True), ("bot", "robot", False), ("", "robot", False),
])
def test_matching_is_forgiving_about_case_spaces_and_plurals(a, b, ok):
    assert sk.matches(a, b) is ok


def test_close_guesses_are_detected_but_not_accepted():
    assert sk.is_close("dragoon", "dragon") and sk.is_close("rokcet", "rocket") is False
    assert not sk.is_close("cat", "dog") and not sk.is_close("cut", "cat")  # too short to call close
    assert not sk.matches("dragoon", "dragon")


# ---- turns, scoring and secrecy

def test_every_player_draws_once_in_order_then_podium():
    g = new(slots=(0, 1, 2))
    drawers = []
    for _ in range(3):
        drawers.append(g.drawer)
        to_draw(g)
        g.remaining = 0.05
        advance(g, 0.2)
        assert g.state == "reveal"
        advance(g, sk.REVEAL_TIME + 0.1)
    assert drawers == [0, 1, 2] and g.state == "over"
    assert len(g.used) == 3  # no repeated words
    assert {s for s, _ in g.ranking} == {0, 1, 2}
    advance(g, sk.OVER_TIME + 0.1)
    assert g.state == "lobby" and g.scores == {}


def test_host_can_choose_two_rounds_each():
    g = sk.Sketch(768, 128, seed=2, words=WORDS)
    joined = pads_for(s0=0, s1=0)
    g.update(DT, joined)
    g.update(DT, pads_for(s0=pad.UP, s1=pad.DOWN))  # only the host's arrows count
    g.update(DT, joined)
    assert g.cycles_i == 1
    begin(g, (0, 1))
    assert g.order == [0, 1, 0, 1]


def test_faster_correct_guess_scores_more_and_drawer_is_paid():
    g = new()
    to_draw(g)
    word, drawer = g.word, g.drawer
    guessers = [s for s in (0, 1, 2) if s != drawer]
    advance(g, 5)
    guess(g, guessers[0], word)
    quick = g.solved[guessers[0]]
    advance(g, 40)
    guess(g, guessers[1], word.upper())
    slow = g.solved[guessers[1]]
    assert 100 < slow < quick <= 560
    assert g.scores[guessers[0]] == quick and g.scores[drawer] == quick // 4 + slow // 4


def test_wrong_and_close_guesses_give_private_feedback_only():
    g = new()
    to_draw(g)
    drawer = g.drawer
    guesser = next(s for s in (0, 1, 2) if s != drawer)
    guess(g, guesser, "banana")
    advance(g, 0.5)
    guess(g, guesser, g.word[:-1] + "x")
    status = [f[1] for f in g.private_info(guesser)["feedback"]]
    assert status[0] == "wrong" and status[1] in ("close", "wrong")
    assert g.scores[guesser] == 0 and guesser not in g.solved
    other = next(s for s in (0, 1, 2) if s not in (drawer, guesser))
    assert g.private_info(other)["feedback"] == []


def test_the_secret_word_only_goes_to_the_drawer_until_the_reveal():
    g = new()
    to_draw(g)
    drawer = g.drawer
    guesser = next(s for s in (0, 1, 2) if s != drawer)
    assert g.private_info(drawer)["word"] == g.word and g.private_info(drawer)["role"] == "drawer"
    assert "word" not in g.private_info(guesser) and g.private_info(guesser)["role"] == "guesser"
    assert g.word not in json.dumps(g.info())  # the public status never carries it
    assert set(g.info()["hint"]) <= {"_", " "}
    g.remaining = 0.05
    advance(g, 0.2)
    assert g.state == "reveal" and g.private_info(guesser)["word"] == g.word


def test_hints_reveal_letters_as_time_runs_out():
    g = new()
    to_draw(g)
    assert g.info()["hint"].count("_") == len(g.word)
    g.remaining = sk.TURN_TIME * 0.49
    advance(g, 0.1)
    assert len(g.revealed) == 1
    g.remaining = sk.TURN_TIME * 0.24
    advance(g, 0.1)
    assert len(g.revealed) == 2 and len(g.info()["hint"].replace("_", "")) == 2


def test_turn_ends_early_once_everyone_guessed_and_cannot_guess_twice():
    g = new()
    to_draw(g)
    for s in (0, 1, 2):
        if s != g.drawer:
            guess(g, s, g.word)
            advance(g, 0.5)
    before = dict(g.scores)
    for s in (0, 1, 2):
        guess(g, s, g.word)  # the drawer and solved players score nothing more
    assert g.scores == before
    advance(g, 3.0)
    assert g.state == "reveal"


def test_guessing_is_rate_limited_and_only_during_the_drawing_phase():
    g = new()
    guess(g, (g.drawer + 1) % 3, g.word)  # intro phase: ignored
    assert not g.solved
    to_draw(g)
    guesser = (g.drawer + 1) % 3
    guess(g, guesser, "aaa")
    guess(g, guesser, g.word)  # immediately again: too fast
    assert not g.solved and len(g.private_info(guesser)["feedback"]) == 1


def test_drawer_leaving_skips_to_the_next_turn():
    g = new()
    to_draw(g)
    drawer = g.drawer
    pads = pads_for(**{f"s{s}": 0 for s in (0, 1, 2) if s != drawer})
    for _ in range(int(1.0 / DT)):
        g.update(DT, pads)
    assert g.state in ("reveal", "intro", "over")


# ---- drawing

def test_only_the_drawer_can_draw_and_strokes_land_on_the_canvas():
    g = new()
    to_draw(g)
    drawer, other = g.drawer, (g.drawer + 1) % 3
    g.on_message(other, {"t": "draw", "ops": [{"c": 1, "s": 6, "p": [[10, 10], [200, 90]]}]})
    assert (g.canvas == sk.BG).all()
    g.on_message(drawer, {"t": "draw", "ops": [{"c": 1, "s": 6, "p": [[10, 10], [200, 90]]}]})
    assert (g.canvas[50, 105] == sk.PALETTE[1]).all()
    g.on_message(drawer, {"t": "draw", "ops": [{"c": 8, "s": 16, "p": [[105, 50]]}]})
    assert (g.canvas[50, 105] == sk.BG).all()  # eraser
    g.on_message(drawer, {"t": "draw", "ops": [{"clear": True}]})
    assert (g.canvas == sk.BG).all()


def test_malformed_drawing_messages_are_ignored_safely():
    g = new()
    to_draw(g)
    d = g.drawer
    for ops in ("nope", [None, 5, {"p": "x"}, {"c": "a", "s": 1, "p": [[1, 1]]}, {"c": True, "s": 1, "p": [[1, 1]]},
                         {"c": 1, "s": 1, "p": [[1], [1, 2, 3], ["a", "b"], [None, 1]]},
                         {"c": 99, "s": 9999, "p": [[-5000, 90000], [1e9, -1e9]]}]):
        g.on_message(d, {"t": "draw", "ops": ops})
    g.on_message(d, {"t": "draw"})
    g.on_message(d, {"t": "unknown"})
    g.on_message(d, {"t": "guess", "text": 5})


def test_canvas_is_cleared_for_each_new_drawer():
    g = new()
    to_draw(g)
    g.on_message(g.drawer, {"t": "draw", "ops": [{"c": 2, "s": 8, "p": [[50, 50], [300, 60]]}]})
    assert (g.canvas != sk.BG).any()
    g.remaining = 0.05
    advance(g, 0.2)
    advance(g, sk.REVEAL_TIME + 0.1)
    assert (g.canvas == sk.BG).all()


def test_sketch_renders_every_state_for_eight_players():
    g = sk.Sketch(768, 128, seed=3, words=WORDS)
    c = canvas()
    g.render(c)
    slots = tuple(range(8))
    begin(g, slots)
    pads = pads_for(**{f"s{s}": 0 for s in slots})
    for state in ("intro", "draw", "reveal"):
        primed = False
        while g.state != state:
            g.update(DT, pads)
            if g.state == "draw" and state == "reveal" and not primed:
                primed = True
                g.on_message(g.drawer, {"t": "draw", "ops": [{"c": 5, "s": 4, "p": [[20, 20], [400, 100]]}]})
                guess(g, (g.drawer + 1) % 8, g.word)
                g.remaining = 0.05
        g.render(c)
    g.state, g.ranking = "over", [(3, 900), (1, 500), (0, 200)]
    g.render(c)
    assert c.any()
    with pytest.raises(ValueError):
        sk.Sketch(64, 32)


def test_the_real_word_list_is_clean():
    from panel_games.games.words import WORDS as W
    assert len(set(W)) > 200 and all(w == w.lower() and 2 < len(w) <= 18 for w in W)


# ---- over the real HTTP API

def test_messages_and_private_state_over_http():
    srv, host, _ = served()
    port = srv.server_address[1]
    h = {"X-Requested-With": "panel-games", "Content-Type": "application/json"}
    api = lambda p, b=None: call("127.0.0.1", port, "POST", p, json.dumps(b or {}), h)
    body = lambda r: json.loads(r[1])
    try:
        assert api("/api/game/start", {"id": "sketch"})[0] == 200
        time.sleep(0.4)
        toks = [body(api("/api/game/join"))["token"] for _ in range(2)]
        api("/api/game/input", {"token": toks[0], "keys": pad.START})
        time.sleep(0.2)
        api("/api/game/input", {"token": toks[0], "keys": 0})
        end = time.time() + 8
        while time.time() < end and body(api("/api/game/state", {"token": toks[0]}))["state"] != "draw":
            time.sleep(0.2)
        states = [body(api("/api/game/state", {"token": t})) for t in toks]
        roles = sorted(s["role"] for s in states)
        assert roles == ["drawer", "guesser"]
        drawer_state = next(s for s in states if s["role"] == "drawer")
        guesser_state = next(s for s in states if s["role"] == "guesser")
        assert drawer_state["word"] and "word" not in guesser_state
        dtok, gtok = (toks[drawer_state["slot"]], toks[guesser_state["slot"]])
        pub = json.loads(call("127.0.0.1", port, "GET", "/api/status", None, h)[1])
        assert drawer_state["word"] not in json.dumps(pub["game"])
        assert api("/api/game/message", {"token": dtok, "data": {"t": "draw", "ops": [{"c": 1, "s": 5, "p": [[5, 5], [300, 100]]}]}})[0] == 200
        assert api("/api/game/message", {"token": gtok, "data": {"t": "guess", "text": drawer_state["word"]}})[0] == 200
        time.sleep(0.4)
        after = body(api("/api/game/state", {"token": gtok}))
        assert after["solved"] and after["score"] > 0 and after["feedback"][-1][1] == "correct"
        assert api("/api/game/message", {"token": "bad", "data": {}})[0] == 400
        assert api("/api/game/message", {"token": gtok, "data": "not an object"})[0] == 400
        assert api("/api/game/message", {"token": gtok, "data": {"t": "draw", "pad": "x" * 50000}})[0] == 400
        assert api("/api/game/state", {"token": "bad"})[0] == 400
    finally:
        host.stop()
        srv.shutdown()
        srv.server_close()

import http.client
import json
import time

import cv2
import numpy as np

from panel_games import web
from panel_games.games import REGISTRY, pad
from panel_games.host import Host
from panel_games.outputs import ScreenOutput

HDR = {"X-Requested-With": "panel-games", "Content-Type": "application/json"}


def call(host, port, method, path, body=None, headers=None):
    c = http.client.HTTPConnection(host, port)
    c.request(method, path, body=body, headers=headers or {})
    r = c.getresponse()
    data, cookie = r.read(), r.getheader("Set-Cookie")
    c.close()
    return r.status, data, cookie


def served(target=(768, 128)):
    screen = ScreenOutput()
    host = Host([screen], target)
    return web.serve_in_thread(host, "127.0.0.1", 0, screen), host, screen


def wait_for(cond, timeout=3.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.02)
    return False


def client(srv):
    port = srv.server_address[1]
    return lambda m, p, b=None, h=HDR: call("127.0.0.1", port, m, p, None if b is None else json.dumps(b), h)


def test_pages_and_catalog():
    srv, host, _ = served()
    api = client(srv)
    try:
        assert api("GET", "/")[0] == 200 and b"Panel Games" in api("GET", "/")[1]
        assert b"pixelated" in api("GET", "/screen")[1]
        games = json.loads(api("GET", "/api/games")[1])
        assert {g["id"] for g in games} == set(REGISTRY) and all(g["available"] for g in games)
        assert api("GET", "/nope")[0] == 404
    finally:
        srv.shutdown()
        srv.server_close()


def test_games_too_big_for_the_target_are_unavailable_not_a_crash():
    srv, host, _ = served((768, 64))
    api = client(srv)
    try:
        games = {g["id"]: g for g in json.loads(api("GET", "/api/games")[1])}
        assert not games["platformer"]["available"] and games["platformer"]["size"] is None
        assert games["elbirdo"]["available"] and games["elbirdo"]["size"] == "768x64"
        status, body, _ = api("POST", "/api/game/start", {"id": "platformer"})
        assert status == 400 and b"768x128" in body
        assert json.loads(api("GET", "/api/status")[1])["game"] is None
        assert api("POST", "/api/game/start", {"id": "elbirdo"})[0] == 200
        assert wait_for(lambda: json.loads(api("GET", "/api/status")[1])["game"]["id"] == "elbirdo")
    finally:
        host.stop()
        srv.shutdown()
        srv.server_close()


def test_play_a_game_over_http_and_see_it_on_the_screen():
    srv, host, screen = served()
    api = client(srv)
    try:
        assert api("POST", "/api/game/join", {})[0] == 400  # nothing running yet
        assert api("POST", "/api/game/start", {"id": "nope"})[0] == 400
        assert api("POST", "/api/game/start", {"id": "platformer"}, {"Content-Type": "application/json"})[0] == 403
        assert api("POST", "/api/game/start", {"id": "platformer"})[0] == 200
        slot = json.loads(api("POST", "/api/game/join", {})[1])
        assert slot["slot"] == 0 and slot["game"]["id"] == "platformer"
        assert api("POST", "/api/game/input", {"token": "bad", "keys": 2})[0] == 400
        assert api("POST", "/api/game/input", {"token": slot["token"], "keys": 9999})[0] == 400
        assert api("POST", "/api/game/input", {"token": slot["token"], "keys": pad.RIGHT | pad.B})[0] == 200
        assert wait_for(lambda: screen.count > 2)
        status, png, _ = api("GET", "/api/screen.png")
        frame = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
        assert status == 200 and frame.shape == (128, 768, 3) and frame.any()
        assert api("POST", "/api/game/state", {"token": slot["token"]})[0] == 200
        assert api("POST", "/api/game/message", {"token": slot["token"], "data": "x"})[0] == 400
        assert api("POST", "/api/game/leave", {"token": slot["token"]})[0] == 200
        assert api("POST", "/api/game/stop", {})[0] == 200
        assert json.loads(api("GET", "/api/status")[1])["game"] is None
        assert not screen.frame.any()  # outputs blank when the game ends
    finally:
        host.stop()
        srv.shutdown()
        srv.server_close()


def test_game_requests_are_rate_limited_per_client():
    srv, host, _ = served()
    api = client(srv)
    try:
        codes = [api("POST", "/api/game/state", {"token": "x"})[0] for _ in range(120)]
        assert 429 in codes and codes[0] == 400
    finally:
        srv.shutdown()
        srv.server_close()


def test_idle_game_ends(monkeypatch):
    from panel_games import host as hostmod
    monkeypatch.setattr(hostmod, "IDLE_SECONDS", 0.2)
    screen = ScreenOutput()
    h = Host([screen])
    h.start("pong")
    assert h.status()["id"] == "pong"
    assert wait_for(lambda: h.status() is None)
    h.stop()

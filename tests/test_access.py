import time

from panel_games import web as access


def test_limiter_burst_then_refill():
    rl = access.RateLimiter()
    assert rl.allow(("a", "x"), 10, 2) and rl.allow(("a", "x"), 10, 2)
    assert not rl.allow(("a", "x"), 10, 2)
    assert rl.allow(("b", "x"), 10, 2)  # other clients are independent
    time.sleep(0.15)
    assert rl.allow(("a", "x"), 10, 2)


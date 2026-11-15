import numpy as np
import pytest

from panel_games import bezel as bz
from panel_games import games
from panel_games.games.resurrection import Resurrection
from panel_games.outputs import DDPOutput, ScreenOutput, fit


def two_strips():
    f = np.full((64, 768, 3), 200, np.uint8)
    f[32:] = 100
    return f


def test_strip_gap_is_a_fifth_of_the_strip_and_keeps_pixels():
    c = bz.config({"strip_height": 32})
    o = bz.compose(two_strips(), c)
    assert o.shape == (70, 768, 3)  # 32 + round(32 / 5) + 32, full width
    assert (o[:32] == 200).all() and (o[38:] == 100).all()
    assert np.array_equal(o[32:38], bz.fill(768, 70, c)[32:38])


def test_no_bezel_settings_leave_frame_alone():
    f = two_strips()
    assert bz.compose(f, bz.config()) is f
    assert bz.compose(f[:32], bz.config({"strip_height": 32})) is not None  # one strip: nothing to split


def test_aspect_letterboxes_with_bezel():
    c = bz.config({"strip_height": 32, "aspect": "16:9"})
    o = bz.compose(two_strips(), c)
    assert o.shape == (432, 768, 3)
    y0 = (432 - 70) // 2
    assert (o[y0 : y0 + 32] == 200).all() and np.array_equal(o[:y0], bz.fill(768, 432, c)[:y0])
    tall = bz.compose(np.zeros((90, 90, 3), np.uint8), bz.config({"aspect": "16:9"}))
    assert tall.shape == (90, 160, 3)  # pillarbox


def test_gradient_and_pattern():
    c = bz.config({"colors": ["#ffffff", "#000000"], "pattern": "none"})
    f = bz.fill(4, 10, c)
    assert (f[0] == 255).all() and (f[-1] == 0).all() and (np.diff(f[:, 0, 0].astype(int)) <= 0).all()
    d = bz.fill(16, 16, bz.config({"colors": ["#000000", "#000000"], "pattern": "checker", "pattern_size": 4}))
    assert d[0, 0, 0] == 10 and d[0, 4, 0] == 0 and d[4, 4, 0] == 10


@pytest.mark.parametrize("bad", [{"pattern": "plaid"}, {"colors": ["#fff"]}, {"aspect": "wide"}, {"nope": 1}])
def test_bad_settings_rejected(bad):
    with pytest.raises(ValueError):
        bz.config(bad)


def test_resurrection_dark_bezel_and_per_game_config():
    c = games.bezel(Resurrection)
    assert c["strip_height"] == 32 and max(bz.fill(768, 70, c)[:, :, :].reshape(-1)) < 60  # dark
    conf = {"default": {"bezel": {"aspect": "16:9"}}, "resurrection": {"size": [768, 64], "bezel": {"gap": 0.5}}}
    games.check_config(conf)
    c = games.bezel(Resurrection, conf)
    assert c["aspect"] == "16:9" and c["gap"] == 0.5 and c["strip_height"] == 32
    assert games.pick_size(Resurrection, 768, 128, conf) == (768, 64)
    for bad in ({"nogame": {}}, {"resurrection": {"speed": 2}}, {"resurrection": {"size": [768]}},
                {"resurrection": {"bezel": {"pattern": "plaid"}}}):
        with pytest.raises(ValueError):
            games.check_config(bad)


def test_outputs_use_bezel():
    s = ScreenOutput()
    s.show(two_strips(), games.bezel(Resurrection))
    assert s.frame.shape == (70, 768, 3)
    c = bz.config({"ddp": True})
    small = np.full((32, 768, 3), 255, np.uint8)
    bg = bz.fill(768, 64, c)
    out = fit(small, 768, 64, bg)
    assert np.array_equal(out[:16], bg[:16]) and (out[16:48] == 255).all()
    assert DDPOutput("127.0.0.1", 9, 768, 64).show(small, c) is None  # sends without error

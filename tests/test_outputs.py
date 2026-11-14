import socket
import struct

import numpy as np

from panel_games.ddp import DDPSender
from panel_games.outputs import DDPOutput, fit


def test_fit_exact_upscale_center_and_crop():
    f = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3) + 1
    assert np.array_equal(fit(f, 3, 2), f)
    up = fit(f, 6, 4)  # 2x nearest neighbour
    assert np.array_equal(up[::2, ::2], f) and np.array_equal(up[1::2, 1::2], f)
    c = fit(f, 8, 4)  # still 2x (8 // 3), centered: one black column each side
    assert not c[:, 0].any() and not c[:, 7].any() and np.array_equal(c[:, 1:7], up)
    small = fit(np.ones((4, 4, 3), np.uint8), 2, 2)  # too big: center crop, never shrink
    assert small.shape == (2, 2, 3) and small.all()
    elbirdo = fit(np.ones((32, 768, 3), np.uint8), 768, 64)  # 768x32 on 768x64: scale 1, centered vertically
    assert not elbirdo[:16].any() and elbirdo[16:48].all() and not elbirdo[48:].any()


def recv_frame(rx):
    packets = []
    while True:
        data = rx.recv(2000)
        packets.append(data)
        if data[0] & 0x01:
            return packets


def test_ddp_packets_offsets_and_push():
    rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx.bind(("127.0.0.1", 0))
    rx.settimeout(2)
    out = DDPOutput("127.0.0.1", rx.getsockname()[1], 768, 64)
    frame = np.random.default_rng(1).integers(0, 255, (32, 768, 3), dtype=np.uint8)
    out.show(frame)
    packets = recv_frame(rx)
    expected = fit(frame, 768, 64).tobytes()
    assert len(expected) == 768 * 64 * 3
    got, seqs = bytearray(len(expected)), set()
    for i, p in enumerate(packets):
        flags, seq, dtype, dest, off, n = struct.unpack(">BBBBIH", p[:10])
        assert flags == (0x41 if i == len(packets) - 1 else 0x40)
        assert dtype == 0x0B and dest == 0x01 and n == len(p) - 10 <= 1440
        assert off == i * 1440
        got[off : off + n] = p[10:]
        seqs.add(seq)
    assert bytes(got) == expected and len(seqs) == 1 and 1 <= seqs.pop() <= 15
    s = DDPSender("127.0.0.1", rx.getsockname()[1])
    for _ in range(16):
        s.send(b"\0" * 3)
    assert s.seq == 1  # wraps 15 -> 1, skipping 0
    rx.close()

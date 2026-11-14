"""Command line: panel-games serve [--ddp HOST[:PORT]] [--target WxH] [--port 8090]."""

import argparse
import re
import time

from .ddp import PORT as DDP_PORT
from .host import Host
from .outputs import DDPOutput, ScreenOutput
from .web import lan_urls, serve_in_thread


def size(s: str) -> tuple[int, int]:
    m = re.fullmatch(r"(\d+)x(\d+)", s)
    if not m or not (0 < int(m[1]) <= 4096 and 0 < int(m[2]) <= 4096):
        raise argparse.ArgumentTypeError("expected WIDTHxHEIGHT, e.g. 768x64")
    return int(m[1]), int(m[2])


def endpoint(s: str) -> tuple[str, int]:
    host, _, port = s.rpartition(":") if ":" in s else (s, "", str(DDP_PORT))
    if not host or not port.isdigit():
        raise argparse.ArgumentTypeError("expected HOST or HOST:PORT")
    return host, int(port)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="panel-games")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve", help="run the game server")
    s.add_argument("--ddp", type=endpoint, help=f"send frames by DDP to HOST[:PORT] (default port {DDP_PORT})")
    s.add_argument("--target", type=size, default=(768, 128),
                   help="output canvas WxH; games that do not fit are unavailable (default 768x128)")
    s.add_argument("--port", type=int, default=8090, help="web port (default 8090)")
    s.add_argument("--bind", default="0.0.0.0", help="web address to listen on (default all, LAN-open)")
    s.add_argument("--fps", type=float, default=60, help="frames sent per second (default 60)")
    a = ap.parse_args(argv)

    screen = ScreenOutput()
    outputs = [screen]
    if a.ddp:
        outputs.append(DDPOutput(a.ddp[0], a.ddp[1], *a.target))
    host = Host(outputs, a.target, a.fps)
    srv = serve_in_thread(host, a.bind, a.port, screen)
    port = srv.server_address[1]
    print(f"panel-games: target {a.target[0]}x{a.target[1]}"
          + (f", DDP to {a.ddp[0]}:{a.ddp[1]}" if a.ddp else ", screen output only"))
    for url in lan_urls(port) or [f"http://localhost:{port}/"]:
        print(f"  phones: {url}    screen: {url}screen", flush=True)
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        host.stop()
        srv.shutdown()

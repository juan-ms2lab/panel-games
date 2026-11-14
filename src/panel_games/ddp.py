"""Minimal DDP (Distributed Display Protocol) sender: one RGB frame per call, split into UDP packets."""

import socket
import struct

PORT = 4048
MAX_DATA = 1440  # 480 RGB pixels per packet


class DDPSender:
    def __init__(self, host: str, port: int = PORT):
        self.addr = (host, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.seq = 0

    def send(self, rgb: bytes):
        """Send a row-major RGB frame; the last packet carries PUSH so the receiver shows it."""
        self.seq = self.seq % 15 + 1  # 1..15, 0 means unused
        for off in range(0, len(rgb), MAX_DATA):
            chunk = rgb[off : off + MAX_DATA]
            flags = 0x40 | (0x01 if off + MAX_DATA >= len(rgb) else 0)
            self.sock.sendto(struct.pack(">BBBBIH", flags, self.seq, 0x0B, 0x01, off, len(chunk)) + chunk, self.addr)

    def close(self):
        self.sock.close()

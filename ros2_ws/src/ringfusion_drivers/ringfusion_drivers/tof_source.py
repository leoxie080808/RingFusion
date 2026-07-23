"""ToF frame source for the AGX, matching the REAL firmware output.

The ESP32-C6 firmware emits ASCII CSV records over USB serial:
    ...TMF8829_FRAME_HEADER,<hdr bytes>,PAYLOAD,<pixel bytes>,<footer>...
A full 32x32 map arrives as TWO 32x16 subframes (even rows, then odd rows),
distances are in 0.25 mm units, confidence is one byte per zone. (The firmware
loads the official CMD_LOAD_CFG_32X32 mode -> 32x32 @ ~40 Hz; earlier builds used
48x32, so COLS here must match the flashed firmware's focal-plane mode.)

This module reuses that exact parsing (ported from the laptop heatmap.py) and
exposes a clean interface the pipeline consumes:

    src = SerialToFSource('/dev/ttyACM0')      # live
    src = ReplayToFSource(lines)               # from a captured text log
    frame = src.read()   # -> ToFFrame or None

ToFFrame.dist_m is metres, NaN where the sensor returned no distance.
"""
from __future__ import annotations
from dataclasses import dataclass
import time
import numpy as np

ROWS, COLS = 32, 32
SUBFRAME_ROWS = ROWS // 2
BYTES_PER_PIXEL = 3
PIXELS_PER_SUBFRAME = COLS * SUBFRAME_ROWS
PIXEL_BYTES_PER_SUBFRAME = PIXELS_PER_SUBFRAME * BYTES_PER_PIXEL
FRAME_FOOTER_BYTES = 12
EXPECTED_PAYLOAD_VALUES = PIXEL_BYTES_PER_SUBFRAME + FRAME_FOOTER_BYTES
DISTANCE_SCALE_MM = 0.25

FRAME_PREFIX = "TMF8829_FRAME_HEADER,"
PAYLOAD_MARKER = ",PAYLOAD,"
PREHEADER_SIZE = 5
FRAME_HEADER_SIZE = 16
RESULT_FRAME_TYPE = 0x10
SUB_RESULT_BIT = 0x40
RESULT_FORMAT_MASK = 0x3F
BASE_RESULT_FORMAT = 0x01
END_MARKER_LOW = 0xF7
END_MARKER_HIGH = 0xE0

# Zones with confidence below this are weak (still measured, just uncertain).
CONFIDENCE_STRONG = 6


@dataclass
class ToFFrame:
    dist_mm: np.ndarray      # (32,48) float32, NaN where no return
    confidence: np.ndarray   # (32,48) uint8
    t: float

    @property
    def dist_m(self):
        return self.dist_mm / 1000.0

    @property
    def valid(self):
        """Boolean mask: a real distance was returned."""
        return np.isfinite(self.dist_mm)

    @property
    def strong(self):
        return self.valid & (self.confidence >= CONFIDENCE_STRONG)


def _ints(text):
    out = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            v = int(item)
        except ValueError:
            return None
        if v < 0 or v > 255:
            return None
        out.append(v)
    return out


def _records_from_line(line):
    if FRAME_PREFIX not in line:
        return []
    parts = line.split(FRAME_PREFIX)
    return [FRAME_PREFIX + p for p in parts[1:] if PAYLOAD_MARKER in p]


def _parse_subframe(record):
    """Returns (subframe_index, dist_mm(16,48), conf(16,48)) or None."""
    if PAYLOAD_MARKER not in record:
        return None
    header_text, payload_text = record.split(PAYLOAD_MARKER, 1)
    header_text = header_text.removeprefix(FRAME_PREFIX)
    payload_text = payload_text.strip().rstrip(",")
    hv = _ints(header_text)
    pv = _ints(payload_text)
    if hv is None or pv is None:
        return None
    if len(hv) < PREHEADER_SIZE + FRAME_HEADER_SIZE:
        return None
    fh = hv[PREHEADER_SIZE:PREHEADER_SIZE + FRAME_HEADER_SIZE]
    frame_type = fh[0] & 0xF0
    layout = fh[1]
    if frame_type != RESULT_FRAME_TYPE:
        return None
    if (layout & RESULT_FORMAT_MASK) != BASE_RESULT_FORMAT:
        return None
    if len(pv) < EXPECTED_PAYLOAD_VALUES:
        return None
    pv = pv[:EXPECTED_PAYLOAD_VALUES]
    pixels = pv[:PIXEL_BYTES_PER_SUBFRAME]
    footer = pv[PIXEL_BYTES_PER_SUBFRAME:PIXEL_BYTES_PER_SUBFRAME + FRAME_FOOTER_BYTES]
    if footer[10] != END_MARKER_LOW or footer[11] != END_MARKER_HIGH:
        return None
    status = footer[8]
    if not (status & 0x01) or (status & 0xC0):      # invalid or aborted
        return None
    pb = np.asarray(pixels, np.uint16).reshape(PIXELS_PER_SUBFRAME, BYTES_PER_PIXEL)
    raw = pb[:, 0] | (pb[:, 1] << 8)
    conf = pb[:, 2].astype(np.uint8)
    dist = raw.astype(np.float32) * DISTANCE_SCALE_MM
    dist = dist.reshape(SUBFRAME_ROWS, COLS)
    conf = conf.reshape(SUBFRAME_ROWS, COLS)
    dist[raw.reshape(SUBFRAME_ROWS, COLS) == 0] = np.nan
    sub_index = 1 if (layout & SUB_RESULT_BIT) else 0
    return sub_index, dist, conf


class _Assembler:
    """Pairs even-row + odd-row subframes into a full 48x32 map."""
    def __init__(self):
        self._pending_even = None

    def feed(self, parsed):
        idx, dist, conf = parsed
        if idx == 0:
            self._pending_even = (dist, conf, time.monotonic())
            return None
        if self._pending_even is None:
            return None
        ed, ec, _ = self._pending_even
        self._pending_even = None
        D = np.full((ROWS, COLS), np.nan, np.float32)
        C = np.zeros((ROWS, COLS), np.uint8)
        D[0::2, :] = ed;   C[0::2, :] = ec
        D[1::2, :] = dist; C[1::2, :] = conf
        return ToFFrame(D, C, time.monotonic())


class SerialToFSource:
    """Live source: read the ESP32-C6 over USB serial."""
    def __init__(self, port, baud=115200, timeout=0.02):
        import serial
        self.ser = serial.Serial(port, baud, timeout=timeout)
        # This is the ESP32-C6's native-USB port; opening it alone does not
        # reset the chip or start streaming (confirmed: idle for 5s+ without
        # this). Toggling DTR/RTS pulses EN via the board's auto-reset
        # circuit, same as esptool/Arduino IDE do before a fresh boot.
        self.ser.dtr = False
        self.ser.rts = True
        time.sleep(0.1)
        self.ser.rts = False
        self.buf = bytearray()
        self.asm = _Assembler()
        time.sleep(1.0)          # let the port settle; do NOT flush (loses a subframe)

    def read(self):
        """Return the next COMPLETE frame, or None if not ready yet."""
        n = self.ser.in_waiting
        chunk = self.ser.read(n if n > 0 else 1)
        if chunk:
            self.buf.extend(chunk)
        if len(self.buf) > 1_000_000:
            self.buf.clear()
        while b"\n" in self.buf:
            raw, _, rest = self.buf.partition(b"\n")
            self.buf = bytearray(rest)
            line = raw.decode("ascii", errors="ignore").strip()
            for rec in _records_from_line(line):
                parsed = _parse_subframe(rec)
                if parsed is None:
                    continue
                frame = self.asm.feed(parsed)
                if frame is not None:
                    return frame
        return None

    def close(self):
        self.ser.close()


class ReplayToFSource:
    """Offline source: feed captured serial text (a list/iterable of lines)."""
    def __init__(self, lines):
        self.lines = iter(lines)
        self.asm = _Assembler()

    def read(self):
        for line in self.lines:
            for rec in _records_from_line(line.strip()):
                parsed = _parse_subframe(rec)
                if parsed is None:
                    continue
                frame = self.asm.feed(parsed)
                if frame is not None:
                    return frame
        return None

    def close(self):
        pass


def synthetic_frame(t=0.0):
    """A tilted wall + near blob in the REAL frame format (for offline runs)."""
    c, r = np.meshgrid(np.arange(COLS), np.arange(ROWS))
    u = c / (COLS - 1) - 0.5
    v = r / (ROWS - 1) - 0.5
    wall = 1500 + 300 * u
    blob = 700 * np.exp(-(((u) ** 2 + (v + 0.2) ** 2) / 0.03))
    D = np.clip(wall - blob, 10, 11000).astype(np.float32)
    D[np.random.rand(ROWS, COLS) < 0.02] = np.nan          # dropouts
    C = np.full((ROWS, COLS), 40, np.uint8)
    C[np.random.rand(ROWS, COLS) < 0.1] = 3                 # some weak zones
    return ToFFrame(D, C, t)

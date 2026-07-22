from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import serial
from serial import SerialException


# ===========================================================================
# Serial configuration
# ===========================================================================

PORT = "COM8"

# COM8 is the ESP32-C6 native USB Serial/JTAG connection.
# PySerial still requires a baud-rate argument.
BAUD_RATE = 115200

SERIAL_TIMEOUT_SECONDS = 0.02
STARTUP_DELAY_SECONDS = 1.0

# Prevent unbounded memory growth if a newline is lost.
MAX_SERIAL_BUFFER_BYTES = 1_000_000

# Live viewer: when the parse/draw loop falls behind the incoming frame rate, keep
# only the most recent complete serial lines each cycle and drop the older (stale)
# ones. ASCII-parsing every frame at 40 Hz is the bottleneck; without dropping, a
# fixed backlog (~0.5 s) builds up and the display always trails real time. We are
# only viewing, not recording, so showing the freshest frame is exactly what we want.
# 4 lines comfortably covers one even+odd subframe pair (one complete map).
KEEP_RECENT_LINES = 4


# ===========================================================================
# TMF8829 frame configuration
# ===========================================================================

ROWS = 32
COLS = 32

# Current result format:
#   byte 0: distance LSB
#   byte 1: distance MSB
#   byte 2: confidence/SNR
BASE_RESULT_FORMAT = 0x01
BYTES_PER_PIXEL = 3

# A 32×32 result arrives as two 32×16 row-interleaved subframes.
SUBFRAME_ROWS = ROWS // 2
PIXELS_PER_SUBFRAME = COLS * SUBFRAME_ROWS

PIXEL_BYTES_PER_SUBFRAME = (
    PIXELS_PER_SUBFRAME * BYTES_PER_PIXEL
)

FRAME_FOOTER_BYTES = 12

EXPECTED_PAYLOAD_VALUES = (
    PIXEL_BYTES_PER_SUBFRAME
    + FRAME_FOOTER_BYTES
)

# Each raw distance unit represents 0.25 mm.
DISTANCE_SCALE_MM = 0.25

# Reject an even-row frame if its odd-row partner takes too long.
MAX_SUBFRAME_PAIR_AGE_SECONDS = 3.0


# ===========================================================================
# Confidence visualization
# ===========================================================================

# Pixels below this confidence are treated as weak, but are still displayed.
CONFIDENCE_FADE_START = 6

# Pixels at or above this confidence are fully opaque.
CONFIDENCE_FULL_OPACITY = 30

# Lowest opacity used for a nonzero distance measurement.
# Increase this to make weak returns easier to see.
# Decrease it to make weak returns less distracting.
LOW_CONFIDENCE_MIN_ALPHA = 0.30


# ===========================================================================
# Heatmap configuration
# ===========================================================================

MIN_DISTANCE_MM = 0
MAX_DISTANCE_MM = 6000

# Number of 90-degree counterclockwise rotations:
#   0 = no rotation
#   1 = 90 degrees
#   2 = 180 degrees
#   3 = 270 degrees
ROTATE_QUARTER_TURNS = 0

FLIP_HORIZONTAL = True
FLIP_VERTICAL = False

FPS_WINDOW_SECONDS = 3.0
STATUS_INTERVAL_SECONDS = 5.0


# ===========================================================================
# TMF8829 protocol constants
# ===========================================================================

FRAME_PREFIX = "TMF8829_FRAME_HEADER,"
PAYLOAD_MARKER = ",PAYLOAD,"

PREHEADER_SIZE = 5
FRAME_HEADER_SIZE = 16

RESULT_FRAME_TYPE = 0x10

SUB_RESULT_BIT = 0x40
RESULT_FORMAT_MASK = 0x3F

END_MARKER_LOW = 0xF7
END_MARKER_HIGH = 0xE0


# ===========================================================================
# Data structures
# ===========================================================================

@dataclass
class ParsedSubframe:
    """One decoded 32×16 TMF8829 subframe."""

    subframe_index: int
    layout: int
    distance_mm: np.ndarray
    confidence: np.ndarray
    received_at: float


@dataclass
class ParserStats:
    """Runtime parser diagnostics."""

    records_seen: int = 0
    valid_subframes: int = 0
    completed_maps: int = 0

    malformed_records: int = 0
    incomplete_payloads: int = 0
    invalid_footers: int = 0
    sensor_invalid_frames: int = 0
    aborted_frames: int = 0
    unsupported_frames: int = 0

    orphan_odd_subframes: int = 0
    replaced_even_subframes: int = 0
    stale_even_subframes: int = 0
    dropped_stale_lines: int = 0

    layout_counts: dict[int, int] = field(default_factory=dict)


stats = ParserStats()

_pending_even_subframe: ParsedSubframe | None = None
_seen_layouts: set[int] = set()


# ===========================================================================
# Utility functions
# ===========================================================================

def parse_integer_list(text: str) -> list[int] | None:
    """Parse comma-separated decimal bytes."""

    values: list[int] = []

    for item in text.split(","):
        item = item.strip()

        if not item:
            continue

        try:
            value = int(item)
        except ValueError:
            return None

        if value < 0 or value > 255:
            return None

        values.append(value)

    return values


def extract_records_from_line(line: str) -> list[str]:
    """
    Extract TMF8829 records from one serial line.

    This tolerates ESP-IDF text appearing before the frame prefix.
    """

    if FRAME_PREFIX not in line:
        return []

    parts = line.split(FRAME_PREFIX)

    return [
        FRAME_PREFIX + part
        for part in parts[1:]
        if PAYLOAD_MARKER in part
    ]


def determine_subframe(layout: int) -> int:
    """
    Determine which half of the 48×32 map this record contains.

    Returns:
        0 for even-numbered rows.
        1 for odd-numbered rows.
    """

    return 1 if (layout & SUB_RESULT_BIT) else 0


def orient_map(array: np.ndarray) -> np.ndarray:
    """Apply the selected rotation and mirroring."""

    result = array

    turns = ROTATE_QUARTER_TURNS % 4

    if turns:
        result = np.rot90(result, k=turns)

    if FLIP_HORIZONTAL:
        result = np.fliplr(result)

    if FLIP_VERTICAL:
        result = np.flipud(result)

    return result


def create_alpha_map(
    distance_map: np.ndarray,
    confidence_map: np.ndarray,
) -> np.ndarray:
    """
    Convert confidence values into per-pixel opacity.

    Nonzero low-confidence measurements remain visible but faint.
    Zero-distance/NaN pixels are fully transparent and therefore white.
    """

    confidence_float = confidence_map.astype(np.float32)

    confidence_range = max(
        1.0,
        float(
            CONFIDENCE_FULL_OPACITY
            - CONFIDENCE_FADE_START
        ),
    )

    normalized = (
        confidence_float - CONFIDENCE_FADE_START
    ) / confidence_range

    normalized = np.clip(normalized, 0.0, 1.0)

    alpha_map = (
        LOW_CONFIDENCE_MIN_ALPHA
        + normalized
        * (1.0 - LOW_CONFIDENCE_MIN_ALPHA)
    )

    # No returned distance remains completely transparent.
    alpha_map[~np.isfinite(distance_map)] = 0.0

    return alpha_map


# ===========================================================================
# TMF8829 record parsing
# ===========================================================================

def parse_tmf8829_record(
    record: str,
) -> ParsedSubframe | None:
    """Validate and decode one TMF8829 CSV record."""

    stats.records_seen += 1

    if not record.startswith(FRAME_PREFIX):
        stats.malformed_records += 1
        return None

    if PAYLOAD_MARKER not in record:
        stats.malformed_records += 1
        return None

    header_text, payload_text = record.split(
        PAYLOAD_MARKER,
        maxsplit=1,
    )

    header_text = header_text.removeprefix(FRAME_PREFIX)
    payload_text = payload_text.strip().rstrip(",")

    header_values = parse_integer_list(header_text)
    payload_values = parse_integer_list(payload_text)

    if header_values is None or payload_values is None:
        stats.malformed_records += 1
        return None

    required_header_values = (
        PREHEADER_SIZE + FRAME_HEADER_SIZE
    )

    if len(header_values) < required_header_values:
        stats.malformed_records += 1
        return None

    frame_header = header_values[
        PREHEADER_SIZE:
        PREHEADER_SIZE + FRAME_HEADER_SIZE
    ]

    frame_id_and_mode = frame_header[0]
    frame_type = frame_id_and_mode & 0xF0
    focal_plane_mode = frame_id_and_mode & 0x0F

    layout = frame_header[1]
    base_result_format = layout & RESULT_FORMAT_MASK

    stats.layout_counts[layout] = (
        stats.layout_counts.get(layout, 0) + 1
    )

    if layout not in _seen_layouts:
        _seen_layouts.add(layout)

        print(
            "Detected TMF8829 frame: "
            f"type=0x{frame_type:02X}, "
            f"mode=0x{focal_plane_mode:X}, "
            f"layout=0x{layout:02X}"
        )

    if frame_type != RESULT_FRAME_TYPE:
        stats.unsupported_frames += 1
        return None

    if base_result_format != BASE_RESULT_FORMAT:
        stats.unsupported_frames += 1

        print(
            f"Unsupported result layout 0x{layout:02X}; "
            f"expected base format "
            f"0x{BASE_RESULT_FORMAT:02X}.",
            file=sys.stderr,
        )

        return None

    if len(payload_values) < EXPECTED_PAYLOAD_VALUES:
        stats.incomplete_payloads += 1

        if (
            stats.incomplete_payloads == 1
            or stats.incomplete_payloads % 10 == 0
        ):
            print(
                "Incomplete TMF8829 payload: "
                f"received {len(payload_values)} values; "
                f"expected at least "
                f"{EXPECTED_PAYLOAD_VALUES}.",
                file=sys.stderr,
            )

        return None

    # Ignore any unexpected trailing data.
    payload_values = payload_values[
        :EXPECTED_PAYLOAD_VALUES
    ]

    pixel_values = payload_values[
        :PIXEL_BYTES_PER_SUBFRAME
    ]

    footer = payload_values[
        PIXEL_BYTES_PER_SUBFRAME:
        PIXEL_BYTES_PER_SUBFRAME + FRAME_FOOTER_BYTES
    ]

    if (
        footer[10] != END_MARKER_LOW
        or footer[11] != END_MARKER_HIGH
    ):
        stats.invalid_footers += 1

        if (
            stats.invalid_footers == 1
            or stats.invalid_footers % 10 == 0
        ):
            print(
                "Invalid TMF8829 end-of-frame marker.",
                file=sys.stderr,
            )

        return None

    frame_status = footer[8]

    frame_is_valid = bool(frame_status & 0x01)
    frame_was_aborted = bool(frame_status & 0xC0)

    if not frame_is_valid:
        stats.sensor_invalid_frames += 1
        return None

    if frame_was_aborted:
        stats.aborted_frames += 1
        return None

    pixel_bytes = np.asarray(
        pixel_values,
        dtype=np.uint16,
    ).reshape(
        PIXELS_PER_SUBFRAME,
        BYTES_PER_PIXEL,
    )

    raw_distance = (
        pixel_bytes[:, 0]
        | (pixel_bytes[:, 1] << 8)
    )

    confidence = pixel_bytes[:, 2].astype(np.uint8)

    distance_mm = (
        raw_distance.astype(np.float32)
        * DISTANCE_SCALE_MM
    )

    half_distance = distance_mm.reshape(
        SUBFRAME_ROWS,
        COLS,
    )

    half_confidence = confidence.reshape(
        SUBFRAME_ROWS,
        COLS,
    )

    raw_distance_map = raw_distance.reshape(
        SUBFRAME_ROWS,
        COLS,
    )

    # Only a zero returned distance is treated as invalid.
    # Low-confidence nonzero distances remain visible.
    half_distance[raw_distance_map == 0] = np.nan

    stats.valid_subframes += 1

    return ParsedSubframe(
        subframe_index=determine_subframe(layout),
        layout=layout,
        distance_mm=half_distance,
        confidence=half_confidence,
        received_at=time.monotonic(),
    )


# ===========================================================================
# Subframe assembly
# ===========================================================================

def assemble_complete_map(
    subframe: ParsedSubframe,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Combine even-row and odd-row subframes into one 32×32 map."""

    global _pending_even_subframe

    if subframe.subframe_index == 0:
        if _pending_even_subframe is not None:
            stats.replaced_even_subframes += 1

        _pending_even_subframe = subframe
        return None

    if _pending_even_subframe is None:
        stats.orphan_odd_subframes += 1

        if (
            stats.orphan_odd_subframes == 1
            or stats.orphan_odd_subframes % 10 == 0
        ):
            print(
                "Orphan odd-row subframe: "
                f"0x01={stats.layout_counts.get(0x01, 0)}, "
                f"0x41={stats.layout_counts.get(0x41, 0)}, "
                f"orphans={stats.orphan_odd_subframes}",
                file=sys.stderr,
            )

        return None

    pair_age = (
        subframe.received_at
        - _pending_even_subframe.received_at
    )

    if pair_age > MAX_SUBFRAME_PAIR_AGE_SECONDS:
        stats.stale_even_subframes += 1
        _pending_even_subframe = None
        return None

    even_frame = _pending_even_subframe
    _pending_even_subframe = None

    complete_distance = np.full(
        (ROWS, COLS),
        np.nan,
        dtype=np.float32,
    )

    complete_confidence = np.zeros(
        (ROWS, COLS),
        dtype=np.uint8,
    )

    # First subframe contains rows 0, 2, 4, ..., 30.
    complete_distance[0::2, :] = even_frame.distance_mm
    complete_confidence[0::2, :] = even_frame.confidence

    # Second subframe contains rows 1, 3, 5, ..., 31.
    complete_distance[1::2, :] = subframe.distance_mm
    complete_confidence[1::2, :] = subframe.confidence

    stats.completed_maps += 1

    return complete_distance, complete_confidence


# ===========================================================================
# Diagnostics
# ===========================================================================

def print_status() -> None:
    """Print current parser statistics."""

    layout_summary = ", ".join(
        f"0x{layout:02X}:{count}"
        for layout, count in sorted(
            stats.layout_counts.items()
        )
    )

    if not layout_summary:
        layout_summary = "none"

    print(
        "Status | "
        f"records={stats.records_seen}, "
        f"subframes={stats.valid_subframes}, "
        f"maps={stats.completed_maps}, "
        f"orphans={stats.orphan_odd_subframes}, "
        f"replaced={stats.replaced_even_subframes}, "
        f"stale={stats.stale_even_subframes}, "
        f"incomplete={stats.incomplete_payloads}, "
        f"invalid-footer={stats.invalid_footers}, "
        f"dropped-stale={stats.dropped_stale_lines}, "
        f"layouts=[{layout_summary}]"
    )


def calculate_moving_fps(
    timestamps: deque[float],
    current_time: float,
) -> float:
    """Calculate FPS over a recent moving window."""

    while (
        timestamps
        and current_time - timestamps[0]
        > FPS_WINDOW_SECONDS
    ):
        timestamps.popleft()

    if len(timestamps) < 2:
        return 0.0

    duration = timestamps[-1] - timestamps[0]

    if duration <= 0:
        return 0.0

    return (len(timestamps) - 1) / duration


# ===========================================================================
# Main application
# ===========================================================================

def main() -> None:
    try:
        serial_port = serial.Serial(
            port=PORT,
            baudrate=BAUD_RATE,
            timeout=SERIAL_TIMEOUT_SECONDS,
        )
    except SerialException as error:
        raise SystemExit(
            f"Could not open {PORT}: {error}\n"
            "Close idf.py monitor, Arduino Serial Monitor, "
            "or any other program using COM8."
        ) from error

    print(f"Reading TMF8829 frames from {PORT}...")

    print(
        "Expected payload after PAYLOAD marker: "
        f"{EXPECTED_PAYLOAD_VALUES} values per subframe"
    )

    print(
        "Confidence opacity: "
        f"minimum alpha={LOW_CONFIDENCE_MIN_ALPHA:.2f}, "
        f"fade start={CONFIDENCE_FADE_START}, "
        f"full opacity={CONFIDENCE_FULL_OPACITY}"
    )

    # Do not clear the serial input buffer here. Clearing it can remove
    # one half of a two-subframe map.
    time.sleep(STARTUP_DELAY_SECONDS)

    initial_sensor_map = np.full(
        (ROWS, COLS),
        np.nan,
        dtype=np.float32,
    )

    initial_confidence_map = np.zeros(
        (ROWS, COLS),
        dtype=np.uint8,
    )

    initial_display_map = orient_map(initial_sensor_map)

    initial_display_confidence = orient_map(
        initial_confidence_map
    )

    initial_alpha = create_alpha_map(
        initial_display_map,
        initial_display_confidence,
    )

    figure, axis = plt.subplots(figsize=(11, 7))

    colormap = plt.get_cmap("turbo").copy()
    colormap.set_bad("white")

    image = axis.imshow(
        initial_display_map,
        origin="upper",
        interpolation="nearest",
        vmin=MIN_DISTANCE_MM,
        vmax=MAX_DISTANCE_MM,
        cmap=colormap,
        aspect="equal",
        alpha=initial_alpha,
    )

    colorbar = figure.colorbar(image, ax=axis)
    colorbar.set_label("Distance (mm)")

    axis.set_title(f"TMF8829 {COLS}×{ROWS} Depth Map")
    axis.set_xlabel("Column")
    axis.set_ylabel("Row")

    figure.tight_layout()

    plt.ion()
    plt.show(block=False)

    serial_buffer = bytearray()
    map_timestamps: deque[float] = deque()

    last_status_time = time.monotonic()

    try:
        while plt.fignum_exists(figure.number):
            bytes_available = serial_port.in_waiting

            if bytes_available > 0:
                chunk = serial_port.read(bytes_available)
            else:
                chunk = serial_port.read(1)

            if chunk:
                serial_buffer.extend(chunk)

            if len(serial_buffer) > MAX_SERIAL_BUFFER_BYTES:
                print(
                    "Serial buffer exceeded its maximum size. "
                    "Clearing incomplete serial data.",
                    file=sys.stderr,
                )

                serial_buffer.clear()

            # Drop stale buffered frames so the display stays current (see
            # KEEP_RECENT_LINES). Only triggers when we've fallen behind; when the
            # loop keeps up there are <= KEEP_RECENT_LINES lines and nothing is dropped.
            newline_count = serial_buffer.count(b"\n")
            if newline_count > KEEP_RECENT_LINES:
                parts = bytes(serial_buffer).split(b"\n")
                stats.dropped_stale_lines += newline_count - KEEP_RECENT_LINES
                serial_buffer = bytearray(
                    b"\n".join(parts[-(KEEP_RECENT_LINES + 1):])
                )

            while b"\n" in serial_buffer:
                raw_line, separator, remaining = (
                    serial_buffer.partition(b"\n")
                )

                if not separator:
                    break

                serial_buffer = bytearray(remaining)

                line = raw_line.decode(
                    "ascii",
                    errors="ignore",
                ).strip()

                records = extract_records_from_line(line)

                for record in records:
                    parsed_subframe = parse_tmf8829_record(
                        record
                    )

                    if parsed_subframe is None:
                        continue

                    assembled = assemble_complete_map(
                        parsed_subframe
                    )

                    if assembled is None:
                        continue

                    distance_map, confidence_map = assembled

                    display_map = orient_map(distance_map)

                    display_confidence = orient_map(
                        confidence_map
                    )

                    alpha_map = create_alpha_map(
                        display_map,
                        display_confidence,
                    )

                    image.set_data(display_map)
                    image.set_alpha(alpha_map)

                    display_rows, display_cols = (
                        display_map.shape
                    )

                    axis.set_xlim(
                        -0.5,
                        display_cols - 0.5,
                    )

                    axis.set_ylim(
                        display_rows - 0.5,
                        -0.5,
                    )

                    finite_mask = np.isfinite(display_map)

                    # Strong pixels meet the normal confidence threshold.
                    strong_mask = (
                        finite_mask
                        & (
                            display_confidence
                            >= CONFIDENCE_FADE_START
                        )
                    )

                    valid_count = int(
                        np.count_nonzero(finite_mask)
                    )

                    strong_count = int(
                        np.count_nonzero(strong_mask)
                    )

                    # Prefer strong pixels when displaying the range.
                    if strong_count > 0:
                        range_values = display_map[strong_mask]
                    else:
                        range_values = display_map[finite_mask]

                    current_time = time.monotonic()
                    map_timestamps.append(current_time)

                    fps = calculate_moving_fps(
                        map_timestamps,
                        current_time,
                    )

                    if range_values.size > 0:
                        nearest = float(
                            np.min(range_values)
                        )

                        farthest = float(
                            np.max(range_values)
                        )

                        axis.set_title(
                            f"TMF8829 {COLS}×{ROWS} Depth Map | "
                            f"{nearest:.0f}–{farthest:.0f} mm | "
                            f"{valid_count}/{ROWS * COLS} measured | "
                            f"{strong_count} strong | "
                            f"{fps:.2f} maps/s"
                        )
                    else:
                        axis.set_title(
                            f"TMF8829 {COLS}×{ROWS} Depth Map | "
                            f"No distance measurements | "
                            f"{fps:.2f} maps/s"
                        )

                    figure.canvas.draw_idle()

            current_time = time.monotonic()

            if (
                current_time - last_status_time
                >= STATUS_INTERVAL_SECONDS
            ):
                print_status()
                last_status_time = current_time

            plt.pause(0.001)

    except KeyboardInterrupt:
        print("\nHeatmap stopped.")

    except SerialException as error:
        print(
            f"\nSerial connection error: {error}",
            file=sys.stderr,
        )

    finally:
        print_status()
        serial_port.close()
        plt.close(figure)


if __name__ == "__main__":
    main()
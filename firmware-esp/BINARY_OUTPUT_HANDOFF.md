# Handoff: switch TMF8829 ToF output from ASCII to binary

## Why

The ESP32-C6 currently streams each ToF frame as **ASCII CSV** (every byte printed as
decimal text `"255,"`, ~3–4× larger than the raw bytes). Long, near-continuous USB
bursts radiate EMI into the camera's CSI ribbon and corrupt it (`PD_CRC_ERR` →
`INVALID_SETTINGS`; confirmed: camera is clean with the MCU unplugged, fails with it
plugged). **Binary output makes each USB burst ~4× shorter**, cutting the EMI duty
cycle. (Physical mitigation — routing the USB away from the ribbon + a ferrite clip —
is still the primary fix; this is complementary.)

Two files change, and they must change **together**:
- **Firmware:** `firmware-esp/components/tmf8829/tmf8829_shim.c` (the emit path)
- **ROS parser:** `ros2_ws/src/ringfusion_drivers/ringfusion_drivers/tof_source.py`

Nothing else in the pipeline changes (tof_driver publishes the frame's dynamic shape;
geometry reads cols/rows from the packet).

---

## Binary frame format

Framed, length-prefixed, CRC-checked so the parser can resync amid interleaved ESP log
text and drop any EMI-corrupted frame:

```
Offset  Size  Field
0       4     MAGIC   = 0xAA 0x55 0xC3 0x3C   (sync word; unlikely in log text)
4       2     LEN     uint16 little-endian = number of BODY bytes (= 21 + payload_len)
6       LEN   BODY    21 header bytes (PRE_HEADER 5 + FRAME_HEADER 16) + payload bytes,
                      copied verbatim from the sensor (unchanged from today's content)
6+LEN   2     CRC16   uint16 LE, CRC-16/CCITT-FALSE (poly 0x1021, init 0xFFFF) over BODY
```

The BODY is exactly what the ASCII path prints between `,PAYLOAD,` markers today (header
+ payload), just raw. For 32×32 each subframe is `LEN = 21 + 1548 = 1569` bytes; total
on-wire = `4 + 2 + 1569 + 2 = 1577` bytes vs ~6800 ASCII chars — the ~4× win.

---

## Firmware changes — `tmf8829_shim.c`

The three result callbacks currently print ASCII incrementally. Change them to **buffer**
the frame and emit **one binary blob** on end. Add near the top of the file:

```c
#include <string.h>          // memcpy

#define TMF_FRAME_MAX  2048  // max header+payload bytes (32x32 subframe = 1569; 48x32 = 2337 -> raise if you ever use 48x32)
static uint8_t  s_frame_buf[TMF_FRAME_MAX];
static uint16_t s_frame_len;

static uint16_t crc16_ccitt(const uint8_t *d, uint16_t n) {
    uint16_t crc = 0xFFFF;
    for (uint16_t i = 0; i < n; ++i) {
        crc ^= (uint16_t)d[i] << 8;
        for (int b = 0; b < 8; ++b)
            crc = (crc & 0x8000) ? (uint16_t)((crc << 1) ^ 0x1021) : (uint16_t)(crc << 1);
    }
    return crc;
}
```

Replace the bodies of the **result** callbacks (leave the histogram ones alone):

```c
void handleReceivedFrameHeaderData(void *dptr, uint8_t *data) {
    (void)dptr;
    s_frame_len = TMF8829_PRE_HEADER_SIZE + TMF8829_FRAME_HEADER_SIZE;   // 21
    memcpy(s_frame_buf, data, s_frame_len);
    s_result_line_open = true;
}

void handleReceivedResultData(void *dptr, uint8_t *data, uint16_t size) {
    (void)dptr;
    if (!s_result_line_open) { s_frame_len = 0; s_result_line_open = true; }  // headerless result
    if ((uint32_t)s_frame_len + size <= TMF_FRAME_MAX) {
        memcpy(s_frame_buf + s_frame_len, data, size);
        s_frame_len += size;
    }   // else: overflow -> frame dropped by CRC/short read on host; safe
}

void handleReceivedResultDataEnd(void *dptr) {
    (void)dptr;
    if (!s_result_line_open) return;
    uint8_t pre[6] = { 0xAA, 0x55, 0xC3, 0x3C,
                       (uint8_t)(s_frame_len & 0xFF), (uint8_t)(s_frame_len >> 8) };
    uint16_t crc = crc16_ccitt(s_frame_buf, s_frame_len);
    uint8_t  crcb[2] = { (uint8_t)(crc & 0xFF), (uint8_t)(crc >> 8) };
    fwrite(pre,  1, 6,           stdout);
    fwrite(s_frame_buf, 1, s_frame_len, stdout);
    fwrite(crcb, 1, 2,           stdout);
    fflush(stdout);
    s_result_line_open = false;
}
```

`print_raw_bytes` and `handleReceivedFrameHeaderData`'s old ASCII prints are no longer
used for results — you can leave `print_raw_bytes` for the histogram path.

### ESP-IDF gotchas (important)

1. **Binary-safe stdout.** The USB-Serial/JTAG VFS may translate `\n`. Prevent it once at
   startup (in `main.c`, before ranging starts):
   ```c
   #include "driver/usb_serial_jtag_vfs.h"   // (or esp_vfs_dev_usb_serial_jtag.h on older IDF)
   usb_serial_jtag_vfs_set_tx_line_endings(ESP_LINE_ENDINGS_LF);
   ```
   With LF endings, `0x0A` bytes pass through unchanged. (Alternative, fully guaranteed:
   emit via `usb_serial_jtag_write_bytes()` instead of `fwrite`.)
2. **Quiet the logs after boot.** ESP_LOG lines interleave with the binary stream. The
   host resyncs via MAGIC+CRC so it's tolerant, but for a clean stream call
   `esp_log_level_set("*", ESP_LOG_NONE);` right after the "ranging started" message.
3. **Boot ROM text** still appears at reset — the host already skips non-frame bytes.

---

## Parser changes — `tof_source.py`

Replace the ASCII line/CSV logic with binary framing. Keep the field extraction (it's the
same layout, just bytes). Constants (`ROWS,COLS=32,32`, footer markers, etc.) are unchanged.

```python
MAGIC = b"\xAA\x55\xC3\x3C"

def _crc16_ccitt(b: bytes) -> int:
    crc = 0xFFFF
    for x in b:
        crc ^= x << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if (crc & 0x8000) else (crc << 1) & 0xFFFF
    return crc

def _parse_binary_body(body: bytes):
    """body = 21 header bytes + payload. Returns (sub_index, dist(16,32), conf(16,32)) or None."""
    if len(body) < PREHEADER_SIZE + FRAME_HEADER_SIZE + PIXEL_BYTES_PER_SUBFRAME + FRAME_FOOTER_BYTES:
        return None
    fh = body[PREHEADER_SIZE:PREHEADER_SIZE + FRAME_HEADER_SIZE]      # 16-byte frame header
    if (fh[0] & 0xF0) != RESULT_FRAME_TYPE:                 return None
    layout = fh[1]
    if (layout & RESULT_FORMAT_MASK) != BASE_RESULT_FORMAT: return None
    payload = body[PREHEADER_SIZE + FRAME_HEADER_SIZE:]
    pixels  = payload[:PIXEL_BYTES_PER_SUBFRAME]                       # 512*3 = 1536
    footer  = payload[PIXEL_BYTES_PER_SUBFRAME:PIXEL_BYTES_PER_SUBFRAME + FRAME_FOOTER_BYTES]
    if footer[10] != END_MARKER_LOW or footer[11] != END_MARKER_HIGH:  return None
    status = footer[8]
    if not (status & 0x01) or (status & 0xC0):              return None
    pb   = np.frombuffer(pixels, np.uint8).reshape(PIXELS_PER_SUBFRAME, BYTES_PER_PIXEL).astype(np.uint16)
    raw  = pb[:, 0] | (pb[:, 1] << 8)
    conf = pb[:, 2].astype(np.uint8).reshape(SUBFRAME_ROWS, COLS)
    dist = (raw.astype(np.float32) * DISTANCE_SCALE_MM).reshape(SUBFRAME_ROWS, COLS)
    dist[raw.reshape(SUBFRAME_ROWS, COLS) == 0] = np.nan
    sub_index = 1 if (layout & SUB_RESULT_BIT) else 0
    return sub_index, dist, conf
```

`SerialToFSource.read()` becomes a byte-stream state machine (replace the line-splitting loop):

```python
def read(self):
    n = self.ser.in_waiting
    chunk = self.ser.read(n if n > 0 else 1)
    if chunk:
        self.buf.extend(chunk)
    if len(self.buf) > 1_000_000:
        del self.buf[:-4]                       # keep a possible partial magic
    while True:
        i = self.buf.find(MAGIC)
        if i < 0:
            if len(self.buf) > len(MAGIC):
                del self.buf[:-(len(MAGIC) - 1)] # drop scanned garbage, keep tail
            return None
        if len(self.buf) < i + 6:               # need MAGIC + LEN
            del self.buf[:i]; return None
        ln = self.buf[i + 4] | (self.buf[i + 5] << 8)
        end = i + 6 + ln + 2                     # + CRC16
        if len(self.buf) < end:
            del self.buf[:i]; return None        # wait for the rest
        body = bytes(self.buf[i + 6 : i + 6 + ln])
        crc_rx = self.buf[i + 6 + ln] | (self.buf[i + 7 + ln] << 8)
        del self.buf[:end]                       # consume this frame
        if _crc16_ccitt(body) != crc_rx:
            continue                             # corrupted (EMI) -> resync to next MAGIC
        parsed = _parse_binary_body(body)
        if parsed is None:
            continue
        frame = self.asm.feed(parsed)            # even/odd -> 32x32, unchanged
        if frame is not None:
            return frame
```

`_Assembler`, `ToFFrame`, and all the resolution constants stay exactly as they are.
`ReplayToFSource` needs the same MAGIC/CRC scan if you keep offline replay (or drop it).

---

## Testing / verification

1. **Flash**, then on the AGX read raw bytes and confirm the magic appears and CRCs pass:
   ```bash
   python3 - <<'PY'
   import serial, time
   MAGIC=b"\xAA\x55\xC3\x3C"
   s=serial.Serial('/dev/ttyACM1',115200,timeout=0.5)
   s.dtr=False; s.rts=True; time.sleep(0.1); s.rts=False; time.sleep(2)
   d=s.read(8000); s.close()
   print("magic occurrences:", d.count(MAGIC), "(expect several)")
   PY
   ```
2. **Parser test** (same harness we used for the 32×32 fix): import `SerialToFSource`, loop
   `read()` for 6 s, expect ~dozens of frames, shape `(32,32)`, ~900+ valid zones, ranges 0.1–2 m.
3. **EMI check** — the point of all this: with the MCU **plugged in**, rerun the camera
   reliability sweep (`gst-launch ... num-buffers=90`, 5×). Success rate should rise and
   `PD_CRC_ERR` count in `journalctl -u nvargus-daemon` should drop vs the ASCII build.

## Rollback

Keep the ASCII path behind a compile flag (`#ifdef TMF_BINARY_OUTPUT`) so you can A/B
without reflashing blind. The parser can sniff the first bytes (MAGIC vs `"TMF8829_"`)
to auto-detect, or just pin it to whichever firmware is flashed.

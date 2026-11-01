"""Tiny explicitly synthetic FIT writer for sync regression tests only.

Emits real FIT definitions, developer descriptions, arrays and CRCs. No personal
recordings or Garmin account are used; production code never writes FIT data.
"""

import struct
from fitparse.records import Crc


def synthetic_fit(*, schema=3, acceleration=True, bbi=True, final=True,
                  total=2, record=True, missing_timing=False, missing_axis=False,
                  native_count=2, invalid_bbi=False, reversed_times=False,
                  callback_after_end=False, elapsed_ms=1000, diagnostic=42,
                  x=1.0, nonfinite=False, header_size=14):
    body = bytearray()

    def definition(local, global_id, fields, dev=()):
        body.extend(bytes([0x40 | (0x20 if dev else 0) | local, 0, 0]))
        body.extend(struct.pack("<HB", global_id, len(fields)))
        for field in fields:
            body.extend(bytes(field))
        if dev:
            body.append(len(dev))
            for number, size in dev:
                body.extend(bytes([number, size, 0]))

    def data(local, value):
        body.append(local)
        body.extend(value)

    if bbi:
        definition(0, 207, [(1, 16, 13), (3, 1, 2)])
        data(0, bytes.fromhex("8e9893ea38064369be89b7ae7eab2d3b") + b"\0")
        descriptions = ([(25, "bbi_snapshot", 0x86, 224), (52, "final_bbi_snapshot", 0x86, 224)] if schema == 4 else [
            (11, "bbi_total", 0x86, 4), (13, "bbi_history", 0x84, 48),
            (14, "bbi_rx_ms", 0x86, 96), (24, "logger_meta", 0x86, 32),
            (42, "final_bbi_total", 0x86, 4), (40, "final_bbi_history", 0x84, 48),
            (41, "final_bbi_rx_ms", 0x86, 96), (51, "final_logger_meta", 0x86, 32)])
        descriptions.append((10, "accel_samples", 0x86, 4))
        definition(1, 206, [(0, 1, 2), (1, 1, 2), (2, 1, 2), (3, 32, 7)])
        for number, name, base, _ in descriptions:
            data(1, bytes([0, number, base]) + name.encode().ljust(32, b"\0"))

    if acceleration:
        fields = [(253, 4, 0x86), (0, 2, 0x84)]
        payload = struct.pack("<IH", 1136160000, 0)  # Arbitrary synthetic FIT epoch time.
        if not missing_timing:
            fields.append((1, 4, 0x84))
            payload += struct.pack("<2H", *([1000, 0] if reversed_times else [0, 1000]))
        for number, values in [(5, [x, x + 1]), (6, [0, 0]), (7, [1000, 999])]:
            if missing_axis and number == 7:
                continue
            fields.append((number, 8, 0x88))
            if nonfinite and number == 5:
                values[0] = float("nan")
            payload += struct.pack("<2f", *values)
        definition(2, 165, fields)
        data(2, payload)

    def snapshot(count, is_final):
        values, arrivals = [0] * 24, [0] * 24
        for sequence in range(max(1, count - 23), count + 1):
            slot = (sequence - 1) % 24
            values[slot] = 0 if invalid_bbi else 800 + sequence
            arrivals[slot] = elapsed_ms + 1 if callback_after_end else sequence * 10
        if schema == 4:
            packed = [4, count, 3, 0, 0, native_count, elapsed_ms, 1767225600] + values + arrivals
            return [(52 if is_final else 25, 224)], struct.pack("<56I", *packed)
        meta = [3, min(count, 24), 3, 0, 0, native_count, elapsed_ms, 1767225600]
        ids = [42, 40, 41, 51] if is_final else [11, 13, 14, 24]
        payload = struct.pack("<I24H24I8I", count, *values, *arrivals, *meta)
        return list(zip(ids, [4, 48, 96, 32])), payload

    if bbi:
        if record:
            dev, payload = snapshot(min(total, 2), False)
            definition(3, 20, [(253, 4, 0x86)], dev + [(10, 4)])
            data(3, struct.pack("<I", 1136160000) + payload + struct.pack("<I", diagnostic))
        if final:
            dev, payload = snapshot(total, True)
            definition(4, 18, [(253, 4, 0x86)], dev)
            data(4, struct.pack("<I", 1136160001) + payload)

    header = struct.pack("<BBHI4s", header_size, 0x20, 2100, len(body), b".FIT")
    if header_size == 14:
        header += struct.pack("<H", Crc.calculate(header))
    content = header + body
    return content + struct.pack("<H", Crc.calculate(content))

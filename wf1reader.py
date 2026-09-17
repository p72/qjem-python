"""Minimal reader for EViews workfiles (.wf1) as distributed with Q-JEM.

Supports the object types used in BASECASE.wf1:
  * type 11 : numeric series  -> numpy array
  * type 12 : coefficient vector (C_XXX) -> numpy array
Layout (reverse-engineered):
  directory of 70-byte records starting at 0xE2
    +0  uint16 type, +2 uint32 block size, +6 uint32 data bytes,
    +10 uint64 file offset of data block, +18 char[32] name
  data block = 22-byte header + float64 little-endian values
"""
import re
import struct

import numpy as np

DIR_START = 0xE2
REC_SIZE = 70
EVIEWS_NA = 1e-37


def read_wf1(path):
    buf = open(path, "rb").read()
    series, coefs = {}, {}
    pos = DIR_START
    while pos + REC_SIZE <= len(buf):
        rec = buf[pos:pos + REC_SIZE]
        name = rec[18:50].split(b"\0")[0]
        if not name or not re.fullmatch(rb"[A-Za-z0-9_]+", name):
            break
        typ, _size, nbytes, off = struct.unpack_from("<HIIQ", rec, 0)
        vals = np.frombuffer(buf[off + 22:off + 22 + nbytes], "<f8").copy()
        vals[np.isclose(vals, EVIEWS_NA, rtol=0, atol=1e-40)] = np.nan
        if typ == 11:
            series[name.decode().upper()] = vals
        elif typ == 12:
            coefs[name.decode().upper()] = vals
        pos += REC_SIZE
    return series, coefs


if __name__ == "__main__":
    import sys
    s, c = read_wf1(sys.argv[1])
    print(f"{len(s)} series, {len(c)} coefficient vectors")

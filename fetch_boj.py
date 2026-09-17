"""Download the official Q-JEM (2019) replication files from the Bank of Japan.

The BOJ files are not redistributed in this repository. They are fetched
from the BOJ website on first use and extracted to data/wp19e07/.
Source: Bank of Japan Working Paper Series No.19-E-7,
        https://www.boj.or.jp/en/research/wps_rev/wps_2019/wp19e07.htm
"""
import io
import os
import urllib.request
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
ZIP_URL = "https://www.boj.or.jp/en/research/wps_rev/wps_2019/data/wp19e07.zip"
DEST = os.path.join(HERE, "data", "wp19e07")
INPUT_DIR = os.path.join(DEST, "qjem_replication", "input")
REQUIRED = ("qjem_plain.txt", "BASECASE.wf1")


def ensure_boj_files():
    """Return the replication input directory, downloading it if needed."""
    if all(os.path.exists(os.path.join(INPUT_DIR, f)) for f in REQUIRED):
        return INPUT_DIR
    print(f"Downloading BOJ replication files from {ZIP_URL} ...")
    req = urllib.request.Request(ZIP_URL, headers={"User-Agent": "qjem-python"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        payload = resp.read()
    zipfile.ZipFile(io.BytesIO(payload)).extractall(DEST)
    missing = [f for f in REQUIRED if not os.path.exists(os.path.join(INPUT_DIR, f))]
    if missing:
        raise RuntimeError(f"unexpected archive layout; missing {missing} under {INPUT_DIR}")
    print(f"Extracted to {DEST}")
    return INPUT_DIR


if __name__ == "__main__":
    ensure_boj_files()

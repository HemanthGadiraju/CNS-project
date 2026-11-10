#!/usr/bin/env python3
"""
Download the firmware images listed in data/download_manifest.csv.

- resumable (skips files already present with a recorded hash)
- concurrent, with retries and per-file timeout
- records size, sha256 and HTTP status in download_log.csv
- handles http(s) and ftp

Many vendor links in the original list are dead or moved. Dead links are
logged with status != "ok" and the evaluation simply reports how many
images were obtainable (report this number in the paper).

Usage:
  python scripts/download.py --manifest data/download_manifest.csv --out firmware/raw \
         [--corpus C2] [--workers 6] [--limit 50]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

UA = "Mozilla/5.0 (X11; Linux x86_64) STUG-research-downloader/1.0"
MAX_BYTES = 600 * 1024 * 1024   # skip anything above 600 MB


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(url: str, dest: Path, timeout: int, retries: int) -> dict:
    tmp = dest.with_suffix(dest.suffix + ".part")
    last_err = ""
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url.replace(" ", "%20"), headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=timeout) as r, tmp.open("wb") as f:
                ctype = r.headers.get("Content-Type", "")
                if "text/html" in ctype.lower():
                    raise ValueError(f"got HTML instead of firmware ({ctype})")
                total = 0
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > MAX_BYTES:
                        raise ValueError("file exceeds size cap")
                    f.write(chunk)
            tmp.rename(dest)
            return {"status": "ok", "bytes": dest.stat().st_size, "sha256": sha256_file(dest),
                    "error": ""}
        except Exception as e:  # noqa: BLE001 - log everything, keep going
            last_err = f"{type(e).__name__}: {e}"
            tmp.unlink(missing_ok=True)
            time.sleep(min(2 ** attempt, 10))
    return {"status": "failed", "bytes": 0, "sha256": "", "error": last_err[:300]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/download_manifest.csv")
    ap.add_argument("--out", default="firmware/raw")
    ap.add_argument("--log", default="data/download_log.csv")
    ap.add_argument("--corpus", help="only rows whose corpus field contains this, e.g. C2, C3, C1r")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    m = pd.read_csv(args.manifest, dtype=str)
    if args.corpus:
        m = m[m["corpus"].str.contains(args.corpus, regex=False)]
    if args.limit:
        m = m.head(args.limit)

    done = {}
    log_path = Path(args.log)
    if log_path.exists():
        prev = pd.read_csv(log_path, dtype=str).fillna("")
        done = {r["image_id"]: r for _, r in prev.iterrows() if r["status"] == "ok"}

    todo = [r for _, r in m.iterrows()
            if not (r["image_id"] in done and (out / r["local_name"]).exists())]
    print(f"{len(m)} in selection, {len(m) - len(todo)} already downloaded, {len(todo)} to fetch")

    results = list(done.values())
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(fetch, r["url"], out / r["local_name"], args.timeout, args.retries): r
                for r in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            r = futs[fut]
            res = fut.result()
            results.append({"image_id": r["image_id"], "local_name": r["local_name"],
                            "url": r["url"], "corpus": r["corpus"], **res})
            print(f"[{i}/{len(todo)}] {res['status']:6s} {r['image_id']} {r['local_name'][:60]}"
                  + (f"  ({res['error'][:80]})" if res["error"] else ""), flush=True)

    with log_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image_id", "local_name", "url", "corpus", "status",
                                          "bytes", "sha256", "error"])
        w.writeheader()
        for row in results:
            w.writerow({k: row.get(k, "") for k in w.fieldnames})
    ok = sum(1 for r in results if r.get("status") == "ok")
    print(f"done: {ok}/{len(m)} available. log -> {log_path}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

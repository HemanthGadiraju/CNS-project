#!/usr/bin/env python3
"""
Unpack downloaded firmware and locate the Linux root filesystem.

Steps per image:
  1. unwrap vendor containers (zip / rar / 7z / tar / gz / xz), recursively
  2. carve the firmware blob with unblob (preferred) or binwalk -Me
  3. find the root filesystem (a directory with etc/ and bin/ or sbin/)
  4. record architecture (ELF header of busybox or the first binary),
     endianness, entropy and whether the image looks encrypted

Writes data/extract_index.csv, the input to run_stug.py.

External tools (install on the analysis machine):
  pip install unblob      # or: apt install binwalk (with sasquatch, jefferson, ubi_reader)
  apt install unrar p7zip-full squashfs-tools

Usage:
  python scripts/extract.py --raw firmware/raw --out firmware/extracted \
         --log data/download_log.csv --index data/extract_index.csv [--tool unblob|binwalk]
"""
from __future__ import annotations

import argparse
import math
import shutil
import subprocess
import tarfile
import zipfile
from collections import Counter
from pathlib import Path

import pandas as pd

ARCH = {0x08: "MIPS", 0x28: "ARM", 0xB7: "AARCH64", 0x03: "x86", 0x3E: "x86_64",
        0x14: "PPC", 0x15: "PPC64", 0xF3: "RISCV"}
ROOTFS_MARKERS = ("etc", "bin")
TIMEOUT = 900


def entropy(path: Path, sample: int = 4 << 20) -> float:
    data = path.read_bytes()[:sample]
    if not data:
        return 0.0
    c = Counter(data)
    n = len(data)
    return -sum(v / n * math.log2(v / n) for v in c.values())


def run(cmd: list[str], cwd: Path | None = None) -> bool:
    try:
        subprocess.run(cmd, cwd=cwd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       timeout=TIMEOUT, check=False)
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def unwrap(path: Path, dest: Path, depth: int = 0) -> list[Path]:
    """Return the innermost candidate blobs after removing vendor archives."""
    if depth > 4:
        return [path]
    name = path.name.lower()
    dest.mkdir(parents=True, exist_ok=True)
    out_dir = dest / f"unwrap_{depth}_{path.stem[:40]}"
    try:
        if zipfile.is_zipfile(path):
            with zipfile.ZipFile(path) as z:
                z.extractall(out_dir)
        elif name.endswith((".tar", ".tgz", ".tar.gz", ".tar.xz")) and tarfile.is_tarfile(path):
            with tarfile.open(path) as t:
                t.extractall(out_dir, filter="data")
        elif name.endswith(".rar"):
            out_dir.mkdir(exist_ok=True)
            if not run(["unrar", "x", "-o+", "-inul", str(path), str(out_dir) + "/"]):
                run(["7z", "x", "-y", f"-o{out_dir}", str(path)])
        elif name.endswith(".7z"):
            run(["7z", "x", "-y", f"-o{out_dir}", str(path)])
        else:
            return [path]
    except Exception:  # noqa: BLE001
        return [path]
    blobs = []
    for f in sorted(out_dir.rglob("*")):
        if f.is_file() and f.stat().st_size > 256 * 1024 and not f.suffix.lower() in {
                ".pdf", ".txt", ".doc", ".htm", ".html", ".exe", ".md5"}:
            blobs.extend(unwrap(f, dest, depth + 1))
    return blobs or [path]


def carve(blob: Path, dest: Path, tool: str) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    if tool == "unblob" and shutil.which("unblob"):
        run(["unblob", "--extract-dir", str(dest), "--depth", "8", str(blob)])
    elif shutil.which("binwalk"):
        run(["binwalk", "-Me", "-C", str(dest), str(blob)])
    return dest


def find_rootfs(root: Path) -> Path | None:
    best, best_score = None, 0
    for d in root.rglob("*"):
        if not d.is_dir():
            continue
        names = {p.name for p in d.iterdir()} if d.exists() else set()
        if not set(ROOTFS_MARKERS) <= names and not {"etc", "sbin"} <= names:
            continue
        score = sum(n in names for n in ("bin", "sbin", "etc", "lib", "usr", "www", "var", "tmp"))
        if score > best_score:
            best, best_score = d, score
    return best


def elf_arch(rootfs: Path) -> tuple[str, str]:
    cands = [rootfs / "bin" / "busybox", rootfs / "bin" / "sh"]
    cands += [p for p in (rootfs / "bin").glob("*")][:30] if (rootfs / "bin").exists() else []
    for c in cands:
        try:
            p = c.resolve()
            if not p.is_file() or not str(p).startswith(str(rootfs.resolve())):
                continue
            h = p.read_bytes()[:20]
        except OSError:
            continue
        if h[:4] == b"\x7fELF":
            endian = "LE" if h[5] == 1 else "BE"
            mach = int.from_bytes(h[18:20], "little" if endian == "LE" else "big")
            return ARCH.get(mach, f"0x{mach:x}"), endian
    return "unknown", ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="firmware/raw")
    ap.add_argument("--out", default="firmware/extracted")
    ap.add_argument("--log", default="data/download_log.csv")
    ap.add_argument("--index", default="data/extract_index.csv")
    ap.add_argument("--tool", choices=["unblob", "binwalk"], default="unblob")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    log = pd.read_csv(args.log, dtype=str).fillna("")
    log = log[log["status"] == "ok"]
    rows = []
    for _, r in log.iterrows():
        raw = Path(args.raw) / r["local_name"]
        work = Path(args.out) / r["image_id"]
        if work.exists() and args.force:
            shutil.rmtree(work)
        rootfs = find_rootfs(work) if work.exists() else None
        ent = entropy(raw) if raw.exists() else 0.0
        if rootfs is None and raw.exists():
            for blob in unwrap(raw, work / "_unwrap"):
                carve(blob, work / "_carved" / blob.stem[:40], args.tool)
            rootfs = find_rootfs(work)
        arch, endian = elf_arch(rootfs) if rootfs else ("unknown", "")
        status = "ok" if rootfs else ("encrypted?" if ent > 7.95 else "no_rootfs")
        rows.append({"image_id": r["image_id"], "corpus": r["corpus"], "raw": str(raw),
                     "rootfs": str(rootfs or ""), "arch": arch, "endian": endian,
                     "entropy": round(ent, 3), "status": status})
        print(f"{r['image_id']}: {status:10s} {arch:8s} {rootfs or ''}", flush=True)
    idx = pd.DataFrame(rows)
    idx.to_csv(args.index, index=False)
    print(idx["status"].value_counts().to_string())


if __name__ == "__main__":
    main()

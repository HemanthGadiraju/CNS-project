#!/usr/bin/env python3
"""
STUG dataset preprocessing.

Turns the raw firmware download list (vendor, product, version, date, url)
into the three corpora used by the STUG evaluation:

  C2  real OEM firmware images (routers + cameras), stratified per vendor/model
  C3  consecutive version pairs of the same device (for differential analysis)
  C1r OpenWrt sysupgrade images across releases (open-source reference chains)

C1 proper (SWUpdate / RAUC / Mender built from source) is not in the list;
see scenarios/ for the controlled scenarios that stand in for it.

Usage:
  python preprocess/preprocess.py --input firmware_download_list.csv --out data/
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from urllib.parse import unquote, urlparse

import numpy as np
import pandas as pd

SEED = 42

# --------------------------------------------------------------------------
# 1. Vendor recovery and normalisation
# --------------------------------------------------------------------------
VENDOR_ALIASES = {"d-link": "dlink", "tplink": "tp-link"}

DOMAIN_VENDOR = {
    "netgear.com": "netgear", "asus.com": "asus", "tp-link.com": "tp-link",
    "dlink.com": "dlink", "tendacn.com": "tenda", "linksys.com": "linksys",
    "belkin.com": "belkin", "trendnet.com": "trendnet", "ui.com": "ubiquiti",
    "ubnt.com": "ubiquiti", "zyxel.com": "zyxel", "zyxel.com.tr": "zyxel",
    "axis.com": "axis", "foscam.com": "foscam", "synology.com": "synology",
    "qnap.com": "qnap", "openwrt.org": "openwrt",
}
THIRD_PARTY_DOMAINS = {"dd-wrt.com": "dd-wrt", "groov.pl": "tomato",
                       "openwrt.org": "openwrt", "archive.org": "archive.org"}


def registered_domain(host: str) -> str:
    parts = host.lower().split(".")
    if len(parts) >= 3 and parts[-2] in {"com", "co"}:
        return ".".join(parts[-3:])
    return ".".join(parts[-2:])


def recover_vendor(row) -> str:
    v = str(row["vendor"]).strip().lower()
    v = VENDOR_ALIASES.get(v, v)
    if v in {"others", "nan", ""}:
        dom = registered_domain(row["host"])
        v = DOMAIN_VENDOR.get(dom, v)
    return v


# --------------------------------------------------------------------------
# 2. File-type filter
# --------------------------------------------------------------------------
FIRMWARE_EXT = {"bin", "zip", "tar", "trx", "chk", "img", "gz", "tgz", "xz",
                "rar", "7z", "w", "pkg", "ptz", "bix", "bbl", "fw", "ubi",
                "itb", "uimage", "dlf", "fmw"}
NON_FIRMWARE_EXT = {"exe", "dmg", "html", "htm", "aspx", "do", "pdf", "txt",
                    "apk", "msi", "iso", "pat", "eap", "spk", "qpkg", "md5",
                    "sha256", "sig", "asc", "mib", "mibs", "doc", "docx"}

LANGPACK_RE = re.compile(r"/lang/|_(?:[a-z]{2})\.tar$", re.I)
JUNK_RE = re.compile(r"boxdesc|mibs?\.zip|win32|win64|utility|driver|"
                     r"sourcecode|opensource|gpl|_src\b|release[_-]?notes|"
                     r"u-boot\.bin|bootloader", re.I)


def file_ext(path: str) -> str:
    name = path.rsplit("/", 1)[-1]
    if name.lower().endswith((".tar.gz", ".tar.xz", ".tar.bz2")):
        return "tgz"
    m = re.search(r"\.([A-Za-z0-9]{1,7})$", name)
    return m.group(1).lower() if m else ""


# --------------------------------------------------------------------------
# 3. Device class (STUG scope: network-facing routers/APs and IP cameras)
# --------------------------------------------------------------------------
CLASS_RULES = [
    # (vendor or '*', regex over model/product/url, class)
    ("asus", r"zenfone|vivo_pc|/zenpad|/notebook|/motherboard|/eeepc|/padfone", "other"),
    ("asus", r"/wireless/|\b(rt|gt|dsl|wl|rp|blue cave|lyra|zenwifi|4g)-", "router"),
    ("netgear", r"\b(gs|gss|xs|xsm|m4\d|m5\d|m7\d|fs|fsm|gc|jgs|ms|prosafe).*switch|^(gs|gss|xs|xsm|m4\d{3}|m7\d{3}|fs\d|fvs|gc\d|jgs|ms\d|gsm|xsm)", "switch"),
    ("netgear", r"^(rn|readynas)", "nas"),
    ("netgear", r"^(vmc|arlo)", "camera"),
    ("netgear", r"^(jr\d|r\d|rax|rbr|rbs|rbk|xr|ac\d|wndr|wnr|jnr|jwnr|dgn|dgnd|d\d{4}|c\d{4}|cg|ex\d|wn\d|wac|wnd9|wnap|ax\d|lax|nighthawk|orbi|mr\d|mk\d|ex|wpn|wgr|dm)", "router"),
    ("tp-link", r"^(nc\d|tapo|kc\d|vigi)", "camera"),
    ("tp-link", r"^(tl-sg|t1|t2|t3|tl-sl|tl-sf|jetstream|tl-st)", "switch"),
    ("tp-link", r"^(tl-r\d|archer|tl-wr|tl-wa|tl-mr|tl-wdr|tl-er|re\d|td-|td\d|eap|deco|m7|mr|tl-wpa|cpe|wbs|tl-wn|ax|ec\d|tx|c\d)", "router"),
    ("dlink", r"^dcs", "camera"),
    ("dlink", r"^(dgs|des|dxs|dms|dws|dqs)", "switch"),
    ("dlink", r"^(dns|dnr|dnh)", "other"),
    ("dlink", r"^(dir|dap|dsl|dwr|dsr|dwl|dva|dhp|covr|go-rt|dwm)", "router"),
    ("tenda", r"^(cp\d|ic\d|ct\d)", "camera"),
    ("tenda", r"^(teg|tef|sg\d)", "switch"),
    ("tenda", r".", "router"),
    ("linksys", r"^(lgs|srw|lgs)", "switch"),
    ("linksys", r"^(wvc|lcam)", "camera"),
    ("linksys", r".", "router"),
    ("belkin", r"^(f7d|f5d|f9k|f6d|f9j|n\d|ac\d|belkin play|belkin)", "router"),
    ("trendnet", r"^(tv-ip|tv-)", "camera"),
    ("trendnet", r"^(teg|tl2|tpe|ti-|te100|tew-?\d+ws|tl\d|tfc|teg)", "switch"),
    ("trendnet", r"^tew", "router"),
    ("ubiquiti", r"^(uvc|camera|g3|g4|airvision|unifi-video)", "camera"),
    ("ubiquiti", r"^(us|usxg|us6xg|es|edgeswitch|usw|ufiber)", "switch"),
    ("ubiquiti", r"^(unifi-cloudkey|uck|nvr)", "other"),
    ("ubiquiti", r".", "router"),
    ("zyxel", r"^(nsa|nas)", "nas"),
    ("zyxel", r"^(xgs|gs\d|es\d|mgs|xs\d)", "switch"),
    ("zyxel", r".", "router"),
    ("axis", r".", "camera"),
    ("foscam", r".", "camera"),
]


def classify_device(vendor: str, model: str, url: str) -> str:
    text_model = (model or "").lower().strip()
    text_url = url.lower()
    for v, pattern, cls in CLASS_RULES:
        if v != vendor and v != "*":
            continue
        rx = re.compile(pattern, re.I)
        if rx.search(text_model) or (pattern.startswith("/") and rx.search(text_url)) \
                or (("/" in pattern) and rx.search(text_url)):
            return cls
    return "unknown"


# --------------------------------------------------------------------------
# 4. Model, hardware revision and version extraction
# --------------------------------------------------------------------------
def path_segments(url: str) -> list[str]:
    return [unquote(s) for s in urlparse(url).path.split("/") if s]


def filename_stem(url: str) -> str:
    name = unquote(urlparse(url).path.rsplit("/", 1)[-1])
    name = re.sub(r"\.(tar\.gz|tar\.xz|tar\.bz2)$", "", name, flags=re.I)
    return re.sub(r"\.[A-Za-z0-9]{1,7}$", "", name)


def clean_model(m: str) -> str:
    m = re.sub(r"\(.*?\)", "", m)            # region tags (EU)(US)
    m = re.sub(r"\s+", "", m)
    return m.strip("_-. ").upper()


def extract_model_hw(vendor: str, product: str, url: str) -> tuple[str, str]:
    segs = path_segments(url)
    stem = filename_stem(url)
    hw = ""
    model = ""
    if vendor == "netgear" and "GDC" in segs:
        model = segs[segs.index("GDC") + 1]
    elif vendor == "asus" and "wireless" in segs:
        model = segs[segs.index("wireless") + 1]
    elif vendor == "tp-link":
        m = re.match(r"(.+?)[ _]?V(\d+(?:\.\d+)?)[_ ]", stem + "_", re.I)
        if m:
            model, hw = m.group(1), "V" + m.group(2)
        else:
            model = product
    elif vendor == "dlink":
        up = [s.upper() for s in segs]
        if "PRODUCTS" in up:
            model = segs[up.index("PRODUCTS") + 1]
        rev = re.search(r"/REV([A-Z])\b", url, re.I)
        hw = "REV" + rev.group(1).upper() if rev else ""
        model = model or product
    elif vendor == "trendnet":
        model = segs[0] if segs else product
        hwm = re.search(r"_v(\d+(?:\.\d+)?)$", model, re.I)
        if hwm:
            hw, model = "V" + hwm.group(1), model[: hwm.start()]
    elif vendor == "axis":
        model = re.sub(r"_(\d+_)+\d+.*$|_fimage$|_latest.*$", "", product)
    elif vendor == "linksys":
        m = re.match(r"FW_([A-Za-z0-9\-]+?)(V\d+)?_", stem, re.I)
        model = m.group(1) if m else product
        if m and m.group(2):
            hw = m.group(2).upper()
    elif vendor == "tenda":
        model = product
        hwm = re.search(r"V(\d+\.\d+)(?:br|rtl|qu|kf)", stem, re.I)
        hw = "V" + hwm.group(1) if hwm else ""
    else:
        model = product
    model = str(model).split("  ")[0].split(" ")[0] if vendor == "netgear" else str(model)
    return clean_model(model), hw.upper()


VERSION_GENERIC = re.compile(r"(?<![A-Za-z0-9])(?:fw|v|ver)?_?(\d+(?:[._]\d+){1,5})(?:[a-z]?\d*)?",
                             re.I)


def extract_version(vendor: str, url: str) -> tuple[str, str]:
    """Return (version_string, method). Empty string when not parseable."""
    stem = filename_stem(url)
    segs = path_segments(url)
    if vendor == "openwrt":
        m = re.search(r"/releases/([^/]+)/", url)
        return (m.group(1), "openwrt-release") if m else ("", "")
    if vendor == "asus":
        m = re.search(r"_(3004)(\d{3})(\d{3,6})", stem)
        if m:
            return f"3.0.0.4.{m.group(2)}.{m.group(3)}", "asus-3004"
    if vendor == "tp-link":
        m = re.search(r"(\d+\.\d+\.\d+)\s*Build\s*(\d{8})", stem, re.I)
        if m:
            return f"{m.group(1)}.{m.group(2)}", "tplink-build"
        m = re.search(r"_V\d+(?:\.\d+)?_(\d{6})", stem, re.I)
        if m:
            d = m.group(1)
            return f"20{d[:2]}.{d[2:4]}.{d[4:]}", "tplink-builddate"
    if vendor == "axis":
        for s in reversed(segs[:-1]):
            if re.fullmatch(r"\d+(?:_\d+)+", s):
                return s.replace("_", "."), "axis-path"
    if vendor == "ubiquiti":
        m = re.search(r"v(\d+\.\d+\.\d+(?:\.\d+)?)", stem, re.I)
        if m:
            return m.group(1), "ubnt"
    if vendor == "trendnet":
        m = re.search(r"\((?:fw)?v?(\d+\.\d+(?:[.b]\d+)*)", stem, re.I)
        if m:
            return m.group(1).lower().replace("b", "."), "trendnet-paren"
    if vendor == "zyxel":
        return "", ""  # vendor code names (e.g. 340RQ8D1) are not orderable
    cands = []
    for m in VERSION_GENERIC.finditer(stem):
        tok = m.group(1).replace("_", ".")
        prefixed = bool(re.match(r"(?i)(fw|v|ver)", m.group(0)))
        cands.append((prefixed, tok.count("."), len(tok), tok))
    if not cands:
        return "", ""
    cands.sort(reverse=True)
    best = cands[0][3]
    # reject things that are clearly model numbers or dates only
    if re.fullmatch(r"\d{4}", best) or best.count(".") == 0:
        return "", ""
    return best, "generic"


def version_key(v: str) -> tuple:
    if not v:
        return ()
    parts = re.split(r"[.\-_]", v)
    key = []
    for p in parts:
        num = re.match(r"(\d+)", p)
        key.append(int(num.group(1)) if num else 0)
        if p.lower().startswith("rc"):      # rc sorts before release
            key[-1] = -1
    return tuple(key)


# --------------------------------------------------------------------------
# 5. Pipeline
# --------------------------------------------------------------------------
def build_clean(df: pd.DataFrame, report: dict) -> pd.DataFrame:
    report["raw_rows"] = len(df)
    df = df.dropna(subset=["url"]).copy()
    df["url"] = df["url"].str.strip()
    df["url_norm"] = df["url"].str.replace(r"\?.*$", "", regex=True)
    df = df.drop_duplicates("url_norm")
    report["after_dedup"] = len(df)

    df["host"] = df["url_norm"].map(lambda u: urlparse(u).netloc.lower())
    df["vendor_norm"] = df.apply(recover_vendor, axis=1)
    dom = df["host"].map(registered_domain)
    df["origin"] = np.where(dom.isin(THIRD_PARTY_DOMAINS.keys()),
                            dom.map(THIRD_PARTY_DOMAINS), "oem")
    df.loc[df["vendor_norm"] == "openwrt", "origin"] = "openwrt"

    df["ext"] = df["url_norm"].map(file_ext)
    keep_ext = df["ext"].isin(FIRMWARE_EXT)
    report["dropped_non_firmware_ext"] = int((~keep_ext).sum())
    df = df[keep_ext]
    junk = df["url_norm"].str.contains(JUNK_RE) | df["url_norm"].str.contains(LANGPACK_RE)
    report["dropped_junk_or_langpack"] = int(junk.sum())
    df = df[~junk].copy()

    mh = df.apply(lambda r: extract_model_hw(r["vendor_norm"], str(r["product"]), r["url_norm"]),
                  axis=1, result_type="expand")
    df["model"], df["hw_rev"] = mh[0], mh[1]
    ver = df.apply(lambda r: extract_version(r["vendor_norm"], r["url_norm"]),
                   axis=1, result_type="expand")
    df["version_parsed"], df["version_method"] = ver[0], ver[1]
    df.loc[(df["version_parsed"] == "") & (df["version"] != "unknown"), "version_parsed"] = df["version"]
    df["date_parsed"] = pd.to_datetime(df["date"].where(df["date"] != "unknown"),
                                       format="%m/%d/%y", errors="coerce")
    df["device_class"] = df.apply(
        lambda r: classify_device(r["vendor_norm"], r["model"], r["url_norm"]), axis=1)
    df["firmware_name"] = df["url_norm"].map(lambda u: unquote(u.rsplit("/", 1)[-1]))
    df["device_key"] = df["vendor_norm"] + "|" + df["model"] + "|" + df["hw_rev"]
    report["clean_rows"] = len(df)
    report["version_parsed_rate"] = round(float((df["version_parsed"] != "").mean()), 3)
    return df


def stratified_take(g: pd.DataFrame, n: int, rng) -> pd.DataFrame:
    return g.sample(min(n, len(g)), random_state=rng)


def select_c2(df: pd.DataFrame, per_vendor: dict, per_model: int, report: dict) -> pd.DataFrame:
    """Real OEM images: routers + cameras, Linux-likely vendors, <= per_model per device."""
    pool = df[(df["origin"] == "oem") & df["device_class"].isin(["router", "camera"])
              & df["vendor_norm"].isin(per_vendor.keys())
              & (df["model"].str.len() > 1)].copy()
    pool["vkey"] = pool["version_parsed"].map(version_key)
    picks = []
    for vendor, quota in per_vendor.items():
        vp = pool[pool["vendor_norm"] == vendor]
        if vp.empty:
            continue
        # newest + one older per device, so C2 mixes current and legacy builds
        per_dev = []
        for _, g in vp.groupby("device_key"):
            g = g.sort_values(["vkey", "date_parsed"], ascending=False, na_position="last")
            per_dev.append(g.head(1))
            if per_model > 1 and len(g) > 2:
                per_dev.append(g.iloc[[len(g) // 2]])
        cand = pd.concat(per_dev)
        # balance device classes inside the vendor quota
        out = []
        classes = cand["device_class"].value_counts()
        for cls, cnt in classes.items():
            share = max(1, round(quota * cnt / classes.sum()))
            out.append(stratified_take(cand[cand["device_class"] == cls], share, SEED))
        sel = pd.concat(out)
        # one image per device first, then fill
        sel = sel.drop_duplicates("url_norm").head(quota)
        picks.append(sel)
    c2 = pd.concat(picks).drop(columns=["vkey"])
    c2["corpus"] = "C2"
    report["C2_images"] = len(c2)
    report["C2_by_vendor"] = c2["vendor_norm"].value_counts().to_dict()
    report["C2_by_class"] = c2["device_class"].value_counts().to_dict()
    return c2


def select_c3(df: pd.DataFrame, vendors: list[str], n_pairs: int, max_pairs_per_device: int,
              report: dict) -> pd.DataFrame:
    """Consecutive version pairs of the same device (vendor|model|hw_rev)."""
    pool = df[(df["origin"] == "oem") & df["device_class"].isin(["router", "camera"])
              & df["vendor_norm"].isin(vendors) & (df["version_parsed"] != "")].copy()
    pool["vkey"] = pool["version_parsed"].map(version_key)
    pool = pool.drop_duplicates(["device_key", "version_parsed"])
    rows = []
    for dk, g in pool.groupby("device_key"):
        if len(g) < 2:
            continue
        g = g.sort_values(["vkey", "date_parsed"])
        recs = g.to_dict("records")
        pairs = list(zip(recs[:-1], recs[1:]))[-max_pairs_per_device:]  # most recent pairs
        for old, new in pairs:
            ok, nk = old["vkey"], new["vkey"]
            # identical, or one is the other plus a build suffix (2.0.9 vs 2.0.9.5346345)
            if ok == nk or ok == nk[:len(ok)] or nk == ok[:len(nk)]:
                continue
            rows.append({
                "vendor": old["vendor_norm"], "device_key": dk, "model": old["model"],
                "hw_rev": old["hw_rev"], "device_class": old["device_class"],
                "old_version": old["version_parsed"], "new_version": new["version_parsed"],
                "old_date": old["date_parsed"], "new_date": new["date_parsed"],
                "old_url": old["url_norm"], "new_url": new["url_norm"],
                "versions_available": len(g)})
    pairs = pd.DataFrame(rows)
    if pairs.empty:
        report["C3_pairs"] = 0
        return pairs
    # balance across vendors
    quota = max(1, n_pairs // pairs["vendor"].nunique())
    sel = pd.concat([g.sort_values("versions_available", ascending=False).head(quota)
                     for _, g in pairs.groupby("vendor")])
    if len(sel) < n_pairs:  # top up round-robin so no vendor dominates
        rest = pairs.drop(sel.index).sort_values("versions_available", ascending=False)
        rest["rr"] = rest.groupby("vendor").cumcount()
        sel = pd.concat([sel, rest.sort_values(["rr", "versions_available"],
                                               ascending=[True, False])
                                  .head(n_pairs - len(sel)).drop(columns="rr")])
    sel = sel.reset_index(drop=True)
    sel.insert(0, "pair_id", [f"P{i:03d}" for i in range(1, len(sel) + 1)])
    report["C3_pairs"] = len(sel)
    report["C3_by_vendor"] = sel["vendor"].value_counts().to_dict()
    report["C3_candidate_pairs_total"] = len(pairs)
    return sel


OPENWRT_RELEASES = ["17.01.7", "18.06.9", "19.07.8", "21.02.1"]


def select_c1_openwrt(df: pd.DataFrame, n_devices: int, report: dict) -> pd.DataFrame:
    """Same device across OpenWrt releases: sysupgrade images, open-source ground truth."""
    o = df[(df["vendor_norm"] == "openwrt") & df["url_norm"].str.contains("sysupgrade")].copy()
    o["release"] = o["url_norm"].str.extract(r"/releases/([^/]+)/")[0]
    o["target"] = o["url_norm"].str.extract(r"/targets/([^/]+/[^/]+)/")[0]
    o = o[o["release"].isin(OPENWRT_RELEASES)]
    # device id = file name minus the release/target prefix
    o["board"] = o["firmware_name"].str.replace(
        r"^(openwrt|lede)-[\d.]+(?:-rc\d)?-[a-z0-9]+-[a-z0-9]+-", "", regex=True).str.replace(
        r"-squashfs.*$|-sysupgrade.*$|-ubifs.*$|-ext4.*$", "", regex=True)
    # Board names changed in 19.07/21.02 (DTS style "tplink_archer-c7-v2" vs
    # "archer-c7-v2"); normalise by dropping a leading "vendor_" token and
    # all separators so the same board matches across releases.
    o["device_id"] = (o["board"].str.replace(r"^[a-z0-9]+_", "", regex=True)
                               .str.replace(r"[^a-z0-9]", "", regex=True))
    o["arch_family"] = np.where(o["target"].str.contains(
        r"ar71xx|ath79|ramips|lantiq|mt7621|ath25|brcm63", na=False), "MIPS",
        np.where(o["target"].str.contains(r"ipq|mvebu|bcm27|mediatek|sunxi|imx|layerscape|kirkwood",
                                          na=False), "ARM", "other"))
    cover = o.groupby("device_id")["release"].nunique()
    o["n_releases"] = o["device_id"].map(cover)
    # prefer boards present in >= 3 releases, fall back to >= 2 per arch
    o = o[o["n_releases"] >= 2]
    uniq_dev = o.sort_values("n_releases", ascending=False).drop_duplicates("device_id")
    parts = []
    for arch, g in uniq_dev.groupby("arch_family"):
        if arch == "other":
            continue
        best = g[g["n_releases"] == g["n_releases"].max()]
        parts.append(best.sample(min(len(best), n_devices // 2), random_state=SEED))
    devs = pd.concat(parts)
    c1 = o[o["device_id"].isin(devs["device_id"])].drop_duplicates(["device_id", "release"])
    c1 = c1.sort_values(["device_id", "release"])
    c1["corpus"] = "C1r"
    report["C1r_devices"] = int(c1["device_id"].nunique())
    report["C1r_images"] = len(c1)
    report["C1r_arch"] = devs["arch_family"].value_counts().to_dict()
    return c1


def dev_test_split(keys: pd.Series, dev_frac: float) -> dict:
    """Group split by device so extraction rules are frozen before test (Sec. XI)."""
    uniq = sorted(keys.dropna().unique())
    rng = np.random.default_rng(SEED)
    dev = set(rng.choice(uniq, size=max(1, int(len(uniq) * dev_frac)), replace=False))
    return {k: ("dev" if k in dev else "test") for k in uniq}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", default="data")
    ap.add_argument("--c2-per-vendor", type=int, default=25)
    ap.add_argument("--c3-pairs", type=int, default=60)
    ap.add_argument("--c1-devices", type=int, default=10)
    ap.add_argument("--dev-frac", type=float, default=0.2)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report: dict = {}

    raw = pd.read_csv(args.input, dtype=str)
    clean = build_clean(raw, report)
    clean.to_csv(out / "firmware_clean.csv", index=False)
    report["clean_by_class"] = clean["device_class"].value_counts().to_dict()
    report["clean_by_origin"] = clean["origin"].value_counts().to_dict()

    c2_vendors = ["netgear", "asus", "tp-link", "dlink", "tenda", "linksys",
                  "trendnet", "ubiquiti", "belkin", "axis"]
    per_vendor = {v: args.c2_per_vendor for v in c2_vendors}
    c2 = select_c2(clean, per_vendor, per_model=2, report=report)

    c3_vendors = ["netgear", "asus", "tp-link", "dlink", "tenda", "linksys",
                  "trendnet", "ubiquiti", "axis"]
    c3 = select_c3(clean, c3_vendors, args.c3_pairs, max_pairs_per_device=2, report=report)
    c1 = select_c1_openwrt(clean, args.c1_devices, report)

    split = dev_test_split(pd.concat([c2["device_key"], c3.get("device_key", pd.Series(dtype=str))]),
                           args.dev_frac)
    c2["split"] = c2["device_key"].map(split)
    if not c3.empty:
        c3["split"] = c3["device_key"].map(split)
    c1["split"] = "reference"

    cols = ["vendor_norm", "model", "hw_rev", "device_class", "device_key", "version_parsed",
            "version_method", "date_parsed", "ext", "firmware_name", "url_norm", "split"]
    c2[cols + ["corpus"]].rename(columns={"vendor_norm": "vendor", "url_norm": "url",
                                          "version_parsed": "version", "date_parsed": "date"}) \
        .to_csv(out / "c2_real_firmware.csv", index=False)
    c3.to_csv(out / "c3_version_pairs.csv", index=False)
    c1[["device_id", "release", "target", "arch_family", "firmware_name", "url_norm", "corpus",
        "split"]].rename(columns={"url_norm": "url"}).to_csv(out / "c1_openwrt_reference.csv",
                                                             index=False)

    # unified download manifest (each URL once, with corpus membership)
    man = []
    for _, r in c2.iterrows():
        man.append((r["url_norm"], r["vendor_norm"], r["device_key"], r["version_parsed"], "C2", r["split"]))
    for _, r in c3.iterrows():
        man.append((r["old_url"], r["vendor"], r["device_key"], r["old_version"], f"C3:{r['pair_id']}:old", r["split"]))
        man.append((r["new_url"], r["vendor"], r["device_key"], r["new_version"], f"C3:{r['pair_id']}:new", r["split"]))
    for _, r in c1.iterrows():
        man.append((r["url_norm"], "openwrt", r["device_id"], r["release"], "C1r", "reference"))
    m = pd.DataFrame(man, columns=["url", "vendor", "device_key", "version", "corpus", "split"])
    m = (m.groupby("url", as_index=False)
          .agg({"vendor": "first", "device_key": "first", "version": "first",
                "corpus": lambda s: ";".join(sorted(set(s))), "split": "first"}))
    m.insert(0, "image_id", [f"IMG{i:04d}" for i in range(1, len(m) + 1)])
    m["local_name"] = m.apply(lambda r: f"{r['image_id']}_{Path(urlparse(r['url']).path).name}", axis=1)
    m.to_csv(out / "download_manifest.csv", index=False)
    report["manifest_unique_images"] = len(m)

    (out / "preprocessing_report.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()

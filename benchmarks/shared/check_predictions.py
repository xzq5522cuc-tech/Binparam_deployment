#!/usr/bin/env python3
"""Verify the integrity of every C4 prediction file before it is scored.

Checks, per sample in the manifest:
  * the file exists and is exactly 6890*3 float32 = 82,680 bytes
  * all vertex coordinates are finite
  * the sample id belongs to the manifest

Writes a JSON report and exits non-zero if anything is invalid, so that a
resumed run cannot silently mix truncated or foreign files into the metrics.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

EXPECT = 6890 * 3 * 4
PACK = Path("/home/odroid/pymaf_final/dataset/3dpw_full")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", default=str(PACK))
    ap.add_argument("--predictions", default="/home/odroid/eval_c4/predictions")
    ap.add_argument("--out", default="/home/odroid/eval_c4/predictions_integrity.json")
    args = ap.parse_args()

    pack = Path(args.pack)
    preds = Path(args.predictions)
    ids = [l.split()[0] for l in (pack / "manifest.txt").read_text().splitlines() if l.strip()]
    known = set(ids)

    files = sorted(preds.glob("*.pred_vertices.bin"))
    bad_size, bad_finite, foreign = [], [], []
    seen = set()
    for f in files:
        sid = f.name.replace(".pred_vertices.bin", "")
        if sid not in known:
            foreign.append(sid)
            continue
        seen.add(sid)
        if f.stat().st_size != EXPECT:
            bad_size.append({"id": sid, "bytes": f.stat().st_size})
            continue
        a = np.fromfile(f, dtype="<f4")
        if a.size != 6890 * 3 or not np.isfinite(a).all():
            bad_finite.append(sid)

    missing = len(ids) - len(seen)
    out = {
        "manifest_samples": len(ids),
        "prediction_files": len(files),
        "valid": len(seen) - len(bad_size) - len(bad_finite),
        "missing": missing,
        "bad_size": bad_size,
        "bad_finite": bad_finite,
        "foreign_ids": foreign,
        "expected_bytes": EXPECT,
        "complete": missing == 0 and not bad_size and not bad_finite and not foreign,
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if not isinstance(v, list)}, indent=2))
    if out["bad_size"] or out["bad_finite"] or out["foreign_ids"]:
        print("INVALID: %d bad_size, %d bad_finite, %d foreign"
              % (len(bad_size), len(bad_finite), len(foreign)))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

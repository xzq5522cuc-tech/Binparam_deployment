#!/usr/bin/env python3
"""Cross-set mesh agreement: mean per-sample vertex distance, mm.

Definition used for the Pi 5 report (pi_mesh_delta_full.json):
  per sample: mean over 6890 vertices of ||a - b|| * 1000
  reported:   mean and p95 over samples
It is a backend-agreement number, NOT an error against ground truth.

Usage: python3 mesh_delta.py DIR_A DIR_B [--out FILE]
"""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    a, b = Path(args.a), Path(args.b)
    names = sorted(p.name for p in a.glob("*.pred_vertices.bin"))
    common = [n for n in names if (b / n).exists()]
    if len(common) != len(names):
        print("warning: %d/%d matched" % (len(common), len(names)))
    dist = []
    for n in common:
        va = np.fromfile(a / n, dtype="<f4").reshape(6890, 3)
        vb = np.fromfile(b / n, dtype="<f4").reshape(6890, 3)
        dist.append(float(np.linalg.norm(va - vb, axis=1).mean() * 1000))
    d = np.array(dist)
    out = {"samples": len(d), "mean_mm": float(d.mean()),
           "p95_mm": float(np.percentile(d, 95)),
           "median_mm": float(np.median(d)), "max_mm": float(d.max()),
           "a": str(a), "b": str(b),
           "definition": "mean per-sample vertex distance (mm), not GT error"}
    print(json.dumps(out, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

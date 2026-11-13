#!/usr/bin/env python3
"""Run the full 3DPW test pack through the C4 resident BNN backend.

Runs ON the C4. Writes <out>/<sample_id>.pred_vertices.bin (6890x3 float32,
metres) exactly like the Pi 5 unified run, so the server can score both with
the same program.

Resumable: samples whose output already exists are skipped.
Usage: python3 c4_full_eval.py [--out DIR] [--limit N] [--start ID]
"""
import argparse
import numpy as np
import hashlib
import json
import os
import struct
import subprocess
import sys
import time
from pathlib import Path

R = Path("/home/odroid/pymaf_final/final_release")
PACK = Path("/home/odroid/pymaf_final/dataset/3dpw_full")
CAND = R / "tools/analysis/openmp_candidate"
ENGINE = R / "tools/analysis/four_stage_diagnostic/cap_release_engine"
BNN = R / "versions/c4_20261020_biasfix/models/bnn"
ASSETS = {
    "smpl": R / "source/data/smpl_neutral_f32.bin",
    "dmap": R / "source/data/smpl_dmap_431x6890.bin",
    "mean": R / "source/data/pymaf_mean_params_f32.bin",
    "smpl-onnx": R / "models/smpl_ort/smpl_neutral_opset16.onnx",
}
PAYLOAD = 20899 * 4
VERTS = 6890 * 3 * 4


def readn(f, n):
    buf = bytearray()
    while len(buf) < n:
        part = f.read(n - len(buf))
        if not part:
            raise RuntimeError("incomplete RPC reply")
        buf.extend(part)
    return bytes(buf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path("/home/odroid/eval_c4/predictions")))
    ap.add_argument("--pack", default=str(PACK))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--start", default=None)
    ap.add_argument("--mode", default="bnn", choices=("bnn", "fp32"))
    ap.add_argument("--affinity", default="0-3")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--allow-missing", action="store_true",
                    help="skip samples whose input file is absent or truncated "
                         "(used while the pack is still being copied)")
    args = ap.parse_args()
    if args.threads != 2 or args.affinity != "0-1":
        ap.error("requires Bolt/ORT 2 and CPU 0-1")

    pack = Path(args.pack)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = [l.split() for l in (pack / "manifest.txt").read_text().splitlines() if l.strip()]
    if args.start:
        rows = [r for r in rows if r[0] >= args.start]
    if args.limit:
        rows = rows[:args.limit]

    def _done(sample):
        f = out / (sample + ".pred_vertices.bin")
        return f.exists() and f.stat().st_size == VERTS

    todo = [r for r in rows if not _done(r[0])]
    print("total %d, todo %d" % (len(rows), len(todo)), flush=True)

    env = os.environ.copy()
    env.update(OMP_NUM_THREADS=str(args.threads), OMP_DYNAMIC="FALSE", OMP_WAIT_POLICY="PASSIVE",
               PYMAF_ORT_DISABLE_SPIN="1", PYMAF_CAP_DOWNSAMPLE_THREADS="1",
               BOLT_BNN_ZERO_PADDING_FIX="1", BOLT_BNN_NUMERIC_ZERO_FIX="0")
    for k in ("BOLT_TIMING_FILE", "BOLT_DIAG_DIR", "PYMAF_PHASE_LOG"):
        env.pop(k, None)
    libs = [ENGINE, CAND / "lib_parallel_border"] + sorted(
        {p.parent for p in (CAND / "build").rglob("*.so")})
    env["LD_LIBRARY_PATH"] = ":".join(str(x) for x in libs)

    model_dir = BNN if args.mode == "bnn" else R / "models/fp32"
    cmd = ["taskset", "-c", args.affinity, str(CAND / "rpc_threads")]
    for k, v in ASSETS.items():
        cmd += ["--" + k, str(v)]
    for n in ("feature", "stage0", "stage1", "stage2"):
        cmd += ["--" + n, str(model_dir / (n + ".bolt"))]
    cmd += ["--smpl-threads", str(args.threads)]

    log = (out / "engine.log").open("a")
    proc = subprocess.Popen(cmd, env=env, stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=log)
    try:
        ready = readn(proc.stdout, 24)
        assert struct.unpack_from("<I", ready)[0] == 0x50595244
        print("init_ms %.1f" % struct.unpack_from("<d", ready, 16)[0], flush=True)
        t0 = time.time()
        done = 0
        pending = 0
        for r in todo:
            sample, rel = r[0], r[1]
            path = pack / rel
            if args.allow_missing:
                if not path.exists() or path.stat().st_size != 3 * 224 * 224 * 4:
                    pending += 1
                    continue
            data = path.read_bytes()
            if len(data) != 3 * 224 * 224 * 4:
                raise RuntimeError("bad input size for " + sample)
            proc.stdin.write(data)
            proc.stdin.flush()
            head = readn(proc.stdout, 104)
            assert struct.unpack_from("<I", head)[0] == 0x50595253
            payload = readn(proc.stdout, PAYLOAD)
            verts = payload[-VERTS:]
            if not np.isfinite(np.frombuffer(verts, dtype="<f4")).all():
                raise RuntimeError("nonfinite vertices for " + sample)
            dst = out / (sample + ".pred_vertices.bin")
            tmp = dst.with_suffix(".tmp")
            tmp.write_bytes(verts)
            tmp.replace(dst)
            done += 1
            if done % 250 == 0:
                el = time.time() - t0
                print("%d/%d  %.1f ms/frame  %.0f min left"
                      % (done, len(todo), 1000 * el / done,
                         (len(todo) - done) * el / done / 60), flush=True)
        proc.stdin.close()
        if proc.wait(timeout=60) != 0:
            raise RuntimeError("backend exited with an error")
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        log.close()

    stats = {
        "mode": args.mode,
        "completed": done,
        "pending_missing": pending,
        "start": todo[0][0] if todo else None,
        "end": todo[-1][0] if todo else None,
        "elapsed_s": time.time() - t0 if todo else 0,
        "models": {n: hashlib.sha256((model_dir / (n + ".bolt")).read_bytes()).hexdigest()
                   for n in ("feature", "stage0", "stage1", "stage2")},
    }
    (out / ("run_%s.json" % args.mode)).write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats), flush=True)


if __name__ == "__main__":
    main()

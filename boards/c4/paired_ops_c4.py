#!/usr/bin/env python3
"""Paired-operator run for the C4 using the instrumented trace engine.

Runs ON the C4. For each precision (fp32/fp16/bnn) and 3 rounds it performs
45 inferences of the same fixed input with BOLT_TIMING_FILE set, so the engine
appends one row per operator per run. The engine logs from run 5 onward, which
gives exactly 40 logged runs; the analyzer then drops the first 10 of those and
keeps 30 measured runs per round (90 per config), matching the Pi 5 run.

Threads: configurable, defaults to Bolt 2 / ORT 2, affinity 0-1, ORT spin off,
downsample cap 1, BNN border fix on.

Usage: python3 paired_ops_c4.py --models models_c4.json --out /home/odroid/eval_c4/paired
"""
import argparse
import hashlib
import json
import os
import struct
import subprocess
import time
from pathlib import Path

import numpy as np

R = Path("/home/odroid/pymaf_final/final_release")
CAND = R / "tools/analysis/openmp_candidate"
ENGINE = R / "tools/analysis/four_stage_diagnostic/cap_release_engine"
TRACE = R / "tools/analysis/four_stage_diagnostic/trace_engine"
PACK = R.parent / "dataset/3dpw_board_200"
ASSETS = {
    "smpl": R / "source/data/smpl_neutral_f32.bin",
    "dmap": R / "source/data/smpl_dmap_431x6890.bin",
    "mean": R / "source/data/pymaf_mean_params_f32.bin",
    "smpl-onnx": R / "models/smpl_ort/smpl_neutral_opset16.onnx",
}
ITERATIONS = 45
PAYLOAD = 20899 * 4


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
    ap.add_argument("--models", required=True)
    ap.add_argument("--out", default="/home/odroid/eval_c4/paired")
    ap.add_argument("--only", default="fp32,fp16,bnn")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--affinity", default="0-1")
    args = ap.parse_args()
    assert args.threads == 2 and args.affinity == "0-1"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    models = json.loads(Path(args.models).read_text())
    wanted = [c.strip() for c in args.only.split(",") if c.strip()]

    rows = (PACK / "manifest.txt").read_text().splitlines()
    data = (PACK / rows[0].split()[1]).read_bytes()

    env = os.environ.copy()
    env.update(OMP_NUM_THREADS=str(args.threads), OMP_DYNAMIC="FALSE",
               OMP_WAIT_POLICY="PASSIVE",
               PYMAF_ORT_DISABLE_SPIN="1", PYMAF_CAP_DOWNSAMPLE_THREADS="1",
               BOLT_BNN_ZERO_PADDING_FIX="1", BOLT_BNN_NUMERIC_ZERO_FIX="0")
    for k in ("PYMAF_PHASE_LOG", "BOLT_DIAG_DIR"):
        env.pop(k, None)
    libs = [TRACE, ENGINE, CAND / "lib_parallel_border"] + sorted(
        {p.parent for p in (CAND / "build").rglob("*.so")})
    env["LD_LIBRARY_PATH"] = ":".join(str(x) for x in libs)

    runs = []
    for rep in (1, 2, 3):
        for cfg in wanted:
            paths = {k: Path(v) for k, v in models[cfg].items()}
            cmd = ["taskset", "-c", args.affinity, str(CAND / "rpc_threads")]
            for k, v in ASSETS.items():
                cmd += ["--" + k, str(v)]
            for n in ("feature", "stage0", "stage1", "stage2"):
                cmd += ["--" + n, str(paths[n])]
            cmd += ["--smpl-threads", str(args.threads)]

            timing_file = out / ("%s%d.log" % (cfg, rep))
            if timing_file.exists():
                timing_file.unlink()
            run_env = dict(env, BOLT_TIMING_FILE=str(timing_file))
            log = (out / ("%s%d_stderr.log" % (cfg, rep))).open("w")
            timings, hashes = [], []
            proc = subprocess.Popen(cmd, env=run_env, stdin=subprocess.PIPE,
                                    stdout=subprocess.PIPE, stderr=log)
            try:
                ready = readn(proc.stdout, 24)
                assert struct.unpack_from("<I", ready)[0] == 0x50595244
                maps = (Path("/proc") / str(proc.pid) / "maps").read_text()
                assert str(TRACE) in maps, "trace engine not loaded"
                (out / ("%s%d_maps.txt" % (cfg, rep))).write_text(maps)
                for i in range(ITERATIONS):
                    proc.stdin.write(data)
                    proc.stdin.flush()
                    head = readn(proc.stdout, 104)
                    assert struct.unpack_from("<I", head)[0] == 0x50595253
                    payload = readn(proc.stdout, PAYLOAD)
                    assert np.isfinite(np.frombuffer(payload, dtype="<f4")).all()
                    hashes.append(hashlib.sha256(payload).hexdigest())
                    timings.append(struct.unpack_from("<11d", head, 16))
                status = (Path("/proc") / str(proc.pid) / "status").read_text()
                peak = next(int(x.split()[1]) / 1024 for x in status.splitlines()
                            if x.startswith("VmHWM:"))
                proc.stdin.close()
                assert proc.wait(timeout=30) == 0
            finally:
                if proc.poll() is None:
                    proc.kill()
                    proc.wait()
                log.close()

            n_lines = sum(1 for _ in timing_file.open())
            runs.append({
                "config": cfg, "round": rep, "iterations": ITERATIONS,
                "logged_operator_rows": n_lines,
                "timings_ms": timings,
                "peak_rss_mib": peak,
                "unique_output_hashes": len(set(hashes[10:])),
                "temperature_c": float(Path("/sys/class/thermal/thermal_zone0/temp")
                                       .read_text()) / 1000,
            })
            (out / "runs.json").write_text(json.dumps(runs, indent=2))
            print("%s round %d: %d operator rows, mean %.2f ms, unique hashes %d"
                  % (cfg, rep, n_lines, float(np.mean([t[0] for t in timings[10:]])),
                     runs[-1]["unique_output_hashes"]), flush=True)

    (out / "models.json").write_text(json.dumps(
        {cfg: {n: {"path": str(Path(v)), "bytes": Path(v).stat().st_size,
                   "sha256": hashlib.sha256(Path(v).read_bytes()).hexdigest()}
               for n, v in models[cfg].items()} for cfg in wanted}, indent=2))
    (out / "protocol.json").write_text(json.dumps({
        "iterations": ITERATIONS,
        "engine_logs_from_run": 5,
        "logged_runs": ITERATIONS - 5,
        "measured_runs_per_round": 30,
        "rounds": 3,
        "threads_bolt": 2,
        "threads_ort": args.threads,
        "affinity": args.affinity,
        "input_sha256": hashlib.sha256(data).hexdigest(),
        "engine": str(TRACE),
        "boundary": "op->run() around each operator; includes internal packing, "
                    "reordering and the border repair; excludes standalone "
                    "Scale/PReLU/Gather/Hadamard nodes",
        "environment": {k: v for k, v in env.items()
                        if k.startswith(("OMP_", "PYMAF_", "BOLT_"))},
    }, indent=2))
    print("done")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""C4 full-backend benchmark, same protocol as the Pi 5 official run.

Protocol (see protocol.json written next to the results):
  3 rounds, order fp32/fp16/bnn -> fp16/bnn/fp32 -> bnn/fp32/fp16
  each: 5 warmup + 30 measured, batch=1, fixed input, resident process
  timing = resident inference from preprocessed tensor to final mesh
  threads: configurable for Bolt, ORT and OpenMP (default 2)
  configurable affinity (default 0-1), ORT spin disabled, downsample cap 1, BNN border fix on
  90 samples merged per scheme, RSS = max process peak, model size = 4 bolts

Runs ON the C4. Usage:
  python3 benchmark_c4.py --models models.json [--only fp32,bnn]
"""
import argparse
import csv
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
PACK = R.parent / "dataset/3dpw_board_200"
ASSETS = {
    "smpl": R / "source/data/smpl_neutral_f32.bin",
    "dmap": R / "source/data/smpl_dmap_431x6890.bin",
    "mean": R / "source/data/pymaf_mean_params_f32.bin",
    "smpl-onnx": R / "models/smpl_ort/smpl_neutral_opset16.onnx",
}
ORDER = [("fp32", "fp16", "bnn"), ("fp16", "bnn", "fp32"), ("bnn", "fp32", "fp16")]
WARMUP, LOOPS = 5, 30
PAYLOAD = 20899 * 4


def readn(f, n):
    buf = bytearray()
    while len(buf) < n:
        part = f.read(n - len(buf))
        if not part:
            raise RuntimeError("incomplete RPC reply")
        buf.extend(part)
    return bytes(buf)


def sysinfo():
    out = {}
    for name, path in (("temp_c", "/sys/class/thermal/thermal_zone0/temp"),
                       ("governor", "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")):
        try:
            val = Path(path).read_text().strip()
            out[name] = float(val) / 1000 if name == "temp_c" else val
        except Exception:
            out[name] = None
    freqs = []
    for cpu in range(4):
        try:
            freqs.append(int(Path("/sys/devices/system/cpu/cpu%d/cpufreq/scaling_cur_freq"
                                  % cpu).read_text()) // 1000)
        except Exception:
            pass
    out["cpu_mhz"] = freqs
    out["loadavg"] = Path("/proc/loadavg").read_text().split()[:3]
    out["service_active"] = subprocess.run(
        ["systemctl", "--user", "is-active", "--quiet", "pymaf-camera.service"]).returncode == 0
    return out


def run_scheme(cfg, paths, data, out_dir, threads, affinity):
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS=str(threads), OMP_DYNAMIC="FALSE", OMP_WAIT_POLICY="PASSIVE",
               PYMAF_ORT_DISABLE_SPIN="1", PYMAF_CAP_DOWNSAMPLE_THREADS="1",
               BOLT_BNN_ZERO_PADDING_FIX="1", BOLT_BNN_NUMERIC_ZERO_FIX="0")
    for k in ("BOLT_TIMING_FILE", "BOLT_DIAG_DIR", "PYMAF_PHASE_LOG"):
        env.pop(k, None)
    libs = [ENGINE, CAND / "lib_parallel_border"] + sorted(
        {p.parent for p in (CAND / "build").rglob("*.so")})
    env["LD_LIBRARY_PATH"] = ":".join(str(x) for x in libs)

    cmd = ["taskset", "-c", affinity, str(CAND / "rpc_threads")]
    for k, v in ASSETS.items():
        cmd += ["--" + k, str(v)]
    for n in ("feature", "stage0", "stage1", "stage2"):
        cmd += ["--" + n, str(paths[n])]
    cmd += ["--smpl-threads", str(threads)]

    before = sysinfo()
    timings, hashes, stages = [], [], []
    first_verts = None
    max_delta_m = 0.0
    log = (out_dir / ("%s.log" % cfg)).open("w")
    proc = subprocess.Popen(cmd, env=env, stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=log)
    try:
        ready = readn(proc.stdout, 24)
        assert struct.unpack_from("<I", ready)[0] == 0x50595244
        init_ms = struct.unpack_from("<d", ready, 16)[0]
        for i in range(WARMUP + LOOPS):
            proc.stdin.write(data)
            proc.stdin.flush()
            head = readn(proc.stdout, 104)
            assert struct.unpack_from("<I", head)[0] == 0x50595253
            payload = readn(proc.stdout, PAYLOAD)
            verts = np.frombuffer(payload[-6890 * 3 * 4:], dtype="<f4")
            if not np.isfinite(np.frombuffer(payload, dtype="<f4")).all():
                raise RuntimeError("non-finite output for " + cfg)
            hashes.append(hashlib.sha256(payload).hexdigest())
            if i >= WARMUP:
                if first_verts is None:
                    first_verts = verts.copy()
                else:
                    max_delta_m = max(max_delta_m,
                                      float(np.abs(verts - first_verts).max()))
            t = struct.unpack_from("<11d", head, 16)
            if i >= WARMUP:
                timings.append(t[0])
                stages.append(list(t))
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
    unique = len(set(hashes[WARMUP:]))
    if max_delta_m > 0.05:
        raise RuntimeError("gross output drift %.3f m for %s" % (max_delta_m, cfg))
    return {"config": cfg, "init_ms": init_ms, "timings_ms": timings,
            "stage_timings_ms": stages,
            "peak_rss_mib": peak, "output_sha256": hashes[-1],
            "unique_output_hashes": unique,
            "max_abs_vertex_delta_m": max_delta_m,
            "environment": before, "finished_at": time.strftime("%F %T")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True, help="json: {cfg: {feature,stage0,stage1,stage2}}")
    ap.add_argument("--out", default="/home/odroid/eval_c4/perf")
    ap.add_argument("--only", default="fp32,fp16,bnn")
    ap.add_argument("--threads", type=int, choices=range(1, 5), default=2)
    ap.add_argument("--affinity", default="0-1")
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    models = json.loads(Path(args.models).read_text())
    wanted = [c.strip() for c in args.only.split(",") if c.strip()]
    missing = [c for c in wanted if c not in models]
    if missing:
        raise SystemExit("models.json lacks: %s" % missing)

    rows = (PACK / "manifest.txt").read_text().splitlines()
    data = (PACK / rows[0].split()[1]).read_bytes()
    assert len(data) == 602112

    runs = []
    for rep, order in enumerate(ORDER, 1):
        for cfg in order:
            if cfg not in wanted:
                continue
            paths = {k: Path(v) for k, v in models[cfg].items()}
            r = run_scheme(cfg, paths, data, out_dir, args.threads, args.affinity)
            r["round"] = rep
            runs.append(r)
            (out_dir / "runs.json").write_text(json.dumps(runs, indent=2))
            print("round %d %-5s mean %.2f ms" % (rep, cfg, np.mean(r["timings_ms"])),
                  flush=True)

    summary = {}
    for cfg in wanted:
        rr = [r for r in runs if r["config"] == cfg]
        ts = [t for r in rr for t in r["timings_ms"]]
        model_bytes = sum(Path(models[cfg][n]).stat().st_size
                          for n in ("feature", "stage0", "stage1", "stage2"))
        summary[cfg] = {
            "n": len(ts),
            "mean_ms": float(np.mean(ts)),
            "p95_ms": float(np.percentile(ts, 95)),
            "min_ms": float(np.min(ts)),
            "max_ms": float(np.max(ts)),
            "model_mib": model_bytes / 1024 ** 2,
            "peak_rss_mib": max(r["peak_rss_mib"] for r in rr),
            "output_sha256": rr[0]["output_sha256"],
            "unique_output_hashes": max(r["unique_output_hashes"] for r in rr),
            "max_abs_vertex_delta_mm": 1000 * max(r["max_abs_vertex_delta_m"] for r in rr),
        }
        st = np.array([s for r in rr for s in r["stage_timings_ms"]], dtype=float)
        names = ["total", "feature", "stage0", "stage1", "stage2", "rotation6d",
                 "smpl_feedback", "dmap", "projection", "grid_sample", "final_smpl"]
        means = {n: float(st[:, i].mean()) for i, n in enumerate(names)}
        named = sum(means[n] for n in names[1:])
        means["residual"] = means["total"] - named
        means["geometry_ops"] = sum(means[n] for n in
                                    ("rotation6d", "smpl_feedback", "dmap",
                                     "projection", "grid_sample"))
        means["stages_sum"] = means["stage0"] + means["stage1"] + means["stage2"]
        summary[cfg]["modules_ms"] = means
    for cfg, v in summary.items():
        v["speedup_vs_fp32"] = summary["fp32"]["mean_ms"] / v["mean_ms"] if "fp32" in summary else None

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    with (out_dir / "summary.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["config", "n", "mean_ms", "p95_ms", "model_mib", "peak_rss_mib",
                    "speedup_vs_fp32"])
        for cfg, v in summary.items():
            w.writerow([cfg, v["n"], "%.3f" % v["mean_ms"], "%.3f" % v["p95_ms"],
                        "%.3f" % v["model_mib"], "%.1f" % v["peak_rss_mib"],
                        "" if v["speedup_vs_fp32"] is None else "%.3f" % v["speedup_vs_fp32"]])
    (out_dir / "protocol.json").write_text(json.dumps({
        "warmup": WARMUP, "loops": LOOPS, "rounds": 3,
        "round_order": [list(o) for o in ORDER],
        "threads_bolt": args.threads,
        "threads_ort": args.threads, "affinity": args.affinity,
        "input_sha256": hashlib.sha256(data).hexdigest(),
        "environment": {
            "OMP_NUM_THREADS": str(args.threads), "OMP_DYNAMIC": "FALSE",
            "OMP_WAIT_POLICY": "PASSIVE", "PYMAF_ORT_DISABLE_SPIN": "1",
            "PYMAF_CAP_DOWNSAMPLE_THREADS": "1",
            "BOLT_BNN_ZERO_PADDING_FIX": "1", "BOLT_BNN_NUMERIC_ZERO_FIX": "0"},
        "boundary": "resident inference from preprocessed tensor to final mesh; "
                    "excludes load, preprocessing, display, I/O",
        "size_scope": "four Bolt models only",
        "models": {cfg: {n: {"path": models[cfg][n],
                             "bytes": Path(models[cfg][n]).stat().st_size,
                             "sha256": hashlib.sha256(Path(models[cfg][n]).read_bytes()).hexdigest()}
                        for n in ("feature", "stage0", "stage1", "stage2")}
                   for cfg in wanted},
        "system": sysinfo(),
    }, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

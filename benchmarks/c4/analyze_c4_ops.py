#!/usr/bin/env python3
"""Pair the 54 binary convolutions with their floating-point counterparts from
per-operator logs, and build the operator table.

Same statistics as the Pi 5 report: per configuration 3 rounds x (10 warmup +
30 measured); per iteration the paired operator times are summed first, then
averaged over 90 iterations. Speedup = float cumulative mean / BNN cumulative
mean (not the mean of per-layer speedups).

The 54 targets are the op names discovered on the Pi 5 run of the same graph
(the .bolt files are hash-identical), passed with --targets-from. A dispatch
probe log can be used instead via --dispatch.

Usage:
  python3 analyze_c4_ops.py --targets-from pi5_all54_summary.json \
      --logs DIR --threads 1 --out all54_c4.json --markdown all54_c4.md
"""
import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

SCOPES = ("feature48", "stage6", "all54")
FLOAT_ALIAS = {"/Conv": "/fc1/Gemm", "/Conv_1": "/fc2/Gemm"}
CONFIGS = []


def parse_audit(path):
    """Return {(pointer index, op name): [ms, ...]} in file order.

    Accepts both log formats:
      AUDIT\t<ptr>\t<name>\t<type>\t<ms>       (audit build, Pi 5)
      <ptr>\t<run>\t<name>\t<type>\t<ms>        (BOLT_TIMING_FILE, C4)
    """
    ptrs, rows = [], defaultdict(list)
    for line in Path(path).read_text(errors="replace").splitlines():
        f = line.split("\t")
        if line.startswith("AUDIT\t") and len(f) == 5:
            _, ptr, name, typ, ms = f
        elif len(f) == 5:
            ptr, _run, name, typ, ms = f
        else:
            continue
        if ptr not in ptrs:
            ptrs.append(ptr)
        rows[(ptrs.index(ptr), name)].append(float(ms))
    return rows


def discover_targets(path, kernel):
    ptrs, pending, targets = [], False, set()
    for line in Path(path).read_text(errors="replace").splitlines():
        if line.startswith("BNN_CALL\t"):
            f = line.split("\t")
            assert f[1] == "DT_BIN11", f
            assert f[2] == kernel, "kernel mismatch: %s" % f[2]
            assert f[-1] == "0", "kernel call failed: %s" % line
            pending = True
        elif line.startswith("AUDIT\t"):
            _, ptr, name, _typ, _ms = line.split("\t")
            if ptr not in ptrs:
                ptrs.append(ptr)
            if pending:
                targets.add((ptrs.index(ptr), name))
                pending = False
    return targets


def targets_from_json(path):
    data = json.loads(Path(path).read_text())
    details = data["details"] if "details" in data else data
    return {(d["module"], d["binary_op"]) for d in details}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dispatch", default=None)
    ap.add_argument("--targets-from", default=None)
    ap.add_argument("--threads", default="1,4")
    ap.add_argument("--logs", required=True)
    ap.add_argument("--kernel", default="ARM_A55")
    ap.add_argument("--out", default="all54_summary.json")
    ap.add_argument("--markdown", default=None)
    args = ap.parse_args()

    threads = [t.strip() for t in args.threads.split(",") if t.strip()]
    CONFIGS[:] = ["%s_%s" % (d, t) for t in threads for d in ("fp32", "fp16", "bnn")]

    if args.targets_from:
        targets = targets_from_json(args.targets_from)
        print("targets loaded from %s" % args.targets_from)
    else:
        if not args.dispatch:
            raise SystemExit("need --dispatch or --targets-from")
        targets = discover_targets(args.dispatch, args.kernel)
    assert len(targets) == 54, "expected 54 binary convs, found %d" % len(targets)
    targets = sorted(targets)
    print("using %d binary convs (%d feature, %d stage)"
          % (len(targets), sum(1 for m, _ in targets if m == 0),
             sum(1 for m, _ in targets if m > 0)))

    logs = Path(args.logs)
    observations, details, summary = {}, [], {}
    for config in CONFIGS:
        observations[config] = defaultdict(list)
        for rep in (1, 2, 3):
            rows = parse_audit(logs / ("%s%d.log" % (config, rep)))
            for module, name in targets:
                mapped = name if module == 0 or config.startswith("bnn") else FLOAT_ALIAS[name]
                values = rows[(module, mapped)]
                assert len(values) == 40, (config, rep, module, mapped, len(values))
                observations[config][(module, name)] += values[10:]
        summary[config] = {}
        for scope in SCOPES:
            keys = [k for k in targets
                    if scope == "all54"
                    or (scope == "feature48" and k[0] == 0)
                    or (scope == "stage6" and k[0] > 0)]
            totals = [sum(observations[config][k][i] for k in keys) for i in range(90)]
            summary[config][scope] = {
                "count": len(keys), "n": 90,
                "mean_ms": statistics.mean(totals),
                "round_means_ms": [statistics.mean(totals[i:i + 30]) for i in (0, 30, 60)],
            }

    for module, name in targets:
        means = {c: statistics.mean(observations[c][(module, name)]) for c in CONFIGS}
        details.append({
            "module": module, "binary_op": name,
            "float_op": name if module == 0 else FLOAT_ALIAS[name],
            "means_ms": means,
            "speedups": {"%s_t%s" % (d, t): means["%s_%s" % (d, t)] / means["bnn_%s" % t]
                         for t in threads for d in ("fp32", "fp16")},
        })
    ratios = {
        scope: {"%s_t%s" % (d, t):
                summary["%s_%s" % (d, t)][scope]["mean_ms"] / summary["bnn_%s" % t][scope]["mean_ms"]
                for t in threads for d in ("fp32", "fp16")}
        for scope in SCOPES}

    output = {
        "scope": "54 binary Conv operators vs 48 floating Conv plus 6 corresponding "
                 "FC/Gemm operators. Excludes separate transforms, Scale, Gather, "
                 "Hadamard and other auxiliary operators; not an end-to-end figure.",
        "kernel": args.kernel, "threads": threads,
        "summary": summary, "speedups": ratios, "details": details,
    }
    Path(args.out).write_text(json.dumps(output, indent=2))

    if args.markdown:
        names = {"feature48": "主干 48 卷积",
                 "stage6": "回归 6 个 FC/Gemm 对应卷积",
                 "all54": "54 目标算子小计"}
        lines = ["# C4 二值算子配对统计", "",
                 "每配置 3 轮，每轮预热 10 次、正式 30 次；每次推理先累加配对算子，再对 90 次求均值。", "",
                 "| 统计范围 | Bolt 线程 | FP32/ms | FP16/ms | BNN/ms | 相对 FP32 | 相对 FP16 |",
                 "|---|---:|---:|---:|---:|---:|---:|"]
        for scope in SCOPES:
            for t in threads:
                vals = [summary["%s_%s" % (d, t)][scope]["mean_ms"]
                        for d in ("fp32", "fp16", "bnn")]
                lines.append("| %s | %s | %s | %.2f× | %.2f× |"
                             % (names[scope], t, " | ".join("%.3f" % v for v in vals),
                                ratios[scope]["fp32_t%s" % t],
                                ratios[scope]["fp16_t%s" % t]))
        Path(args.markdown).write_text("\n".join(lines) + "\n", encoding="utf-8")

    for s in SCOPES:
        for t in threads:
            print("%-9s t%s  FP32/BNN %.2fx   FP16/BNN %.2fx"
                  % (s, t, ratios[s]["fp32_t%s" % t], ratios[s]["fp16_t%s" % t]))
    for c in CONFIGS:
        print(c, {s: round(summary[c][s]["mean_ms"], 3) for s in SCOPES})


if __name__ == "__main__":
    main()

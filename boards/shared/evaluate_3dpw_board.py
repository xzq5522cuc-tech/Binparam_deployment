#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np


def similarity_transform(source, target):
    source = source.T
    target = target.T
    mu1 = source.mean(axis=1, keepdims=True)
    mu2 = target.mean(axis=1, keepdims=True)
    x1, x2 = source - mu1, target - mu2
    variance = np.sum(x1 ** 2)
    covariance = x1 @ x2.T
    u, singular, vh = np.linalg.svd(covariance)
    rotation = vh.T @ u.T
    if np.linalg.det(rotation) < 0:
        vh[-1, :] *= -1
        rotation = vh.T @ u.T
    scale = np.trace(rotation @ covariance) / variance
    translation = mu2 - scale * rotation @ mu1
    return (scale * rotation @ source + translation).T


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pack", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    pack, predictions = Path(args.pack), Path(args.predictions)
    regressor = np.fromfile(pack / "J_regressor_h36m_f32.bin", dtype="<f4").reshape(17, 6890)
    mapping = np.fromfile(pack / "H36M_to_J14_i32.bin", dtype="<i4")

    pves, mpjpes, pa_mpjpes, per_sample = [], [], [], []
    for line in (pack / "manifest.txt").read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        sample_id, _, gt_rel = line.split()[:3]
        gt_vertices = np.fromfile(pack / gt_rel, dtype="<f4").reshape(6890, 3)
        pred_path = predictions / f"{sample_id}.pred_vertices.bin"
        pred_vertices = np.fromfile(pred_path, dtype="<f4").reshape(6890, 3)
        pve = np.linalg.norm(pred_vertices - gt_vertices, axis=1).mean()
        gt_all = regressor @ gt_vertices
        pred_all = regressor @ pred_vertices
        gt_joints = gt_all[mapping] - gt_all[[0]]
        pred_joints = pred_all[mapping] - pred_all[[0]]
        mpjpe = np.linalg.norm(pred_joints - gt_joints, axis=1).mean()
        pred_pa = similarity_transform(pred_joints, gt_joints)
        pa_mpjpe = np.linalg.norm(pred_pa - gt_joints, axis=1).mean()
        pves.append(pve); mpjpes.append(mpjpe); pa_mpjpes.append(pa_mpjpe)
        per_sample.append({"sample_id": sample_id, "pve_mm": float(pve * 1000),
                           "mpjpe_mm": float(mpjpe * 1000),
                           "pa_mpjpe_mm": float(pa_mpjpe * 1000)})

    result = {
        "samples": len(pves),
        "pve_mm": float(np.mean(pves) * 1000),
        "mpjpe_mm": float(np.mean(mpjpes) * 1000),
        "pa_mpjpe_mm": float(np.mean(pa_mpjpes) * 1000),
        "per_sample": per_sample,
    }
    Path(args.output).write_text(json.dumps(result, indent=2))
    print("*** ODROID 3DPW Results ***")
    print(f"Samples: {result['samples']}")
    print(f"PVE: {result['pve_mm']:.6f} mm")
    print(f"MPJPE: {result['mpjpe_mm']:.6f} mm")
    print(f"PA-MPJPE: {result['pa_mpjpe_mm']:.6f} mm")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Convert four already-exported PyMAF ONNX subgraphs with X2bolt."""
import argparse
import hashlib
from pathlib import Path
import shutil
import subprocess
import tempfile

MODE_SUFFIX = {"FP32": "f32", "FP16": "f16", "BNN": "f16_b"}
MODEL_NAMES = ("feature", "stage0", "stage1", "stage2")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--converter", required=True, type=Path)
    parser.add_argument("--onnx-dir", required=True, type=Path)
    parser.add_argument("--mode", required=True, choices=MODE_SUFFIX)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    converter = args.converter.resolve(strict=True)
    output = args.out.resolve()
    if output.exists():
        parser.error(f"Output already exists: {output}")

    inputs = {}
    for name in MODEL_NAMES:
        stem = "feature_pyramid" if name == "feature" else name
        candidates = [args.onnx_dir / f"{name}.onnx",
                      args.onnx_dir / f"{stem}_opset16.onnx"]
        matches = [path for path in candidates if path.is_file()]
        if len(matches) != 1:
            parser.error(f"Expected exactly one input for {name}: {candidates}")
        inputs[name] = matches[0].resolve()

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".x2bolt-", dir=output.parent) as temp:
        staging = Path(temp)
        release = staging / "release"
        release.mkdir()
        checksums = []
        for name, source in inputs.items():
            # X2bolt takes a directory and basename without the extension.
            shutil.copyfile(source, staging / f"{name}.onnx")
            log_path = release / f"{name}.log"
            command = [str(converter), "-d", str(staging), "-m", name, "-i", args.mode]
            with log_path.open("wb") as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
            model = staging / f"{name}_{MODE_SUFFIX[args.mode]}.bolt"
            if result.returncode or not model.is_file() or model.stat().st_size == 0:
                tail = log_path.read_text(errors="replace").splitlines()[-20:]
                raise SystemExit(f"Conversion failed for {name} ({args.mode})\n" + "\n".join(tail))
            model.rename(release / model.name)
            with (release / model.name).open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            checksums.append(f"{digest}  {model.name}\n")
        (release / "SHA256SUMS").write_text("".join(checksums), encoding="ascii")
        release.rename(output)
    print(output)


if __name__ == "__main__":
    main()

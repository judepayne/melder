#!/usr/bin/env python3
"""Export a fine-tuned sentence-transformers model to ONNX for use with melder.

Uses optimum-cli to export model.onnx into the same directory as the
HuggingFace model files. After export the directory contains everything
melder's encoder needs: model.onnx + tokenizer files.

Usage:
    python export.py --model-dir models/round_1
"""

import argparse
import os
import subprocess
import sys


def export_onnx(model_dir: str) -> str:
    """Export model_dir to ONNX in-place. Returns path to model.onnx.

    The ONNX file is written into model_dir alongside the existing
    tokenizer files, producing the layout expected by melder's local-path
    encoder loader.
    """
    onnx_path = os.path.join(model_dir, "model.onnx")

    # optimum-cli export onnx --model <dir> --task feature-extraction <out_dir>
    # Writing to the same dir keeps model.onnx next to tokenizer.json etc.
    cmd = [
        "optimum-cli", "export", "onnx",
        "--model", model_dir,
        "--task",  "feature-extraction",
        model_dir,
    ]

    print(f"  Exporting ONNX: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)

    if result.returncode != 0:
        raise RuntimeError(
            f"optimum-cli export failed (code {result.returncode}). "
            "Ensure 'optimum[exporters]' is installed: pip install 'optimum[exporters]'"
        )

    if not os.path.exists(onnx_path):
        raise FileNotFoundError(
            f"Export succeeded but {onnx_path} was not created. "
            "Check optimum-cli output above."
        )

    size_mb = os.path.getsize(onnx_path) / 1_048_576
    print(f"  Exported → {onnx_path}  ({size_mb:.1f} MB)")
    return onnx_path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", required=True,
                   help="Directory containing the fine-tuned HuggingFace model")
    args = p.parse_args()

    export_onnx(args.model_dir)


if __name__ == "__main__":
    main()

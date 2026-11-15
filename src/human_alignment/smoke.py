"""Run setup checks; no benchmark accuracy claims are made."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from .models import FAMILIES, VARIANTS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=[*FAMILIES, "all"], default="all")
    parser.add_argument("--variant", choices=[*VARIANTS, "all"], default="all")
    parser.add_argument("--cache-dir", default="models")
    parser.add_argument("--download", action="store_true", help="Download missing official checkpoints")
    parser.add_argument("--images", nargs="+", help="Optional local images; defaults to generated patterns")
    parser.add_argument("--output-dir", default="outputs/smoke")
    args = parser.parse_args()
    families = list(FAMILIES) if args.family == "all" else [args.family]
    variants = list(VARIANTS) if args.variant == "all" else [args.variant]
    if len(families) * len(variants) > 1:
        failed = []
        for family in families:
            for variant in variants:
                command = [sys.executable, "-m", "human_alignment.smoke", "--family", family,
                           "--variant", variant, "--cache-dir", args.cache_dir,
                           "--output-dir", args.output_dir]
                if args.download:
                    command.append("--download")
                if args.images:
                    command.extend(["--images", *args.images])
                if subprocess.run(command, check=False).returncode:
                    failed.append(f"{family}/{variant}")
        if failed:
            raise SystemExit(f"Failed: {', '.join(failed)}")
        return
    run_one(args)


def run_one(args):
    import numpy as np
    import tensorflow as tf
    from .models import VisionModel, load_images

    for device in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(device, True)
    model = VisionModel(args.family, args.variant, args.cache_dir, download=args.download)
    shape = model.input_spec.shape.as_list()
    height, width = shape[1] or 224, shape[2] or 224
    if args.images:
        images = load_images(args.images, size=(width, height))
        labels = list(args.images)
    else:
        images = np.zeros((2, height, width, 3), dtype=np.float32)
        images[0, height//4:3*height//4, width//4:3*width//4, 0] = 1
        yy, xx = np.ogrid[:height, :width]
        images[1, (yy-height//2)**2 + (xx-width//2)**2 < (min(height, width)//4)**2, 2] = 1
        labels = ["red_square", "blue_circle"]
    # Add an exact duplicate to check consistent outputs within the same batch.
    batch = np.concatenate([images, images[:1]], axis=0)
    predictions = model.predict(batch)
    for key, value in predictions.items():
        if value.shape[0] != len(batch) or not np.isfinite(value).all():
            raise RuntimeError(f"Invalid output {key}: {value.shape}")
    similarities = {}
    for key in ("pre_logits", "triplet_logits"):
        if key not in predictions:
            raise RuntimeError(f"Missing documented output: {key}")
        values = predictions[key]
        if values.ndim != 2 or np.any(np.linalg.norm(values, axis=1) == 0):
            raise RuntimeError(f"Invalid embeddings: {key}")
        np.testing.assert_allclose(values[0], values[-1], rtol=1e-4, atol=1e-4)
        normalized = values / np.linalg.norm(values, axis=1, keepdims=True)
        similarities[key] = (normalized @ normalized.T).tolist()
    repeated = model.predict(batch)
    for key in ("pre_logits", "triplet_logits"):
        np.testing.assert_allclose(predictions[key], repeated[key], rtol=1e-4, atol=1e-4)
    report = {
        "model": model.name, "tensorflow": tf.__version__,
        "devices": [device.name for device in tf.config.list_logical_devices()],
        "checkpoint": str(model.path), "input_signature": str(model.input_spec),
        "inputs": [*labels, "duplicate_of_first"],
        "output_shapes": {key: list(value.shape) for key, value in predictions.items()},
        "cosine_similarity": similarities, "status": "passed",
    }
    directory = Path(args.output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{model.name}.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(directory / f"{model.name}.npz",
                        pre_logits=predictions["pre_logits"], triplet_logits=predictions["triplet_logits"])
    print(f"PASS {model.name}: {path}", flush=True)


if __name__ == "__main__":
    main()

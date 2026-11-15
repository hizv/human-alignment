# Human alignment: checkpoint setup

Minimal inference code for the official DINOv2-B and SigLIP-B exports, each in Base, AligNet, and UnAligNet variants. All six use the publisher's TensorFlow SavedModels.

## Install

Use Python 3.12 or 3.13:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Optional NVIDIA GPU dependencies on Linux: `python -m pip install -e '.[gpu]'`.

## Smoke tests

Start with one checkpoint:

```bash
alignment-smoke --family siglip --variant alignet --download
```

Run all six:

```bash
alignment-smoke --download
```

Use your own images:

```bash
alignment-smoke --family dino --variant base --download --images cat.jpg dog.jpg
```

Missing models download only with `--download`. Downloads are cached under `models/`; change this with `--cache-dir /path/to/checkpoints`. Subsequent runs can omit `--download`. The runner isolates each checkpoint in a separate process to release model memory between runs. Each archive must be downloaded and unpacked, so allow space for both during extraction.

Default samples are a generated red square and blue circle. Checks cover loading, finite outputs, nonzero embeddings, agreement of duplicate images, and repeatable inference. Different sample images have no expected semantic ordering. JSON reports contain output shapes, devices, and similarity matrices; NPZ files contain embeddings. Both go under `outputs/smoke/`.

## Python API

```python
from human_alignment.models import VisionModel, load_images

model = VisionModel("dino", "unalignet", download=True)
images = load_images(["cat.jpg", "dog.jpg"])
features = model.embed(images)  # pre_logits by default
triplet_features = model.embed(images, representation="triplet_logits")
outputs = model.predict(images)
```

Families: `dino` means **DINOv2-B**; `siglip` means **SigLIP-B**. Variants map to official archive suffixes: `base` -> `base_model`, `alignet` -> `alignet`, `unalignet` -> `untransformed`.

Input preprocessing uses RGB, direct bicubic resize, float32, and [0, 1] scaling, following the released SigLIP training configuration. The smoke runner obtains image dimensions from the export signature. The public repository does not document a separate DINO inference preprocessing recipe; verify this convention before benchmark comparisons. Both feature outputs are retained so a future experiment can explicitly select its representation.

Dependencies and runtime loading have not been tested in this workspace. The smoke commands are intended to verify compatibility on the machine where you run them. If TensorFlow reports unsupported exported operations, retain the full traceback and signature in the issue report before changing versions or preprocessing.

## Sources

- [Official models and loading API](https://github.com/google-deepmind/alignet#alignet-models)
- [Checkpoint index](https://storage.googleapis.com/alignet/models/index.html)
- [Released preprocessing configuration](https://github.com/google-deepmind/alignet/blob/main/configs/siglip.py)
- [Paper](https://www.nature.com/articles/s41586-025-09631-6)

Checkpoint licenses are supplied by their publisher. Research planning lives in `proposal.md` and `visual-benchmark-research.md`.

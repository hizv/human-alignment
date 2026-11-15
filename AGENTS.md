# Project instructions

## Scope

This repository compares existing Base, AligNet, and UnAligNet vision checkpoints. The initial scope is loading and smoke testing DINOv2-B and SigLIP-B. Extend experiments only when requested.

## Model conventions

- Use official TensorFlow SavedModel exports for all three variants, including Base.
- `dino` refers to DINOv2-B; `siglip` refers to SigLIP-B.
- UnAligNet is the official `untransformed` export.
- Keep preprocessing consistent within each model family and document its provenance.
- Make representation selection explicit in experiments: `pre_logits` and `triplet_logits` are different outputs.
- Keep model downloads in ignored `models/` and results in ignored `outputs/`.
- Do not treat generated smoke images as benchmark evidence.

## Working rules

- Do not download models, install heavy dependencies, or run inference locally unless explicitly requested. The user runs this code elsewhere.
- State which checks were actually run. Do not claim runtime compatibility before an actual smoke run.
- Keep the implementation small; no training framework or experiment infrastructure is needed yet.
- Preserve the research documents unless asked to revise them.
- Do not add credentials or downloaded checkpoint files to Git.
- Explain methods directly in plain language. Avoid unsolicited contrasts and hypothetical-confusion asides.

## Entry points

- `src/human_alignment/models.py`: checkpoint discovery, cache, loading, image preprocessing.
- `src/human_alignment/smoke.py`: command-line smoke tests and saved diagnostics.
- `README.md`: installation and usage.

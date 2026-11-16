# Experiment directive: Navon global–local similarity

## Status and objective

This document specifies the next experiment for planning and later implementation. No experiment has been run as part of this directive.

Compare existing Base, AligNet, and UnAligNet checkpoints on hierarchical letter images: a large letter constructed from repeated smaller letters.

Main question:

> Does human alignment change the relative similarity of a compound image to its global letter versus its local letter?

Treat the direction of the effect as exploratory. Evaluate both isolated-letter references and compound references. Use frozen embeddings and cosine similarity; no classifier training is required for this experiment.

## Models and representations

- Families: DINOv2-B (`dino`) and SigLIP-B (`siglip`).
- Variants: official TensorFlow SavedModel Base, AligNet, and UnAligNet (`untransformed`) exports.
- Primary contrast: AligNet minus UnAligNet within each family. Base provides context for the effects of additional training.
- Proposed primary representation: `pre_logits`, matching the current embedding API default. Confirm this choice before evaluation.
- Treat `triplet_logits` as a separate secondary analysis only after checking the training status and comparability of that output for all variants, especially Base. Do not pool outputs or select an output based on which produces the desired effect.
- Use identical stimuli and preprocessing across variants within each family. Record checkpoint identifiers, output selection, preprocessing, and source provenance.
- The current README flags DINO preprocessing provenance as unresolved. Resolve or explicitly document this limitation before interpreting benchmark results.

## Stimuli and notation

Write `C(G, L)` for a compound with global letter G and local letter L; for example, `C(H, S)` is a large H made of small Ss. Write `I(H)` for an isolated H.

Use a balanced set of distinct letters rather than relying on one H/S example. Include reversed compounds such as `C(S, H)` and ensure letters appear comparably often in global and local roles.

Include congruent compounds such as `C(H, H)` for descriptive checks. The main preference measurements use incongruent compounds with distinct global and local identities.

Keep font, foreground color, background, canvas dimensions, and placement rules consistent. Record the global bounding box, local font size, spacing, element count, and ink coverage. Exact pixel matching across glyphs is not assumed; inspect and document systematic differences.

Vary local element size and spacing in a small predefined set of conditions. Record which parameters are coupled by the rendering method. Verify that local letters remain identifiable and that the global form is preserved after actual model preprocessing.

## Comparison A: compound to isolated letters

For `C(H, S)`, embed the compound, `I(H)`, and `I(S)` separately.

Render isolated H and S with the same font and size setting, color, background, and placement rules. Adding isolated references is a valid measurement; matching their rendering conditions avoids coupling letter identity with reference size.

Repeat the comparison at predefined reference sizes, including a size near the global scale and a size near the local-element scale. At every reference size, render both candidate identities using the same settings and analyze that size separately.

For each compound, define:

`isolated_margin = cosine(C(G, L), I(G)) - cosine(C(G, L), I(L))`

Here cosine operates on embeddings from the selected checkpoint and representation.

- Positive: closer to the isolated global letter.
- Negative: closer to the isolated local letter.
- Zero: tie.

Save both similarities as well as their difference. This measures similarity to the two letter identities under the chosen reference rendering.

## Comparison B: compound to matching compounds

Use a third letter T, distinct from G and L:

| Role | Example | Shared property with reference |
| --- | --- | --- |
| Reference | `C(H, S)` | — |
| Global match | `C(H, T)` | Global H |
| Local match | `C(T, S)` | Local S |

Define:

`compound_margin = cosine(C(G, L), C(G, T)) - cosine(C(G, L), C(T, L))`

Positive values favor the global match; negative values favor the local match. Match rendering parameters across the three images. Balance third-letter identities and role assignments so the result does not depend on one replacement letter.

This comparison measures global versus local preference when both candidates have compound structure. Analyze it separately from comparison A.

## Pilot and design freeze

Before the main evaluation:

1. Render a small grid covering isolated references, reversed compounds, congruent compounds, and each proposed size/spacing condition.
2. Inspect originals and preprocessed images for clipping, legibility, spacing, and preserved global structure.
3. Verify stimulus metadata and that each reference comparison uses matched rendering settings.
4. If model-based task development is necessary, use Base checkpoints only. Reserve fresh stimulus configurations for the final evaluation.
5. Fix the letter set, font(s), rendering conditions, representation, reference sizes, scoring, and uncertainty method before examining AligNet–UnAligNet effects.

The exact stimulus count and rendering values remain to be selected during planning/piloting. More renderings of the same letter combination should not be treated as independent evidence.

## Analysis

For each family, variant, comparison type, and rendering condition, report:

- Mean similarity margin and its distribution.
- Fraction of global choices (`margin > 0`), local choices (`margin < 0`), and ties.
- Paired AligNet–UnAligNet differences in margin and global-choice proportion.
- Base comparisons for context.
- Results by global/local letter pair and reference size to expose identity or scale dependence.

Use paired uncertainty estimates on the same stimuli. A proposed method is a 95% cluster-bootstrap interval, grouping repeated renderings and reversed roles of a letter pair together. For compound comparisons, also account for reuse of letters and candidate images; choose the resampling unit after the final stimulus design is fixed. State whether uncertainty reflects rendering variation, letter combinations, or both.

Cosine-score distributions can differ across checkpoints. Interpret margin changes together with choice proportions and condition patterns. Keep model families and representation outputs separate.

## Interpretation

| Observation | Supported interpretation |
| --- | --- |
| AligNet makes more global choices than UnAligNet across balanced conditions | Alignment shifts similarity preference toward global structure under this evaluation |
| Isolated and compound references yield different effects | The effect depends on reference format; inspect the condition-specific scores |
| The shift depends on element size, spacing, or reference size | Alignment interacts with the visibility or scale of global/local information |
| An interval spans zero | The estimate remains uncertain at the measured precision |

A greater global preference alone does not establish greater human agreement. A claim about human alignment on this task requires an appropriate human comparison on matching stimuli and conditions. Published Navon findings provide motivation; they do not supply human labels for newly generated images.

This first experiment measures similarity preference. A later interference experiment could introduce global/local identification tasks and compare congruent, incongruent, and neutral conditions.

## Outputs for a later implementation

Keep generated stimuli, embeddings, trial scores, and figures under ignored `outputs/navon/`; keep checkpoints under ignored `models/`.

Save:

- Stimulus manifest with unique image IDs, letter roles, rendering parameters, and seeds where applicable.
- Original and preprocessed example grids.
- Embeddings labeled by checkpoint and representation.
- Trial-level similarities, margins, choices, and reference-image IDs.
- Paired comparison tables and condition-level figures.
- A concise run record describing actual checks, preprocessing provenance, and limitations.

Follow the repository's uv workflow when implementation is requested. Do not download checkpoints, install heavy dependencies, or run inference locally without explicit authorization. Generated smoke-test images are not benchmark evidence.

## Sources

- [Navon (1977): Forest before trees](https://cognitionandculture.net/wp-content/uploads/Navon.pdf)
- [Public Navon stimuli and generator](https://github.com/katherine-h/navon): inspect rendering and license before reuse; balance any adopted rotations across comparisons.
- [AligNet model exports and output documentation](https://github.com/google-deepmind/alignet#alignet-models)
- [Earlier project research notes](../visual-benchmark-research.md)

# Psychology-inspired visual benchmarks for AligNet

Research notes for comparing existing Base, AligNet, and UnAligNet checkpoints in the SigLIP-B and DINOv2-B families. These are candidate experiments, not results. The adaptations below are proposals; availability of published stimuli does not necessarily imply availability of matched human responses.

## Main recommendation

Use shape–texture cue conflict as the main experiment, clean recognition as a control, and silhouettes/edges as a companion test. An optional Kimchi–Palmer experiment could test global versus local similarity without fitting a classifier. Navon figures are the closest analogue of the Stroop idea, but task-specific predictions require additional evaluation design.

The central question could be:

> Does human conceptual alignment improve usable shape representations, or mainly change the relative influence of competing visual cues?

## Background correction

AligNet's teacher is aligned using THINGS judgments; students learn from teacher-generated ImageNet triplets. Levels is an evaluation dataset. The published improvement on Levels is background motivation, not a new contribution for this project. A small replication may still serve as an implementation check.

Source: [AligNet paper](https://www.nature.com/articles/s41586-025-09631-6).

## What is a readout?

A vision encoder converts an image into a vector of numbers, called an embedding. A **readout** is the rule that converts that vector into an answer or score for an experiment. It need not be another neural network or require training.

Examples:

- **Reference matching:** choose whichever reference image has the highest cosine similarity to the test image.
- **Category prototypes:** average embeddings of separate clean cat images and dog images; choose the category whose average is closest to the test embedding.
- **Linear classifier:** fit a weighted combination of embedding coordinates to predict a label, while keeping the image encoder frozen.

A **task-specific readout** produces the answer required by a particular task. A Navon global-letter readout predicts the large letter; a local-letter readout predicts the small letters. The input embedding can be identical in both cases.

A **constrained readout** has deliberately limited flexibility, such as nearest-prototype matching or a linear classifier with a fixed training protocol. This is not a distinct architecture or standardized term. The purpose is to reduce the possibility that a powerful added classifier learns the experiment largely on its own.

For a fair checkpoint comparison, keep the readout method, data splits, labels, and fitting rules the same. Each checkpoint normally needs its own reference embeddings or fitted weights because its embedding coordinates differ.

## Similarity scores as measures of conflict

Similarity and prediction scores are useful alternatives to binary accuracy when studying conflict. They are not direct measures of human reaction time.

For a global-target task, define a target margin:

`margin(image) = cosine(embedding(image), target_reference) - cosine(embedding(image), alternative_reference)`

Use the same references for congruent and incongruent conditions within a checkpoint. A proposed conflict cost is:

`conflict_cost = mean(margin_congruent - margin_incongruent)`

A positive cost means conflicting information reduces support for the correct target under the chosen measurement rule. Report accuracy as well. Use matched stimulus sets and independent references; do not choose references after observing alignment effects.

Important qualifications:

- This measures representational or prediction interference, not response speed or executive control.
- Cosine similarities are not calibrated probabilities. Their ranges and distributions can change across checkpoints, so raw score magnitudes alone are insufficient for cross-model conclusions.
- Reference design determines the question being asked. A single-letter reference might favor either local letter identity or global outline depending on scale and layout.
- Congruent-minus-incongruent differences can combine facilitation and interference. A suitable neutral condition can help distinguish them.
- A frozen image encoder does not change its processing when instructed to report a different attribute. Switching the references or classifier changes the evaluation rule, not necessarily the encoder's attention.

## 1. Stroop: competing information under different instructions

**Human experiment:** a color word such as RED is printed in green. Participants either read the word or name the ink color. The classic result is asymmetric interference: conflicting words disrupt color naming more than conflicting colors disrupt word reading.

**Possible model adaptation:** evaluate word identity and ink color using separate reference sets or classifiers on frozen embeddings. Compare congruent, incongruent, and suitable neutral stimuli.

**Question:** does human alignment change how competing attributes affect their accessibility in the representation?

**Limitations:** reading ability is a substantial confound across model families. This adaptation does not directly reproduce human task switching, automaticity, or reaction times. It is a less attractive first experiment than nonverbal hierarchical figures.

Source: [Stroop, 1935](https://psychclassics.yorku.ca/Stroop/).

## 2. Navon figures: global versus local interference

**Human experiment:** identify a large letter made of smaller letters, or identify the small letters. A large H made of small Hs is congruent; a large H made of small Ss is incongruent. Navon's original work found an asymmetry in interference favoring global processing under its experimental conditions.

**Proposed question:** does human alignment change the balance between global configuration and local elements?

**Adaptation:** evaluate global and local identity separately. Options include carefully designed reference matching or separate simple classifiers. For classifiers, use independent training stimuli with global and local identities varied independently. Measure accuracy and target-score changes under conflict on held-out stimuli.

**Controls:** verify that local elements remain legible after preprocessing; vary size, spacing, and density; hold rendering identities and other nuisance factors out of evaluation where appropriate. Global preference should not be assumed universal across stimulus settings.

**Interpretation:** greater global interference would indicate changed representation/readout behavior, not automatically better shape recognition or human-like attentional control.

**Availability:** a public model-oriented dataset and generator contains 26 × 26 letter combinations at five positions, with rotations. Its existence does not establish matched human responses for every image.

Sources:

- [Navon, 1977: Forest before trees](https://cognitionandculture.net/wp-content/uploads/Navon.pdf)
- [Public Navon stimuli and generator](https://github.com/katherine-h/navon)
- [The Origins and Prevalence of Texture Bias in Convolutional Neural Networks](https://arxiv.org/abs/1911.09071)

## 3. Kimchi–Palmer: similarity to the whole or its elements

**Human experiment:** similarity judgments on hierarchical patterns, manipulating the number and relative size of local elements. The perceived organization changes with those manipulations; a blanket claim that people always prefer the global form would be inappropriate.

**Example proposed adaptation:**

| Image | Global arrangement | Local elements |
| --- | --- | --- |
| Reference | Triangle | Small squares |
| Candidate A | Triangle | Small triangles |
| Candidate B | Square | Small squares |

Choose the candidate whose embedding is closer to the reference. A matches global configuration; B matches local elements. Neither is intrinsically the correct answer in a preference task.

**Proposed question:** does conceptual alignment change how global form and local structure contribute to visual similarity?

**Measurements:** global-match proportion and similarity margin across element-size, number, and spacing conditions. Analyze the condition-dependent pattern, not just one average preference.

**Advantages:** works naturally with frozen embeddings and similarity matching; geometric elements avoid letter-recognition requirements.

**Limitations:** this is similarity preference rather than Stroop interference. Without human responses on the actual generated images, comparison with published effects is qualitative. Navon and Kimchi–Palmer tasks need not measure the same phenomenon.

Sources:

- [Kimchi & Palmer, 1982: Form and Texture in Hierarchically Constructed Patterns](https://iipdm.haifa.ac.il/images/publications/Rutie_Kimchi/Kimchi%20and%20Palmer_1982_JEPHPP.pdf)
- [Understanding Navon: Different designs of local-global tasks capture different bias effects](https://link.springer.com/article/10.3758/s13414-026-03300-0)

## 4. Silhouettes and line drawings: recognition without surface cues

**Human experiment:** compare recognition of photographs and simplified line drawings. Biederman and Ju found broadly similar recognition performance across these formats in their experiments.

**Proposed question:** if AligNet chooses shape more often under cue conflict, does it also recognize objects better when surface information is removed?

**Adaptation:** use a shared category-prototype method with separate clean reference images. Evaluate silhouettes, edges, and sketches separately because each removes different information. Compare accuracy and agreement with recorded human responses; inspect matched source images where available.

**Why this matters:** increased shape choice could accompany improved shape information, reduced texture information, or both. Recognition without texture helps interpret the conflict result, though no single manipulation cleanly isolates all mechanisms.

**Availability:** Model-vs-Human provides silhouette, edge, sketch, and cue-conflict datasets with human comparison data. Verify trial mappings and source-image overlap before analysis.

Sources:

- [Biederman & Ju, 1988: Surface versus edge-based determinants of visual recognition](https://citeseerx.ist.psu.edu/document?doi=91f2cb5a7ba9da57c140e6f2303cdd16a4c77ecb&repid=rep1&type=pdf)
- [Model-vs-Human benchmark](https://github.com/bethgelab/model-vs-human/)
- [On the Reliability of Cue Conflict and Beyond](https://arxiv.org/abs/2603.10834)

## 5. Object–scene incongruity: semantic context helps or hurts

**Human experiment:** manipulate relations between objects and scenes, including whether an object belongs in a setting, its position, and its size. Classic work found effects on object detection.

**Proposed question:** does human alignment increase reliance on semantic context, helping recognition in expected settings but hurting it in unexpected ones?

**Adaptation:** compare target recognition across appropriate and inappropriate contexts, ideally preserving target appearance. Include object-alone and background-alone controls and a defined way of locating the target.

**Value:** closely connects conceptual alignment to recognition and allows human-like behavior to have a performance cost.

**Risks:** whole-image embeddings may describe the background rather than the target. Context effects depend on the task; a universal congruency benefit should not be assumed.

**Availability:** SCEGRAM contains controlled scene inconsistencies, validated with consistency ratings and eye-movement data. Access requires requesting a password. These human measures are not automatically recognition labels.

Sources:

- [Biederman, Mezzanotte & Rabinowitz, 1982](https://www.cs.princeton.edu/courses/archive/spr08/cos598B/Readings/Biederman1982.pdf)
- [SCEGRAM database](https://www.scenegrammarlab.com/databases/scegram-database/)
- [Scene Context Impairs Perception of Semantically Congruent Objects](https://www.psychologicalscience.org/journals/psychological-science/09567976211032676/)

## 6. Contour deletion: which missing structure matters?

**Human experiment:** remove comparable amounts of contour near vertices/junctions or along contour midsegments. Biederman's recognition-by-components work reports greater disruption from removing diagnostic vertices under the studied conditions.

**Proposed question:** does alignment change sensitivity to structural information that supports human object recognition?

**Adaptation:** evaluate intact drawings and matched deletion conditions at several severities using the same recognition rule. Compare the loss associated with each deletion type.

**Value:** offers a specific structural prediction rather than testing generic robustness to arbitrary fragmentation.

**Risks:** matching removed information and constructing reliable vertex annotations takes work. The classic paper is evidence for a human effect, not confirmation of a ready-to-use benchmark with trial-level responses.

Source: [Biederman, 1987: Recognition-by-components](https://www.cs.utexas.edu/~grauman/courses/spring2007/395T/papers/Recognition-by-Components.PDF).

## Practical shortlist

| Priority | Experiment | Main purpose |
| --- | --- | --- |
| Main | Shape–texture cue conflict | Choices when cues disagree |
| Control | Clean recognition | General recognition ability |
| Companion | Silhouettes and edges | Recognition with surface information removed |
| Optional extension | Kimchi–Palmer matching | Global versus local similarity without fitting a classifier |
| Alternative extension | Navon interference | Closest visual analogue to Stroop |

Keep AligNet versus UnAligNet as the main comparison, with Base providing context. Verify checkpoint training differences before making causal claims about human alignment alone.

Across tasks, preserve paired trials when calculating uncertainty, keep source-image derivatives together where necessary, and specify scoring before examining final alignment effects. Report shape, texture, and other responses separately on conflict images. Similarity preference, classification accuracy, and agreement with humans are distinct outcomes.

Additional implementation resource: [MindSet: Vision](https://arxiv.org/abs/2404.05290) provides psychology-inspired stimuli, generators, and several model-evaluation approaches. Specific task coverage and human-data availability should be checked before committing to an experiment.

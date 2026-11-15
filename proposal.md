# **Does learning human concepts help a model see shapes better?**

CS 7651: Human and Machine Learning

## The idea

A model can learn that a dog is more similar to a cat than to a car. But does learning these human-like relationships also help it recognize the same shape after it has been rotated or moved?

Our project tests whether **improvement in conceptual similarity carries over to a basic visual task**.

We will also test whether human alignment changes which cue the model follows when shape and texture disagree. For example, if an image has the shape of a cat and the texture of elephant skin, does the model classify it as a cat, an elephant, or something else?

We expect the model to improve at conceptual similarity but possibly show little improvement at shape matching. That is a prediction, not a result. We will also test whether shape matching improves or gets worse.

Our second prediction is that AligNet will follow object shape more often than UnAligNet when texture is misleading. The AligNet authors found that the two models sometimes disagree over object meaning and surface appearance. We will test that pattern on an existing benchmark whose trials were chosen independently of these models’ predictions.

## Relevance

The project connects to a central question in human and machine learning: **does making a model more human-like on one task improve its usefulness as a model of human cognition more generally?**

The two visual tests ask different questions. Generated shapes let us test whether a model can recognize the same unfamiliar outline after a controlled change. Shape–texture conflict images let us test whether a model relies on the same cues as people when recognizing familiar objects. A model could improve on either test without improving on the other.

We will use existing human judgments from Levels and recorded human responses from the Geirhos et al. benchmark. The generated shapes will have a known correct answer, but we will have no human performance measurements on those images. No new human study is planned.

## Models to compare

We will use two model families, **SigLIP-B** and **DINOv2-B**. Each has three versions, giving us six models to evaluate:

| Version | What it represents |
| :---- | :---- |
| **Base** | The original pretrained model. |
| **AligNet** | A version further trained using similarity targets designed to reflect human judgments. |
| **UnAligNet** | A version further trained to match a larger SigLIP teacher’s original image-similarity judgments, before the teacher’s representations are adjusted using human judgments. |

The main comparison is **AligNet versus UnAligNet**. The base model helps us understand what changed after further training.

We need to verify the released training details before claiming that human alignment alone caused any difference. We are evaluating existing models, not training six new ones.

## Research Questions

### 1\. Does conceptual similarity actually improve?

Use the coarse-grained subset of **Levels**, where people chose which image was the odd one out. This subset emphasizes conceptual relationships across object categories.

We will check whether AligNet predicts those choices better than both the base model and UnAligNet. This confirms that the change we want to study is present.

We will test whether AligNet agrees with human judgments more often than both the base model and UnAligNet. For each comparison, we will report the difference in percentage points and its confidence interval. We will conclude that the improvement holds across both model families if all four intervals lie above zero.

### 2\. Does that improvement carry over to shape matching?

Show each model:

- A reference shape.  
- The same shape after a change in position or rotation.  
- A different shape serving as the wrong answer.

Each model turns the images into numerical representations. We use the same similarity calculation to choose which candidate matches the reference, then compare accuracy.

The central question is whether AligNet performs better than UnAligNet on this task.

### 3\. Does ordinary teacher training help recognize clean images?

Test whether UnAligNet classifies clean images more accurately than the base model. Use the same object categories as the shape-texture benchmark. The larger teacher supplies image-similarity information that could improve recognition, and the original paper reports gains from ordinary distillation on some tasks and checkpoints. Whether it helps here is a question we can test.

### 4\. Does human alignment change what happens when shape and texture disagree?

Use Geirhos et al.’s shape-texture conflict images. For each image, record whether the model chose the category indicated by the shape, the category indicated by the texture, or another category. The main comparison is how often AligNet and UnAligNet choose the shape category on the same trials. Compare their choices with the recorded human responses where the labels can be matched.

For both object-recognition experiments, build a reference representation for each category by averaging that checkpoint’s embeddings of separate clean reference images. Classify each test image by cosine similarity to those category references. Where we can match a conflict image to its original clean source, compare how each checkpoint’s prediction changes from clean to conflict. This helps distinguish a shift in cue use from a general improvement in recognition.

## To keep the comparison fair

- Use the same images and scoring method for all versions within each model family.  
- Choose task difficulty using only the base models, before examining AligNet’s shape-matching results.  
- Keep performance above guessing level and below near-perfect accuracy.  
- Control the similarity of the wrong answer to the reference.  
- Check that image overlap, shape size, or candidate position cannot provide an easy shortcut.  
- Check whether rotation or translation actually changes base-model performance. If both leave little room to detect an effect, pilot one planned alternative, such as contour fragmentation.  
- For object recognition, use the same category labels, clean reference images, and test trials for every checkpoint. Keep benchmark source images and their derivatives out of the clean reference set.  
- Record the scoring and analysis rules before the main evaluation.

No task-specific model training is planned.

## Data needed

| Purpose | Number of shape pairs |
| :---- | ----: |
| Develop the task and choose its settings | 500 |
| Check the chosen setup on fresh examples | 1,000 |
| Run the final experiment | 6,500 |
| **Total** | **8,000** |

Each pair contains one reference shape and one different-shape alternative. The correct match is a transformed copy of the reference.

This requires **16,000 distinct source shapes**. A shape cannot appear in another pair, even as the wrong answer. Both model families use the same final pairs.

We will start with the **MindSet stimulus generator**. If it cannot produce enough distinct shapes, the fallback is a procedural generator of closed outlines.

For object recognition, we need the published conflict images and their shape and texture labels, the corresponding human responses where available, and clean images from the same categories. Split the clean images into a category reference set and a separate test set. Check whether conflict trials reuse source images before deciding how to calculate uncertainty.

## Work split

| Person | Main responsibility | Main outputs |
| :---- | :---- | :---- |
| 1\. Images and task checks | Build the shape pairs, prevent repeated identities, control wrong-answer similarity, and pilot difficulty. | Image sets, generation code, and a record of the task checks. |
| 2\. Models and predictions | Load the six model versions, verify image processing and training details, and run the shape experiment. Implement the shared readout for clean and conflict images. | Saved model representations, predictions, and reproducible evaluation code. |
| 3\. Human similarity test | Prepare Levels and reproduce its scoring. Map the conflict benchmark’s categories and human responses. | Conceptual-similarity results and a table linking conflict images, cue labels, and available human responses. |
| 4\. Analysis and figures | Compare models on the same trials and calculate uncertainty ranges. Show whether predictions shift among shape, texture, and other categories. | Statistical analysis, main figures, and an explanation of what the results support. |

Everyone contributes to the literature review, interpretation, report, and presentation. Each person should review another person’s outputs.

## Interpretation Plan

For shape matching, we will test whether AligNet is more accurate than UnAligNet on the same trials. We will report the difference in percentage points and its confidence interval for each model family.

| Result | What we could conclude |
| :---- | :---- |
| AligNet’s shape-matching accuracy is higher than UnAligNet’s, with the confidence interval above zero | The benefit of alignment carries over to this controlled shape task. |
| The shape-matching confidence interval includes zero | We found evidence of a conceptual gain. The evidence is inconclusive about whether shape matching improved |
| Conceptual similarity improves, but shape matching gets worse. | Alignment may involve a trade-off on this task. |
| The conceptual check fails, or the shape-matching uncertainty range is too wide. | The experiment does not support a clear conclusion about transfer. |
| UnAligNet improves over base on clean images, and AligNet chooses the shape category more often than UnAligNet on conflict images. | Ordinary teacher training helps clean recognition, and the human-aligned checkpoint follows shape more often under conflict. The matched clean-to-conflict comparison helps assess whether that second result goes beyond general recognition gains. |
| AligNet follows shape more often in conflict images, while its generated shape-matching performance changes little. | Alignment may change how models weigh shape against texture when recognizing familiar objects, even when it yields little benefit for matching unfamiliar outlines. |

We will report the estimated change and its uncertainty range. To make a conclusion that holds across both model families, both must support it. Different outcomes between families are also useful findings.

For cue conflict, report all three response categories and human agreement separately. A fall in texture choices could reflect more shape choices or more unrelated errors. The shape-matching equivalence margins and sample-size calculation apply only to the generated shape task.

## Checks (to be done)

Before committing to the final evaluation, we need to establish that:

- The generated task measures shape matching without obvious shortcuts.  
- We can produce enough distinct shapes.  
- The released model versions provide a suitable comparison.  
- The Levels improvement meets our conceptual criterion.  
- The sample size and available computing resources are sufficient.  
- The conflict stimuli, cue labels, and human responses can be linked.  
- The clean reference and test sets are separate, and the category readout can recognize clean images above chance.

For the two-week scope, start with Levels and the clean/conflict evaluation using published data. Develop the generated shape experiment in parallel and run it at full scale if the pilot passes**.** If the evidence is weak, we will report the uncertainty rather than force a conclusion.

## Final outputs

We will deliver:

- Image-generation and model-evaluation code.  
- Saved predictions and a reproducible analysis.  
- Conceptual-similarity and shape-matching results.  
- Clean-image accuracy, shape-versus-texture choices, and comparisons with existing human responses.  
- Clear figures comparing the models.  
- A short report and presentation explaining the cognitive-science implications.

The contribution is to test both whether conceptual alignment improves controlled shape matching and whether it changes which cue drives recognition when shape and texture disagree. Our targeted search found no report of the exact base–UnAligNet–AligNet comparison on the Geirhos benchmark.

## Starting resources

- [AligNet models and code](https://github.com/google-deepmind/alignet)  
- [AligNet paper](https://www.nature.com/articles/s41586-025-09631-6)  
- [Levels human similarity dataset](https://gin.g-node.org/fborn/Dataset_Levels)  
- [MindSet image generator](https://github.com/ValerioB88/mindset-vision)  
- [Geirhos et al. shape–texture cue-conflict paper](https://arxiv.org/abs/1811.12231)  
- [Model-vs-Human benchmark and evaluation code](https://github.com/bethgelab/model-vs-human)
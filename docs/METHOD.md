# Exact CRSS protocol

## Stage I

The fixed expert order is `vit, clip, dino, deit3, convnextv2`. Inputs are
normalized frozen features, observed labels and original sample indices.
Clean training labels are absent from the scientific evidence API.

1. For each expert, sum the features for each observed class and L2-normalize
   that sum. Similarities to these prototypes induce hard predictions,
   intrinsic confidence (top-two cosine gap) and softmax probabilities with
   `tau=0.1`. The original two-pass numerical evaluation order is preserved.
2. Extrinsic consensus is the fraction of the other four hard predictions
   agreeing with that expert. Rank IC and EC separately across samples with
   average ties: `(rank-1)/(N-1)`. Expert reliability is their arithmetic mean.
3. USC is the **uniform mean** of five soft probability vectors. Its argmax is
   the USC label; its maximum probability is Q_USC. Sort by descending Q_USC,
   with ascending original index as the tie break, and retain `floor(rho*N)`
   candidate samples. Reliability does not weight USC or the candidate rank.
4. Delta_R equals the sum of reliable hard votes supporting the USC label
   minus the sum supporting the observed label. RC is candidate with equal
   USC/observed labels. CN is candidate with unequal labels and `Delta_R > 0`.
   Every other sample is U. A zero Delta_R does not permit a correction.

The masks are disjoint and exhaustive. `group_id` is 0=RC, 1=CN, 2=U.
Changing rho reroutes samples using the same core evidence. Changing tau or
input features/observed labels requires different evidence.

## Stage II

One `torch.nn.Linear` consumes the corresponding frozen ViT features. For
sample i with probabilities p and observed label y, the loss is:

| Group | Loss |
| --- | --- |
| RC | `-log p[y]` |
| CN | `-log p[argmax(USC)]` |
| U | `(1-p[y]**q)/q`, q=0.7 |

All samples enter the shuffled optimizer loop; the last incomplete batch is
retained. Loss reduction is the ordinary batch mean, without group balancing,
extra holdouts, teacher updates, soft CN targets, or backbone training.

Defaults: AdamW, learning rate 0.001, weight decay 0.0001, batch size 256,
maximum 100 epochs, constant learning rate, training seed 0, patience 20.
Test accuracy is evaluated after every epoch. Only strict improvement replaces
the checkpoint; ties retain the earliest epoch. Restore that checkpoint for
all final classification metrics, including ECE.

This protocol uses test GT for model selection. Clean **training** GT is only
read for retrospective diagnostics and does not influence the loss or masks.

## Raw evaluation

`training/linear.py:evaluate` computes accuracy, raw ECE15, NLL and multiclass
Brier score from restored logits. Softmax evaluation uses float64. ECE has
15 equal-width confidence bins with internal-edge ties assigned to the lower
bin (`searchsorted(..., side='left')`). Each bin contributes its sample fraction
times the absolute confidence/accuracy gap. There is no fitted temperature.
Accuracy/ECE/F1 are stored as fractions; multiply by 100 for percentage tables.

## Partition diagnostics and F1 conventions

Clean/noisy truth always describes the **original observed label**:
`observed_label == clean_label` is clean. CN correction accuracy evaluates the
USC corrected label separately; it does not redefine original-label noise truth.

The original Stage I JSON reports:

- RC/CN/U counts/fractions, candidate fraction and decided coverage.
- RC observed-label accuracy, CN noisy precision/correction accuracy,
  U observed-label and USC accuracies.
- Clean=RC and noisy=CN precision/recall/F1, with full-population recall
  denominators; U can contribute false negatives to both tasks.
- Selective clean/noisy F1 and macro F1 on RC∪CN only.

For the later **full-coverage U–USC** comparison convention, use these offline
binary predictions without changing training or stratification:

```python
pred_clean = result.rc_mask | (result.u_mask &
                              (result.usc_prediction == result.observed_labels))
pred_noisy = result.cn_mask | (result.u_mask &
                              (result.usc_prediction != result.observed_labels))
# Evaluate against observed_label == clean_label over every sample.
from utils.evaluation import partition_metrics
full_metrics = partition_metrics(pred_clean, pred_noisy,
                                 result.observed_labels, clean_labels)
full_population_macro_f1 = full_metrics['macro_f1']
```

Do not substitute selective F1 for full-population F1, or corrected-label
accuracy for original-label clean/noisy F1. Zero-division behavior of this full
binary metric is 0. Aggregate each metric within dataset × scale × noise over
seeds, using sample std (`ddof=1`); one fixed Human realization has no replicate
variation to estimate.

## Cache contracts

Feature metadata includes family, scale, display/model name, checkpoint,
preprocessing, dimension, frozen/pretrained flags and matrix SHA256. Test
features additionally bind the matching train hash. Core evidence binds all
five source hashes, observed-label hash, tau and task identity. Partitions bind
core hash and rho. Stage II binds partition, training config and test hashes.

Only Base can read the legacy family-level feature layout. The release
preserves feature model identities but renames shared utility modules;
implementation hashes of newly produced outputs can consequently differ
from development-repository outputs. Fresh outputs remain separated by run
identity. Do not edit historical DONE hashes to force a resume.

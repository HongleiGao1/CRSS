# Cross-Expert Reliability-Guided Supervision Stratification for Learning with Noisy Labels

PyTorch implementation of **Cross-Expert Reliability-Guided Supervision
Stratification (CRSS)**.

**Honglei Gao, Senyu Hou, Gaoxia Jiang, Wenjian Wang**

This repository provides the core CRSS implementation: frozen feature
extraction, Stage I supervision stratification, and Stage II robust linear training.
The released reproduction protocol covers CIFAR-10 and CIFAR-100 under
Weak/Base/Strong expert configurations.

## Abstract

Label noise is widespread in large-scale datasets and can substantially degrade supervised
learning and model generalization. Traditional learning with noisy labels (LNL) methods
typically adopt an end-to-end training paradigm, using noisy labels for both sample selection
and representation learning, which may lead to self-confirmation bias. Frozen pretrained expert
models (PEMs) provide stable visual representations and offer a promising approach to learning
with noisy labels. However, existing methods often rely on a single PEM and remain susceptible
to model-specific prediction biases. To address this limitation, we propose Cross-Expert
Reliability-Guided Supervision Stratification (CRSS), which exploits complementary evidence from
multiple frozen PEMs. CRSS first exploits complementary predictions from multiple frozen
pretrained experts and models their instance-wise reliability to assess noisy supervision. It
then combines cross-expert consensus with reliability-validated correction to partition training
samples into reliable clean, correctable noisy, and uncertain subsets. Finally, a lightweight
linear classifier is trained with differentiated supervision for the three subsets, enabling
robust utilization of all training samples. Theoretically, we analyze the proposed method from
the perspectives of consensus-based sample selection, reliability-guided label correction, and
robust learning with uncertain supervision. Experiments across different expert configurations
and datasets with synthetic and real-world label noise demonstrate that the proposed method
achieves superior performance in sample stratification, classification accuracy, and prediction
calibration.

## Framework

![Figure 1. Motivation of CRSS: single-expert failure and multi-expert complementarity.](assets/framework.png)

**Figure 1. Motivation of CRSS.** A single pretrained expert can reinforce an
incorrect observed label, while complementary expert predictions can recover
the correct class. Partially overlapping errors and the gap between the best
single expert and the retrospective any-expert oracle motivate combining
cross-expert consensus with instance-wise reliability assessment. The figure
is reproduced from Figure 1 of the manuscript, including all three panels.

CRSS has two stages:

1. **Reliability-guided supervision stratification.** Instance-wise Expert
   Reliability Modeling (IERM) combines individual confidence and expert
   consistency. Uniform Soft Consensus (USC) proposes labels and ranks
   candidates, while Reliability-Validated Correction (RVC) validates disputed
   corrections. Every sample is assigned to the reliable clean (RC),
   correctable noisy (CN), or uncertain (U) subset.
2. **Differentiated robust training.** A lightweight linear classifier uses
   frozen ViT features from the corresponding scale. RC uses observed-label
   cross entropy, CN uses hard consensus-label cross entropy, and U uses
   observed-label generalized cross entropy. All training samples contribute
   to optimization; the pretrained experts remain frozen.

See [the method and evaluation protocol](docs/METHOD.md) for the exact rules.

## 1. Preparing the Python Environment

Run all commands from the repository root. Python 3.11 is the environment
target; a CUDA-capable PyTorch installation is recommended for full experiments.

```bash
git clone https://github.com/HongleiGao1/CRSS.git
cd CRSS
```

```bash
conda env create -f environment.yml
conda activate crss
```

Alternatively, use an existing Python environment:

```bash
pip install -r requirements.txt
```

`requirements.txt` lists compatibility lower bounds. The local environment
used for release checks is recorded in [release notes](docs/RELEASE_NOTES.md);
it is not a claim that every possible package combination has been tested.
Install the PyTorch/torchvision pair appropriate to the server's CUDA runtime.

## 2. Datasets and Pretrained Models

### 2.1 Datasets and noisy labels

The released protocol supports **CIFAR-10 and CIFAR-100**, with 50,000 training
and 10,000 test samples each, in torchvision's original sample order.

| Noise setting | CLI name | Protocol | Observed-label seeds |
| --- | --- | --- | --- |
| Human | `human` | CIFAR-10N `worse_label`; CIFAR-100N `noisy_label` (fine labels) | 1, fixed annotation |
| Symmetric | `sym` | 0.6 transition probability to other classes | 1–5 |
| Pairflip | `pair` | 0.3 cyclic transition from class c to (c+1) mod C | 1–5 |
| Instance-dependent | `idn` | Audited FM-SRM JYP matrix generator, nominal 0.4 | 1–5 |

Each dataset therefore has **48 CRSS runs**:
3 model scales × (1 Human + 5 Sym + 5 Pair + 5 IDN).
Both datasets have 96 CRSS runs. Observed-label seeds are distinct from
the fixed head initialization/shuffle seed (`training_seed=0`).

The default layout is:

```text
../DATASETS/
  cifar-10-batches-py/
  cifar-100-python/
  noise_label_human/
    CIFAR-10_human.pt
    CIFAR-100_human.pt
```

Use `--data-root /path/to/DATASETS` to override the location. CIFAR images can
be downloaded with `--download` in the preparation/extraction wrappers.
Supply the CIFAR-N annotation files separately; their clean labels must match
the torchvision sample order.

Prepare the shared immutable observed-label caches once:

```bash
python scripts/prepare_labels.py --dataset cifar10 --data-root ../DATASETS/
python scripts/prepare_labels.py --dataset cifar100 --data-root ../DATASETS/
```

These commands create 16 label realizations per dataset under
`cache/noise/stage12/<dataset>/`. Every scale reads those same labels.
Existing unique caches are validated and reused, without regeneration.
Keep the `.npz` and adjacent `.json` files together when transferring caches.
If matching caches conflict, the command fails instead of choosing one silently.

To reproduce the **exact existing experimental results**, reuse the original
observed-label files. Regeneration uses the same audited generator, but IDN
can differ across devices/library versions; equal seeds alone do not certify
bitwise equality to historical labels. `--reference-results` optionally imports
archived FM-SRM partition labels. Preparation defaults to CPU and records the
generator/device/package provenance. No clean training GT is used to choose
CRSS partitions or optimize the classification heads.

### 2.2 Five expert families and three scales

| Family | Weak | Base (default) | Strong |
| --- | --- | --- | --- |
| ViT | B/32 | B/16 | L/16 |
| CLIP | B/32 | B/16 | L/14 |
| DINOv2 | S/14 | B/14 | L/14 |
| DeiTIII | S/16 | B/16 | L/16 |
| ConvNeXtV2 | Tiny | Base | Large |

Only these five families are exposed. The authoritative checkpoint identifiers
are in [models/model_registry.py](models/model_registry.py) and listed in
[docs/MODELS.md](docs/MODELS.md). Base checkpoint identities, preprocessing,
and normalized feature extraction remain those of the validated implementation.
All encoders run in evaluation mode with gradients disabled. Checkpoints are
obtained by timm/open_clip, or through supported local checkpoint paths.

## 3. Running CRSS

### 3.1 Extract frozen features

One command extracts **all three scales × five models**, training and test sets,
for one dataset. Train extraction precedes test extraction for each scale.

```bash
python scripts/extract_features.py --dataset cifar10 --model-scale all --split all --device cuda:0
python scripts/extract_features.py --dataset cifar100 --model-scale all --split all --device cuda:0
```

For one scale, the original entry remains available:

```bash
python run_extract.py --config configs/server.yaml --dataset cifar10 --data-root ../DATASETS/ --model-scale base --noise-type none
python run_extract.py --config configs/server.yaml --dataset cifar10 --data-root ../DATASETS/ --model-scale base --noise-type none --split test
```

Omitting `--model-scale` in the single-scale entries selects Base. Extraction
uses `--noise-type none` because geometry is independent of noisy labels;
this does **not** create the Stage I observed-label realizations. Run the label
preparation commands in Section 2.1 separately. Optional wrapper flags include
`--batch-size`, `--num-workers`, `--models`, and `--data-root`.

### 3.2 Stage I: RC/CN/U stratification

```bash
python scripts/run_crss.py --stage stage1 --dataset cifar10 --model-scale base --noise sym --seed 1
```

The default candidate fraction is `rho=0.6`, prototype temperature `tau=0.1`.
Core evidence and rho-specific partitions are stored separately, with input
hashes and scale metadata. Stage I reads existing features; it never instantiates
a foundation model or regenerates observed labels.

### 3.3 Stage II: robust linear training

```bash
python scripts/run_crss.py --stage stage2 --dataset cifar10 --model-scale base --noise sym --seed 1 --cuda 0 --resume
```

Stage II requires the requested Stage I artifact and a matching frozen ViT test
cache. It fails if they are missing or inconsistent. To run both stages:

```bash
python scripts/run_crss.py --stage all --dataset cifar10 --model-scale base --noise sym --seed 1 --cuda 0 --resume
```

### 3.4 Full dataset matrix

After label preparation and feature extraction, run all 48 CRSS experiments:

```bash
python scripts/run_crss_matrix.py --dataset cifar10 --cuda 0 --resume
python scripts/run_crss_matrix.py --dataset cifar100 --cuda 0 --resume
```

Use `--dataset all` for both datasets, `--stage stage1`/`--stage stage2` to
separate stages, or `--dry-run` to print the task list. One failed task does not
stop the remaining matrix; the exit code and failure manifest report failures.
`--resume` skips only complete artifacts whose identities and hashes match.
It does not resume a partially completed optimizer trajectory.

## 4. Method Defaults and Evaluation

| Parameter | Value |
| --- | --- |
| CRSS candidate fraction / prototype temperature | rho=0.6 / tau=0.1 |
| Stage II target feature | ViT of the requested scale |
| RC / CN / U loss | observed CE / hard USC CE / observed GCE |
| GCE exponent | 0.7 |
| Optimizer / learning rate / weight decay | AdamW / 0.001 / 0.0001 |
| Batch size / maximum epochs | 256 / 100 |
| Early stopping patience | 20 epochs without strictly improved test accuracy |
| Scheduler / training seed | none / 0 |
| Checkpoint ties | earliest epoch |
| ECE | raw probabilities, 15 equal-width bins, at the selected checkpoint |

**The existing experiments select checkpoints using test accuracy every epoch.**
The test set is therefore used for model selection, rather than being an
independent final evaluation set. The release preserves this behavior and
records it explicitly in metadata. ECE is measured after restoring the best
accuracy checkpoint; it is not independently minimized. No temperature fitting
or third stage is included.

Training-label GT is used only for retrospective partition metrics. CRSS reports
RC/CN/U sizes, decided coverage, clean/noisy precision/recall/F1, CN correction
accuracy, and U observed/USC accuracy. CRSS's selective F1 must not be treated
as a full-population binary F1. [docs/METHOD.md](docs/METHOD.md) explains the
optional full-coverage U–USC comparison convention used in the analysis reports.
Mean/std reporting groups seeds within dataset × scale × noise, with sample
standard deviation (`ddof=1`). Human has one fixed label realization, not five
independent random trials.

## 5. Reusing Existing Feature Caches

```text
cache/features/<dataset>/<scale>/<family>/<model_identity>/
  train_l2.npz
  test_l2.npz
cache/noise/stage12/<dataset>/
  <noise>_seed<N>_<identity>.npz
  <noise>_seed<N>_<identity>.json
```

You can copy existing `cache/features` and `cache/noise/stage12` into this
repository. Model identities and scientific feature arrays are preserved.
Legacy `cache/features/<dataset>/<family>/...` directories are recognized
**only for Base**. Weak/Strong cannot fall back to Base. Test files must remain
beside their matching train files. Do not rename or mix checkpoint identities.

For cache-only partition/training, use `--cache-root /existing/cache` and
`--noise-cache-root /existing/cache/noise/stage12`. For extraction reuse after
moving data, `--data-root`
must match the train-cache dataset provenance; use the original data location
or perform a fresh extraction into a separate cache root.

For an offline server, transfer the required timm/Hugging Face/OpenCLIP weights
or use `models.<family>.checkpoint_path` in a copied feature configuration.
`hf_endpoint` can point to an accessible Hugging Face endpoint; it does not
configure the OpenAI CLIP download host or replace a general network proxy.
Keep preprocessing identical to the recorded cache. Missing pretrained weights
cause an error, not random-weight fallback.

## 6. Outputs and Code Layout

```text
outputs/
  feature_extraction/             # extraction logs and provenance
  crss_stage1/<dataset>/<scale>/<noise>/seed_N/
    core/<evidence_key>/          # core.npz, expert_probs.npy, core_metadata.json
    partitions/<rho_tag>/<key>/   # partition.npz, stage1_metrics.json/csv,
                                 # partition_metadata.json, sample_assignments.csv
  crss_stage2/<dataset>/<scale>/<noise>/seed_N/<rho_tag>/run_<hash>/
    metrics.json / metrics.csv / history.csv / metadata.json
    best_head.pt / test_predictions.npz / DONE.json
```

The key review artifacts are **Stage I metrics and partition metadata**, then
**Stage II metrics, epoch history, and training metadata**. NPZ masks/evidence
and best heads support sample-level inspection and checkpoint verification.
Generated outputs/caches/weights/data are ignored by Git; synchronize them
separately when comparing server results. A copied metrics file alone is not
enough for `--resume`, which validates complete artifacts.

```text
crss/                  # prototypes, reliability, USC, RVC, RC/CN/U rules
crss_two_stage/        # artifact identity and Stage I/II orchestration
training/              # shared linear trainer, device resolution, raw evaluation
models/ features/      # frozen loaders, scale registry, normalized caches
datasets/              # indexed CIFAR input, immutable noise preparation/readers
scripts/               # preparation, extraction, partition/training entry points
vendor/                # self-contained FM-SRM support snapshot
configs/ docs/ assets/
```

The public repository contains only CRSS and the shared data/model utilities
needed to reproduce it. Comparison-method implementations, historical
verification sweeps, ablations, calibration pipelines, analysis reports,
experimental outputs, private data paths, and development Git history are
excluded. CRSS can run independently of the development repository and the
local FM-SRM checkout.

## Citation

If you use this code or build on this work, please cite:

```bibtex
@unpublished{gao2026crss,
  title  = {Cross-Expert Reliability-Guided Supervision Stratification for Learning with Noisy Labels},
  author = {Gao, Honglei and Hou, Senyu and Jiang, Gaoxia and Wang, Wenjian},
  year   = {2026},
  note   = {Manuscript}
}
```

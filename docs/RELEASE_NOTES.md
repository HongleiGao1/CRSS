# CRSS reproduction release

## Public scope

The public repository contains five frozen model families at Weak/Base/Strong
scales, feature extraction, immutable observed-label preparation, CRSS Stage I
stratification and Stage II robust linear training. Its production entry points
are `run_extract.py`, `scripts/extract_features.py`, `scripts/prepare_labels.py`,
`scripts/run_crss.py` and `scripts/run_crss_matrix.py`.

Comparison-method implementations, their configurations/runners and their
specialized feature-view helpers are local-only. They are excluded by the Git
allowlist and are absent from the public commit history. Historical sweeps,
analysis scripts, generated results, data and model weights are also excluded.

The selected source revision and file hashes are in `source_manifest.json`.
Shared data/model/noise utilities under `vendor/` are required by CRSS; no
comparison-method training or sample-selection pipeline is bundled there.
Original attribution in the shared source modules is retained.

## Packaging changes (2026-10-12)

- The README uses the current manuscript title, abstract, author order, complete
  Figure 1 and a short manuscript citation.
- The public instructions describe only CRSS reproduction.
- The CRSS matrix runner now enumerates its own tasks instead of importing a
  comparison-method configuration module. Task order, fixed Human seed, and
  the 48 runs per dataset are unchanged.
- CRSS losses, partition rules, model versions, preprocessing, cache identities,
  optimizer defaults, checkpoint selection and metric calculations are unchanged.

## GitHub publication (2026-10-14)

The public repository is `https://github.com/HongleiGao1/CRSS`. Standalone
copyright/license documents are local-only, alongside comparison-method code.
Their entries are excluded from the public source manifests and the initial
public commit. The README retains the manuscript citation.

## Validation

Release preparation checked the original scientific core against the
source implementation, including CRSS evidence, partitions, losses, training
history, best-epoch selection, raw evaluation and legacy Base cache behavior.
The current publication split additionally checks standalone source imports,
production CLI help, label/task dry runs and that no comparison code is present
in the exported source or reachable Git history.

These are packaging and compatibility checks, not new benchmark experiments.
Full GPU feature extraction/training and actual CIFAR-N annotation loading
were not rerun. No pretrained weights or experimental results are bundled.

## Preparation environment

| Component | Version |
| --- | --- |
| Python | 3.11.15 |
| PyTorch | 2.5.1+cu121 |
| torchvision | 0.20.1+cu121 |
| timm | 1.0.29 |
| open_clip_torch | 3.3.0 |
| NumPy | 2.4.6 |
| SciPy | 1.17.1 |
| scikit-learn | 1.9.0 |
| pandas | 3.0.5 |
| PyYAML | 6.0.3 |

Use matching feature and observed-label caches for exact historical comparisons;
regenerated IDN labels can differ across library versions or devices.

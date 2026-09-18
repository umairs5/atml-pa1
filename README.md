# ATML PA1

Reproducible code and machine-readable outputs for Programming Assignment 1. Task 1 uses STL-10 to compare frozen ImageNet ResNet-50 and ViT-B/16 features with OpenAI CLIP ViT-B/32 features under controlled visual interventions.

## Environment

Create a Python environment with the packages in `requirements.txt`. Install a CUDA-compatible PyTorch and torchvision build separately when using a GPU.

```bash
pip install -r requirements.txt
```

## Task 1 reproduction

All commands are run from the repository root. They use seed 6304 and the configuration in `task1/configs/clean_baseline.yaml`.

```bash
python task1/scripts/make_subset.py
python task1/scripts/run_clean_baseline.py
python task1/scripts/run_color_interventions.py
python task1/scripts/run_spatial_interventions.py
```

The exact 80/20 STL-10 training/validation split and the balanced 500-image official-test subset are saved in `task1/data/splits/stl10_seed6304.json`. Raw datasets and checkpoints are intentionally excluded from version control.

### Cue conflicts

Clone the external AdaIN implementation and supply its local path when generating cue conflicts:

```bash
git clone https://github.com/naoto0804/pytorch-AdaIN.git
python task1/scripts/generate_cue_candidates.py --adain-root /path/to/pytorch-AdaIN
python task1/scripts/select_cue_candidates.py
python task1/scripts/run_cue_conflicts.py
python task1/scripts/select_cue_examples.py
```

The screening rule is stored in `task1/results/cue_conflicts/screening_protocol.json`; the balanced selection log is in `selection_metadata.csv` and `selection_summary.json`. Candidate PNGs and contact sheets are generated locally, but excluded from Git because they are derived artifacts. The recorded run generated 25 candidates for each of ten directions, retained 20 per direction, and evaluated 200 conflicts.

### Representations and figures

```bash
python task1/scripts/run_representation_analysis.py
python task1/scripts/make_task1_figures.py
```

The representation script calculates cosine stability and creates a UMAP figure for each backbone. The figure script produces the compact color/patch comparison, translation curves, and cue-conflict summary under `task1/results/figures/`.

## Results

Small CSV files in `task1/results/metrics/` are the report-facing outputs. They include clean baselines, color and spatial interventions, cue-conflict counts and coverage, selected qualitative examples, and representation stability. Configuration and split files are committed so every reported aggregate can be regenerated.

## External code and assets

- AdaIN stylization uses the public [pytorch-AdaIN](https://github.com/naoto0804/pytorch-AdaIN) implementation by Naoto Inoue, based on Huang and Belongie (2017). It is not vendored in this repository.
- Model loading uses torchvision pretrained weights and the `open_clip_torch` package. The required model variants are declared in `task1/models/backbones.py`.

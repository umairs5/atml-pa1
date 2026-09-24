# ATML PA1: Learning Beyond IID and Closed-Set Classification

This repository contains the code, fixed data protocols, configuration files,
and machine-readable results for all four parts of ATML Programming Assignment
1. The required seed is `6304`. Raw datasets, downloaded pretrained weights,
cached features, and trained checkpoints are intentionally excluded.

## Repository map

| Location | Contents |
| --- | --- |
| `task1/` | STL-10 visual-cue interventions on frozen ResNet-50, ViT-B/16, and CLIP ViT-B/32. |
| `shared/` | PACS parsing and the common Task 2/3 split protocol. |
| `task2/` | Target-aware PACS adaptation: Source-only, DAN, DANN, and CDAN. |
| `task3/` | Target-free PACS generalization: ERM, DAN-DG, and SAM. |
| `task4/` | CIFAR-10/CIFAR-100 open-set recognition: Vanilla, GCSC, and PROSER. |
| `requirements.txt` | Python package specification. |

Each task directory also has a focused README. Committed `results/` files are
the machine-readable counterparts of the report tables and figures.

## Environment

Use Python 3.10 or newer. Install a platform-compatible PyTorch/torchvision
pair (especially for CUDA), then install the remaining dependencies:

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

`torch` and `torchvision` are not pinned to one CUDA build. Select the correct
pair for the machine from [pytorch.org](https://pytorch.org/get-started/locally/).

## Data and fixed splits

Do not commit datasets. Expected roots are:

```text
data/
  pacs/PACS_Original/{art_painting,cartoon,photo,sketch}/
  cifar/
```

Torchvision downloads STL-10, CIFAR-10, and CIFAR-100 under the supplied data
root. PACS must be downloaded and extracted manually. Reuse these committed
manifests rather than regenerating a different split:

- `task1/data/splits/stl10_seed6304.json`: STL-10 80/20 split and balanced
  500-image official-test subset.
- `shared/splits/pacs_sketch_seed6304.json`: PACS source split and Sketch
  protocol for Tasks 2 and 3.
- `task4/data/splits/cifar10_seed6304.json`: CIFAR-10 90/10 split.

## Task 1: Inductive biases and representations

```bash
python task1/scripts/make_subset.py
python task1/scripts/run_clean_baseline.py
python task1/scripts/run_color_interventions.py
python task1/scripts/run_spatial_interventions.py
python task1/scripts/run_representation_analysis.py
python task1/scripts/make_task1_figures.py
```

Cue conflicts require the external AdaIN implementation and its weights:

```bash
git clone https://github.com/naoto0804/pytorch-AdaIN.git /path/to/pytorch-AdaIN
python task1/scripts/generate_cue_candidates.py --adain-root /path/to/pytorch-AdaIN
python task1/scripts/select_cue_candidates.py
python task1/scripts/run_cue_conflicts.py
python task1/scripts/select_cue_examples.py
```

The screening protocol and selection logs are committed under
`task1/results/cue_conflicts/`; generated candidate images are ignored. Report
evidence is in `task1/results/metrics/`.

## Task 2: Target-aware domain adaptation

After placing PACS at `data/pacs/PACS_Original/`, create the shared protocol
and run every fixed configuration:

```bash
python -m shared.make_pacs_protocol --data-root data/pacs/PACS_Original
python -m task2.train --config task2/configs/source_only.yaml
python -m task2.train --config task2/configs/dan.yaml
python -m task2.train --config task2/configs/dann.yaml
python -m task2.train --config task2/configs/cdan.yaml
python -m task2.train --config task2/configs/dan_lambda_0_1.yaml
python -m task2.train --config task2/configs/dan_lambda_10.yaml
```

Only after every model and setting is fixed using source validation, release
Sketch labels for final evaluation:

```powershell
python -m task2.evaluate_final `
  --checkpoint source_only=task2/results/source_only/best.pt `
  --checkpoint dan=task2/results/dan/best.pt `
  --checkpoint dann=task2/results/dann/best.pt `
  --checkpoint cdan=task2/results/cdan/best.pt `
  --checkpoint dan_lambda_0_1=task2/results/dan_lambda_0_1/best.pt `
  --checkpoint dan_lambda_10=task2/results/dan_lambda_10/best.pt `
  --target-labels-released
python -m task2.evaluation.class_analysis
python -m task2.evaluation.confusion_summary
python -m task2.plot_results
python -m task2.evaluation.report_evidence
```

See `task2/README.md` for the objectives and label-leakage safeguards.

## Task 3: Target-free domain generalization

Task 3 reuses Task 2 Source-only as ERM. Do not load Sketch during Task 3
training, diagnostics, checkpoint selection, or hyperparameter selection.

```bash
python -m task3.train --config task3/configs/dan_dg.yaml
python -m task3.train --config task3/configs/sam.yaml
python -m task3.train --config task3/configs/dan_dg_lambda_0_1.yaml
python -m task3.train --config task3/configs/dan_dg_lambda_10.yaml
```

After all source-only choices are final:

```powershell
python -m task3.evaluate_final `
  --checkpoint erm=task2/results/source_only/best.pt `
  --checkpoint dan_dg=task3/results/dan_dg/best.pt `
  --checkpoint sam=task3/results/sam/best.pt `
  --checkpoint dan_dg_lambda_0_1=task3/results/dan_dg_lambda_0_1/best.pt `
  --checkpoint dan_dg_lambda_10=task3/results/dan_dg_lambda_10/best.pt `
  --sketch-labels-released
python -m task3.evaluation.class_analysis
python -m task3.plot_results
```

Run `task3.evaluation.report_evidence` with the matching Task 2/3 ERM hashes
as described in `task3/README.md` to create final curves, failure summaries,
and the checkpoint-integrity record.

## Task 4: Open-set recognition

Task 4 trains only on CIFAR-10. CIFAR-100 unknowns are accessed only after
checkpoints, scores, and validation-calibrated thresholds are fixed.

```bash
python -m task4.train --config task4/configs/vanilla.yaml
python -m task4.train --config task4/configs/gcsc.yaml
python -m task4.proser --vanilla-checkpoint task4/results/vanilla/best.pt
python -m task4.evaluate \
  --checkpoint vanilla=task4/results/vanilla/best.pt \
  --checkpoint gcsc=task4/results/gcsc/best.pt \
  --checkpoint proser=task4/results/proser/best.pt
python -m task4.report_evidence
```

`task4/results/final_evaluation/` contains post-hoc and trained-model score
tables, ROC/training figures, accepted-unknown failures, and the per-class MLS
acceptance summary. The optional RPL extension was not implemented.

## Results and repository hygiene

Small CSV, JSON, and PNG report artifacts are tracked. `.gitignore` excludes
raw `data/`, caches, generated cue-conflict images, virtual environments, and
all model checkpoints (`*.pt`, `*.pth`, `*.ckpt`). Before committing, review:

```bash
git status --short
git diff --cached --stat
```

Never add raw datasets, downloaded pretrained weights, or unnecessary large
checkpoints. The report source is intentionally separate from reproducibility
code and results.

# Task 2: PACS unsupervised domain adaptation

## Dataset placement

Download and extract the original PACS dataset so that the four domain folders are located at:

```text
data/pacs/PACS_Original/
  art_painting/
  cartoon/
  photo/
  sketch/
```

The data directory is ignored by Git. Generate the fixed protocol once after the dataset is in place:

```powershell
python -m shared.make_pacs_protocol --data-root data/pacs/PACS_Original
```

The manifest makes an 80/20 stratified split within each labeled source domain using seed 6304. It records only Sketch image paths, so target labels cannot enter adaptation or model selection by accident.

## Method configurations

`configs/base.yaml` contains the common assignment protocol. The method files only specify the method-specific settings:

- `source_only.yaml`: domain-balanced source ERM.
- `dan.yaml`: feature-level MMD with the specified three RBF widths.
- `dann.yaml`: gradient-reversal domain adversarial training.
- `cdan.yaml`: class-conditioned domain adversarial training.

Run any method by supplying its configuration:

```powershell
python -m task2.train --config task2/configs/dan.yaml
python -m task2.train --config task2/configs/dann.yaml
python -m task2.train --config task2/configs/cdan.yaml
```

For the controlled DAN study, run the two additional configurations. All settings other than `mmd_lambda` remain unchanged:

```powershell
python -m task2.train --config task2/configs/dan_lambda_0_1.yaml
python -m task2.train --config task2/configs/dan_lambda_10.yaml
```

## First run: source-only baseline

```powershell
python -m task2.train --config task2/configs/source_only.yaml
```

This uses the same 24 source images and 24 Sketch images per update required for every method. The Sketch batch is intentionally loaded even for source-only ERM so that the data schedule remains fixed when adaptation losses are introduced.

## Final evaluation

Only after every method, checkpoint, and controlled-study setting is fixed, run final target evaluation. This command releases Sketch labels for metrics, per-class accuracy, confusion matrices, and the label-free domain-separability diagnostic:

```powershell
python -m task2.evaluate_final `
  --checkpoint source_only=task2/results/source_only/best.pt `
  --checkpoint dan=task2/results/dan/best.pt `
  --checkpoint dann=task2/results/dann/best.pt `
  --checkpoint cdan=task2/results/cdan/best.pt `
  --checkpoint dan_lambda_0_1=task2/results/dan_lambda_0_1/best.pt `
  --checkpoint dan_lambda_10=task2/results/dan_lambda_10/best.pt `
  --target-labels-released
```

Then create the per-class changes relative to Source-only:

```powershell
python -m task2.evaluation.class_analysis
python -m task2.evaluation.confusion_summary
python -m task2.plot_results
python -m task2.evaluation.report_evidence
```

These commands write CSV summaries plus `main_method_comparison.png` and
`dan_lambda_study.png` in `task2/results/final_evaluation/`. The final command
also writes `training_curves.png` and deterministic selected per-class failure
cases in `selected_failure_cases.csv`.

## Adversarial optimization details

For DANN and CDAN, the 512-dimensional feature passed to the domain
discriminator is L2-normalized and the full model gradient norm is clipped to
0.1. The classifier still receives the unnormalized feature. These settings are
recorded in the corresponding method configurations.

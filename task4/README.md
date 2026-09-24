# Task 4: Open-set recognition

This task uses the required CIFAR-specific ResNet-18 and a fixed stratified
CIFAR-10 90/10 split (`seed=6304`).  CIFAR-100 is never loaded by the training
scripts: it is evaluation-only and is first read after all checkpoints and
scores are fixed.

```powershell
python -m task4.train --config task4/configs/vanilla.yaml
python -m task4.train --config task4/configs/gcsc.yaml
```

PROSER uses five dummy classifiers and layer-2 manifold mixup, initialized from
the selected Vanilla checkpoint:

```powershell
python -m task4.proser --vanilla-checkpoint task4/results/vanilla/best.pt
```

Each stage writes a checkpoint selected by CIFAR-10 validation accuracy and a
small `history.csv`. Checkpoints are intentionally ignored by Git; commit the
split manifest, histories, final CSVs, and figures instead.

After all three checkpoints are frozen, run the evaluation once. It caches
frozen outputs, evaluates Vanilla with MSP/MLS/Energy/Mahalanobis, evaluates
GCSC with MLS, and evaluates PROSER with MLS and its placeholder score.

```powershell
python -m task4.evaluate `
  --checkpoint vanilla=task4/results/vanilla/best.pt `
  --checkpoint gcsc=task4/results/gcsc/best.pt `
  --checkpoint proser=task4/results/proser/best.pt
python -m task4.report_evidence
```

The final command produces `task4/results/final_evaluation/`:

- `osr_metrics.csv`: all model/score/near/far/all-unknown measurements;
- `vanilla_posthoc_score_comparison.csv` and `trained_model_comparison.csv`;
- `vanilla_posthoc_roc.png` and `training_curves.png`;
- `vanilla_mls_accepted_failures.csv` with three near and three far accepted
  unknowns.
- `vanilla_mls_per_class_acceptance.csv` with the fixed-threshold acceptance
  rate and dominant forced known label for every near and far unknown class.

The optional RPL extension was intentionally not implemented.

# Task 3: PACS domain generalization

Task 3 uses the PACS source split fixed in Task 2. Photo, Art Painting, and
Cartoon are the only domains available to Task 3 training, model selection, and
source-side diagnostics. Sketch is loaded only by the final evaluation command.

## Methods

- ERM reuses `task2/results/source_only/best.pt` unchanged.
- DAN-DG aligns every pair of observed source-domain feature batches with the
  same multi-kernel MMD used by Task 2 DAN.
- SAM uses a non-adaptive perturbation radius of 0.05 and two gradient passes.

Train DAN-DG, SAM, and the controlled λ study with source data only:

```powershell
python -m task3.train --config task3/configs/dan_dg.yaml
python -m task3.train --config task3/configs/sam.yaml
python -m task3.train --config task3/configs/dan_dg_lambda_0_1.yaml
python -m task3.train --config task3/configs/dan_dg_lambda_10.yaml
```

After every setting and checkpoint has been fixed, release Sketch labels exactly
once for final evaluation:

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

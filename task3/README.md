# Task 3: PACS domain generalization

Task 3 uses the PACS source split fixed in Task 2. Photo, Art Painting, and
Cartoon are the only domains available to Task 3 training, model selection, and
source-side diagnostics. Sketch is loaded only by the final evaluation command.

## Methods

- ERM reuses `task2/results/source_only/best.pt` unchanged.
- DAN-DG aligns every pair of observed source-domain feature batches with the
  same multi-kernel MMD used by Task 2 DAN.
- SAM uses a non-adaptive perturbation radius of 0.05 and two gradient passes.

## ERM checkpoint recovery

The normal protocol reuses `task2/results/source_only/best.pt` unchanged. If
that local checkpoint is genuinely unavailable, recover it once with the locked
Task 2 Source-only configuration:

```powershell
python -m task3.recover_erm --config task2/configs/source_only.yaml
```

This recovery uses Task 3's source-only loaders so that it does not construct
or load Sketch batches; it writes the checkpoint to the required Task 2 path.

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

After extracting the archived Task 2 ERM hash and the local reused ERM hash,
create the evidence figures and final-analysis tables with:

```powershell
python -m task3.evaluation.report_evidence `
  --task2-erm-sha256 <task2-source-only-sha256> `
  --task3-erm-sha256 <task3-erm-sha256>
```

This writes training curves, deterministic selected failure cases, dominant
Sketch confusions, and an integrity-labelled Task 2 DAN versus Task 3 DAN-DG
comparison. The comparison is marked ready only when both tasks use the same
ERM checkpoint hash.

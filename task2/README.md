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

Implementation and experiment commands will be added alongside each method.

## First run: source-only baseline

```powershell
python -m task2.train --config task2/configs/source_only.yaml
```

This uses the same 24 source images and 24 Sketch images per update required for every method. The Sketch batch is intentionally loaded even for source-only ERM so that the data schedule remains fixed when adaptation losses are introduced.

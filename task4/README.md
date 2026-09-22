# Task 4: Open-set recognition

The first phase fixes the CIFAR-10 90/10 split and implements the required
CIFAR-specific ResNet-18, Vanilla, and GCSC training recipes. CIFAR-100 is not
loaded by `task4.train`, preventing unknown-data leakage during selection.

```powershell
python -m task4.train --config task4/configs/vanilla.yaml
python -m task4.train --config task4/configs/gcsc.yaml
```

Next phases add frozen-output extraction, post-hoc scores, PROSER, and the
final evaluation-only CIFAR-100 protocol.

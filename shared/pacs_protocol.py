"""The fixed data protocol shared by Tasks 2 and 3."""

from __future__ import annotations

import json
from pathlib import Path

from sklearn.model_selection import train_test_split

from shared.pacs import PACS_CLASSES, discover_domain_paths, discover_domain_samples, require_pacs_root


SOURCE_DOMAINS = ("photo", "art_painting", "cartoon")
TARGET_DOMAIN = "sketch"
PROTOCOL_VERSION = 1


def make_pacs_protocol(root: str | Path, seed: int = 6304) -> dict[str, object]:
    """Create the prescribed source splits and an unlabeled target listing."""
    root = require_pacs_root(root)
    source_splits: dict[str, dict[str, list[dict[str, object]]]] = {}

    for domain in SOURCE_DOMAINS:
        samples = discover_domain_samples(root, domain)
        labels = [int(sample["label"]) for sample in samples]
        train, validation = train_test_split(
            samples,
            test_size=0.20,
            random_state=seed,
            stratify=labels,
        )
        source_splits[domain] = {
            "train": sorted(train, key=lambda item: str(item["path"])),
            "validation": sorted(validation, key=lambda item: str(item["path"])),
        }

    return {
        "version": PROTOCOL_VERSION,
        "seed": seed,
        "classes": list(PACS_CLASSES),
        "source_domains": list(SOURCE_DOMAINS),
        "target_domain": TARGET_DOMAIN,
        "source": source_splits,
        # Deliberately paths only: target labels are not part of adaptation data.
        "target": {"paths": discover_domain_paths(root, TARGET_DOMAIN)},
    }


def save_pacs_protocol(protocol: dict[str, object], output_path: str | Path) -> Path:
    """Write the protocol manifest in a reviewable, deterministic format."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    return output_path


def load_pacs_protocol(path: str | Path) -> dict[str, object]:
    """Load and minimally validate a previously fixed PACS protocol."""
    with Path(path).open(encoding="utf-8") as handle:
        protocol = json.load(handle)
    if protocol.get("version") != PROTOCOL_VERSION:
        raise ValueError(f"Unsupported PACS protocol version: {protocol.get('version')}")
    if protocol.get("seed") != 6304:
        raise ValueError("This assignment requires the PACS seed to be 6304.")
    return protocol

"""Create the fixed PACS source split and unlabeled Sketch manifest."""

from __future__ import annotations

import argparse

from shared.pacs_protocol import make_pacs_protocol, save_pacs_protocol


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument(
        "--output",
        default="shared/splits/pacs_sketch_seed6304.json",
    )
    args = parser.parse_args()

    protocol = make_pacs_protocol(args.data_root)
    output = save_pacs_protocol(protocol, args.output)
    print(f"Saved fixed PACS protocol to {output}")


if __name__ == "__main__":
    main()

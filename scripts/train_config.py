import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from zshdrtv.train import run_training


def main() -> None:
    parser = argparse.ArgumentParser(description="Run training from a YAML config.")
    parser.add_argument("--config", required=True, help="Path to the training YAML config.")
    args = parser.parse_args()
    run_training(args.config)


if __name__ == "__main__":
    main()

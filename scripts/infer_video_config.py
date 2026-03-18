import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from zshdrtv.infer import infer_video_sequence


def main() -> None:
    parser = argparse.ArgumentParser(description="Run video inference from a YAML config.")
    parser.add_argument("--config", required=True, help="Path to the inference YAML config.")
    args = parser.parse_args()
    infer_video_sequence(args.config)


if __name__ == "__main__":
    main()

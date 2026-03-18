import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from zshdrtv.train import run_training


if __name__ == "__main__":
    run_training(str(ROOT / "configs" / "train_stage1_image.yaml"))

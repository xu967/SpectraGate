# ============================================================
# SpectraGate: 验证入口
# 用法:
#   python valMM.py --weights runs/mm_train/xxx/weights/best.pt --data data_drone_mm.yaml
# ============================================================
import argparse
from pathlib import Path

from ultralytics import YOLOMM

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description="SpectraGate validation")
    parser.add_argument("--weights", type=str, required=True, help="trained weights (.pt)")
    parser.add_argument("--data", type=str, default=str(ROOT / "data_drone_mm.yaml"))
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--name", type=str, default="val")
    args = parser.parse_args()

    model = YOLOMM(args.weights)
    model.val(
        data=args.data,
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(ROOT / "runs/val"),
        name=args.name,
        exist_ok=True,
    )


if __name__ == "__main__":
    main()

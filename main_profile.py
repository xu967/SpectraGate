# ============================================================
# SpectraGate: 模型结构与计算量分析
# 用法:
#   python main_profile.py
# ============================================================
import argparse
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

from ultralytics import YOLOMM

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description="SpectraGate model profiling")
    parser.add_argument("--model", type=str,
                        default=str(ROOT / "ultralytics/cfg/models/mm/11/yolo11n-mm-mbsg-cm-cascade-V5.yaml"))
    parser.add_argument("--scale", type=str, default="m")
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    model = YOLOMM(args.model)
    model.info(detailed=True)
    try:
        model.profile(imgsz=[args.imgsz, args.imgsz])
    except Exception as e:  # noqa: BLE001
        print(e)
    model.fuse()


if __name__ == "__main__":
    main()

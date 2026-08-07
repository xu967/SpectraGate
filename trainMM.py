# ============================================================
# SpectraGate: 可见光-红外双模态目标检测训练入口
# 论文: SpectraGate: Learnable Band Decomposition and
#       State-Regularized Fusion for Visible-Infrared Object Detection
# 用法示例:
#   python trainMM.py --data data_drone_mm.yaml --epochs 200 --batch 8 --name spectragate-drone
#   python trainMM.py --resume runs/mm_train/xxx/weights/last.pt
# ============================================================
import argparse
import os
from pathlib import Path

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from ultralytics import YOLOMM

ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "ultralytics/cfg/models/mm/11/yolo11n-mm-mbsg-cm-cascade-V5.yaml"


def main():
    parser = argparse.ArgumentParser(description="SpectraGate visible-infrared detection training")
    parser.add_argument("--model", type=str, default=str(DEFAULT_MODEL),
                        help="model yaml (default: SpectraGate V5 cascade yaml)")
    parser.add_argument("--data", type=str, default=str(ROOT / "data_drone_mm.yaml"),
                        help="dataset yaml with rgb/ir modality config")
    parser.add_argument("--scale", type=str, default="m", choices=["n", "s", "m", "l", "x"],
                        help="model scale (paper uses m)")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--name", type=str, default="spectragate")
    parser.add_argument("--optimizer", type=str, default="SGD")
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--amp", action="store_true", help="enable AMP")
    parser.add_argument("--resume", type=str, default=None,
                        help="resume training from a weights path")
    parser.add_argument("--fixed-cutoff", type=float, default=None,
                        help="fix the learnable band cutoff to a constant (ablation)")
    args = parser.parse_args()

    model = YOLOMM(args.resume or args.model)

    train_kwargs = dict(
        data=args.data,
        model_scale=args.scale,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        cache="true",
        device=args.device,
        workers=args.workers,
        optimizer=args.optimizer,
        amp=args.amp,
        exist_ok=True,
        project=str(ROOT / "runs/mm_train"),
        name=args.name,
        patience=args.patience,
        resume=args.resume is not None,
        # ---- FSCL: SpectraGate 融合状态约束损失 ----
        loss_fusion_v2=True,
        fusion_spectral_weight=0.03,
        fusion_gate_weight=0.005,
        fusion_cutoff_weight=0.001,
        fusion_target_types=["LFEM_D_V5", "NLFEM_D_V5", "FDFTM_V6"],
    )
    if args.fixed_cutoff is not None:
        train_kwargs["fixed_cutoff"] = args.fixed_cutoff

    model.train(**train_kwargs)


if __name__ == "__main__":
    main()

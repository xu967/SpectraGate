# ============================================================
# SpectraGate: 特征可视化入口（特征图 / Grad-CAM 热力图）
# 用法:
#   python visMM.py --weights best.pt --rgb img_rgb.png --ir img_ir.png \
#       --method feature --layers 22 23 24
# ============================================================
import argparse

from spectragate import YOLOMM


def main():
    parser = argparse.ArgumentParser(description="SpectraGate feature visualization")
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--rgb", type=str, required=True, help="RGB image path")
    parser.add_argument("--ir", type=str, default=None, help="IR image path")
    parser.add_argument("--method", type=str, default="feature", choices=["feature", "gradcam"])
    parser.add_argument("--layers", type=int, nargs="+", default=[22, 23, 24],
                        help="model layer indices (V5 cascade: 22=LMSG, 23=CPSE, 24=CCSF at P3)")
    parser.add_argument("--overlay", type=str, default="rgb", choices=["rgb", "x", "dual"])
    parser.add_argument("--alg", type=str, default="gradcam")
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--name", type=str, default="vis")
    args = parser.parse_args()

    model = YOLOMM(args.weights)
    model.vis(
        rgb_source=args.rgb,
        x_source=args.ir,
        method=args.method,
        layers=args.layers,
        overlay=args.overlay,
        alg=args.alg,
        save=args.save,
        project="runs/vis",
        name=args.name,
    )


if __name__ == "__main__":
    main()

# ============================================================
# SpectraGate: 双模态推理入口
# 用法:
#   python predictMM.py --weights best.pt --rgb img_rgb.png --ir img_ir.png --save
#   python predictMM.py --weights best.pt --rgb ./rgb/ --ir ./ir/ --save
# 单模态消融: 不传 --ir 时红外模态用零填充（反之亦然）
# ============================================================
import argparse

from spectragate import YOLOMM


def main():
    parser = argparse.ArgumentParser(description="SpectraGate visible-infrared inference")
    parser.add_argument("--weights", type=str, required=True, help="trained weights (.pt)")
    parser.add_argument("--rgb", type=str, required=True, help="RGB image path or directory")
    parser.add_argument("--ir", type=str, default=None, help="IR image path or directory")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--save", action="store_true", help="save visualized results")
    parser.add_argument("--name", type=str, default="predict")
    args = parser.parse_args()

    model = YOLOMM(args.weights)
    model.predict(
        rgb_source=args.rgb,
        x_source=args.ir,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        save=args.save,
        project="runs",
        name=args.name,
        exist_ok=True,
    )


if __name__ == "__main__":
    main()

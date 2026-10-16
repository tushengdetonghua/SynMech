from argparse import ArgumentParser

from ultralytics import YOLO


def parse_args():
    parser = ArgumentParser(description="Evaluate a YOLO11 instance-segmentation model.")
    parser.add_argument("--model", required=True, help="Trained .pt weight file.")
    parser.add_argument("--data", required=True, help="Dataset YAML file.")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--project", default="runs/val")
    parser.add_argument("--name", default="synmech_eval")
    return parser.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.model)
    model.val(
        data=args.data,
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
    )


if __name__ == "__main__":
    main()

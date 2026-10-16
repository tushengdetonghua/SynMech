from argparse import ArgumentParser

from ultralytics import YOLO


def parse_args():
    parser = ArgumentParser(description="Train a YOLO11 instance-segmentation model.")
    parser.add_argument("--model", required=True, help="Model YAML or pretrained .pt file.")
    parser.add_argument("--data", required=True, help="Dataset YAML file.")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--optimizer", default="SGD")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--close-mosaic", type=int, default=10)
    parser.add_argument("--project", default="runs/train")
    parser.add_argument("--name", default="synmech")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.model)
    model.train(
        data=args.data,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        optimizer=args.optimizer,
        workers=args.workers,
        device=args.device,
        close_mosaic=args.close_mosaic,
        seed=args.seed,
        deterministic=True,
        cache=False,
        amp=True,
        project=args.project,
        name=args.name,
    )


if __name__ == "__main__":
    main()

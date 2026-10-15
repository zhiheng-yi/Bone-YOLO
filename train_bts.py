from ultralytics import YOLO


def main():
    model = YOLO('yolo11n.pt')
    model.train(
        data='./data.yaml',
        epochs=100,
        imgsz=640,
        batch=16,
        device=0,
        patience=30,
        seed=0,
        deterministic=True,
        amp=True,
        workers=4,
        optimizer='SGD',
        lr0=0.01,
        cos_lr=True,
        close_mosaic=30,
        scale=0.9,
        project='./runs',
        name='bone_yolo',
        exist_ok=True,
    )


if __name__ == '__main__':
    main()
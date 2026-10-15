import argparse
import os
from ultralytics import YOLO


def evaluate(weights, use_tta=False):
    model = YOLO(weights)
    metrics = model.val(
        data='./data.yaml',
        split='test',
        imgsz=640,
        batch=16,
        device=0,
        augment=use_tta,
        verbose=True,
    )
    tag = os.path.basename(weights)
    if use_tta:
        tag += ' (with TTA)'
    print(f'\n===== {tag} =====')
    print(f'mAP50    : {metrics.box.map50:.4f}')
    print(f'mAP50-95 : {metrics.box.map:.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='./weights/bone_yolo_best.pt',
                        help='Path to model weights')
    parser.add_argument('--tta', action='store_true',
                        help='Enable test-time augmentation')
    args = parser.parse_args()
    evaluate(args.weights, use_tta=args.tta)
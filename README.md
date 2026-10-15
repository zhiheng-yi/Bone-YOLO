# Bone-YOLO

Official implementation of **Bone-YOLO: Training Strategy Optimization and Test-Time Augmentation for Archaeological Human Bone Detection**.

## Overview

This repository provides the training and evaluation code for Bone-YOLO, which combines a parameter-efficient training strategy (BTS) and test-time augmentation (TTA) for archaeological human bone detection using YOLO11n.

- **BTS**: SGD optimizer, cosine learning rate scheduling, close mosaic (last 30 epochs), scale augmentation (0.9).
- **TTA**: horizontal flipping and multi-scale testing at inference.

**Main results** (3 random seeds, mAP50-95):

| Model | mAP50-95 |
|---|---|
| Baseline (YOLOv11n) | 80.4% ± 0.9% |
| TTA-only | 81.7% ± 0.8% |
| BTS-only | 81.6% ± 1.4% |
| Bone-YOLO (BTS + TTA) | 81.7% ± 1.4% |

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Ultralytics 8.4.174

Install dependencies:

```bash
pip install -r requirements.txt
```

## Dataset

The dataset contains 1,604 archaeological human bone images with 18 categories. Due to copyright restrictions, the dataset is **not publicly available**. Researchers may request access from the corresponding author (weipianpian@fudan.edu.cn).

After obtaining the dataset, organize it as follows:

```
dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

Create a `data.yaml` file in the root directory with the following content:

```yaml
path: ./dataset
train: images/train
val: images/val
test: images/test

nc: 18
names:
  0: Skull
  1: Mandible
  2: Clavicle
  3: Scapula
  4: Humerus
  5: Radius
  6: Ulna
  7: Hand_bone
  8: Hip_bone
  9: Femur
  10: Patella
  11: Tibia
  12: Fibula
  13: Foot_bone
  14: Vertebra
  15: Rib
  16: Sacrum
  17: Sternum
```

## Training

To train Bone-YOLO with BTS from COCO-pretrained YOLO11n weights (default: 100 epochs, batch 16, image size 640):

```bash
python train_bts.py
```

For multi-seed validation, run the script with different seeds (modify the seed in the script or pass via argument if supported).

## Evaluation

Evaluate the baseline or the trained Bone-YOLO model on the test set:

```bash
# Baseline without TTA
python eval_tta.py --weights ./weights/baseline_best.pt

# Bone-YOLO with TTA
python eval_tta.py --weights ./weights/bone_yolo_best.pt --tta
```

## Model Weights

The following weights are provided in the `weights/` directory:

- `bone_yolo_best.pt`: Bone-YOLO trained with BTS (seed=0).
- `baseline_best.pt`: Baseline YOLO11n (seed=0).

## License

This project is licensed under the **AGPL-3.0 License**. See the [LICENSE](LICENSE) file for details.


## Acknowledgments

This work uses the Ultralytics YOLO11 framework.

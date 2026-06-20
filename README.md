# FGCRL: Foreground-Guided Contrastive Representation Learning

This repository provides the implementation of **FGCRL**, a foreground-guided self-supervised representation learning framework for diseased leaf detection in complex natural scenes.

FGCRL is designed to improve the transferability of visual representations learned from unlabeled diseased leaf images to downstream YOLO-based detection tasks. Instead of relying only on image-level contrastive learning, FGCRL introduces foreground priors, position-aware representation learning, and raw foreground-prior distillation to encourage the backbone to focus on disease-related foreground regions.

---

## Introduction

Deep learning-based plant disease detection often suffers from complex natural backgrounds, small lesion regions, occlusion, illumination variation, and limited annotated data. Although YOLO-based detectors can achieve efficient object detection, their performance is highly dependent on the quality of the learned visual representations, especially when disease symptoms are small or visually ambiguous.

To address this issue, FGCRL introduces a two-stage learning framework:

1. **Stage 1: Foreground-guided self-supervised pretraining**
   The detector backbone is pretrained using unlabeled diseased leaf images and foreground priors. This stage aims to learn foreground-sensitive and disease-relevant representations.

2. **Stage 2: Downstream disease detection**
   The pretrained backbone is transferred to a YOLO-based detector and fine-tuned on labeled disease detection datasets.

The goal of FGCRL is not to replace the detector head, but to improve the representation quality of the backbone before supervised detection.

---

## Framework

The overall FGCRL framework consists of the following components:

* **Foreground-guided dual views**
  Two augmented views are generated from each unlabeled image together with its corresponding foreground prior mask.

* **Position-prior representation module**
  A position-aware representation module is introduced to encode spatial foreground information into multi-scale backbone features.

* **Snapshot teacher module**
  A momentum-updated teacher network provides stable representation targets during self-supervised pretraining.

* **Raw foreground-prior distillation**
  Multi-scale position-aware features from the online encoder are used as teacher targets. A Smooth L1 objective is applied to guide the student branch toward foreground-aware representations.

* **YOLO-based downstream detection**
  The pretrained backbone weights are transferred to YOLO detectors for supervised disease detection.

The framework follows the idea that disease-aware representations should be learned before the detector is trained, so that the downstream detector can better localize small and weak disease regions in natural orchard scenes.

---

## Dataset

FGCRL is evaluated on both public plant disease datasets and real-world durian leaf disease datasets.

### Public datasets

The public datasets are used to validate the transferability of the learned representation:

| Dataset                       | Task Type                                    | Usage                |
| ----------------------------- | -------------------------------------------- | -------------------- |
| PlantDoc                      | Single-class / multi-class disease detection | Transfer validation  |
| PlantSeg                      | Single-class / multi-class disease detection | Transfer validation  |
| Public durian disease dataset | Durian leaf disease detection                | In-domain validation |

### Private durian leaf dataset

A private real-orchard durian leaf dataset is also used to evaluate disease detection under realistic field conditions.

The dataset is organized into three subsets:

| Subset | Source            | Annotation             | Usage                               |
| ------ | ----------------- | ---------------------- | ----------------------------------- |
| Set_A  | Multiple orchards | Unlabeled              | Stage-1 self-supervised pretraining |
| Set_B  | Training orchards | Labeled bounding boxes | Stage-2 supervised training         |
| Set_C  | Held-out orchard  | Labeled bounding boxes | Cross-orchard testing               |

The held-out orchard setting is used to evaluate whether the learned representation can generalize to unseen orchard environments.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/FGCRL.git
cd FGCRL
```

### 2. Create the environment

```bash
conda create -n fgcrl python=3.9
conda activate fgcrl
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

The implementation is based on PyTorch and YOLO-style detection training.

Recommended environment:

```text
Python >= 3.8
PyTorch >= 1.12
CUDA >= 11.3
Ultralytics YOLO framework
```

---

## Pretraining

Stage-1 pretraining aims to learn foreground-sensitive representations from unlabeled leaf images.

### Configuration

The main FGCRL pretraining configuration includes:

```yaml
backbone: yolov8n
layer_indices: [4, 6, 8]

contrastive:
  queue_size: 4096
  momentum: 0.999
  temperature: 0.2
  proj_dim: 256

position_prior:
  pos_pe_channels: 64
  pos_pe_spans: [2, 1, 1]
  pos_init_scales: [0.30, 0.45, 0.60]

raw_prior_distillation:
  enabled: true
  loss_type: smooth_l1
  weight: 0.15
  start_epoch: 6
  scale_weights: [0.30, 0.40, 0.30]
  student_temperature: 0.10
  teacher_temperature: 0.07
```

### Run pretraining

```bash
python train_stage1_fgcrl.py \
  --config configs/stage1_fgcrl.yaml \
  --data data/Set_A
```

After pretraining, the learned backbone weights will be saved to:

```text
runs/stage1_fgcrl/weights/best.pt
```

---

## Detection Fine-tuning

Stage-2 transfers the pretrained FGCRL backbone to a YOLO-based detector.

### Run downstream detection training

```bash
python train_stage2_detection.py \
  --weights runs/stage1_fgcrl/weights/best.pt \
  --data configs/durian_detection.yaml \
  --imgsz 640 \
  --epochs 100 \
  --batch 16
```

### Training details

The downstream detector is trained using the following settings:

```yaml
epochs: 100
batch_size: 16
image_size: 640
optimizer: AdamW
initial_learning_rate: 0.001
final_lr_factor: 0.01
momentum: 0.937
weight_decay: 0.0005
augmentation: randaugment
```

For detector-coupled fine-tuning, differential learning rates can be used to preserve the foreground-sensitive representations learned during Stage-1:

```yaml
backbone_lr: 0.0001
neck_lr: 0.0005
head_lr: 0.001
```

---

## Evaluation

The detection performance is evaluated using standard object detection metrics:

| Metric       | Description                                                                    |
| ------------ | ------------------------------------------------------------------------------ |
| Precision    | Ratio of correctly predicted disease instances among all predicted instances   |
| Recall       | Ratio of correctly detected disease instances among all ground-truth instances |
| mAP@0.5      | Mean average precision at IoU threshold 0.5                                    |
| mAP@0.5:0.95 | Mean average precision averaged over IoU thresholds from 0.5 to 0.95           |
| F1-score     | Harmonic mean of precision and recall                                          |

For small lesion detection, additional size-stratified analysis can be conducted:

| Lesion Size | Description                                |
| ----------- | ------------------------------------------ |
| Small       | Small disease spots or weak lesion regions |
| Medium      | Moderate-sized disease regions             |
| Large       | Clearly visible disease regions            |

Run evaluation:

```bash
python val.py \
  --weights runs/stage2_detection/weights/best.pt \
  --data configs/durian_detection.yaml \
  --imgsz 640
```

---

## Results

### Overall detection performance

| Method        | Pretraining                                          | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
| ------------- | ---------------------------------------------------- | --------: | -----: | ------: | -----------: |
| YOLO baseline | Random initialization                                |         - |      - |       - |            - |
| YOLO + SSL    | Self-supervised pretraining without foreground prior |         - |      - |       - |            - |
| YOLO + FGCRL  | Foreground-guided pretraining                        |         - |      - |       - |            - |

### Transfer validation

| Dataset        | Detector | Baseline mAP@0.5 | FGCRL mAP@0.5 | Improvement |
| -------------- | -------- | ---------------: | ------------: | ----------: |
| PlantDoc       | YOLOv8n  |                - |             - |           - |
| PlantSeg       | YOLOv8n  |                - |             - |           - |
| Durian dataset | YOLOv8n  |                - |             - |           - |

### Ablation study

| Variant           | Foreground Prior | Position Prior | RPD | mAP@0.5 |
| ----------------- | ---------------- | -------------- | --- | ------: |
| Baseline          | ✗                | ✗              | ✗   |       - |
| SSL only          | ✗                | ✗              | ✗   |       - |
| + Position prior  | ✗                | ✓              | ✗   |       - |
| + Foreground mask | ✓                | ✓              | ✗   |       - |
| Full FGCRL        | ✓                | ✓              | ✓   |       - |

Please replace the placeholder values with the final experimental results.

---

## Visualization

Visualization is used to analyze whether FGCRL improves foreground attention and lesion-related representation learning.

The following visualization methods are recommended:

* Feature heatmap visualization
* Foreground response map
* Class-wise detection examples
* TP / FP / FN case analysis
* Small-lesion detection examples

Example directory structure:

```text
assets/
├── framework.png
├── heatmap_comparison.png
├── detection_examples.png
└── failure_cases.png
```

To display figures in this README:

```markdown
![FGCRL framework](assets/framework.png)

![Detection examples](assets/detection_examples.png)
```

---

## Project Structure

```text
FGCRL/
├── configs/
│   ├── stage1_fgcrl.yaml
│   └── durian_detection.yaml
├── data/
│   ├── Set_A/
│   ├── Set_B/
│   └── Set_C/
├── models/
│   ├── fgcrl/
│   └── yolo/
├── scripts/
│   ├── train_stage1_fgcrl.py
│   ├── train_stage2_detection.py
│   └── evaluate.py
├── runs/
│   ├── stage1_fgcrl/
│   └── stage2_detection/
├── assets/
│   ├── framework.png
│   └── detection_examples.png
├── requirements.txt
└── README.md
```

---

## Citation

If you use this repository or find this work useful, please cite:

```bibtex
@article{liu2026fgcrl,
  title={Foreground-Guided Contrastive Representation Learning for Diseased Leaves in Complex Natural Scenes},
  author={Liu, Wenjuan and Mohamed, Ahmad Sufril Azlan and Osman, Mohd Azam and Kie, Kim Hwa and Wong, Chow Jeng},
  journal={To be updated},
  year={2026}
}
```

---

## License

This project is released for academic research purposes.

Please check the license file before using this code for commercial applications.

---

## Acknowledgements

This work is part of a research project on deep learning-based durian leaf disease detection in real-world orchard environments.

We thank the contributors and collaborators who supported dataset collection, annotation, experimental validation, and manuscript preparation.

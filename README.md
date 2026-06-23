**FGCRL: Foreground-Prior-Guided Self-Supervised Pretraining for Lesion-Sensitive Visual Representation Learning in Complex Natural Scenes**

FGCRL is a foreground-prior-guided self-supervised pretraining framework for learning lesion-sensitive visual representations from unlabeled diseased leaf images. Instead of treating foreground masks as lesion annotations, FGCRL uses SAM2-assisted coarse leaf masks as weak spatial priors to guide contrastive view construction, preserve foreground-supported local structures, and distil foreground-aware responses into the backbone. The learned backbone can then be transferred to YOLO-based detectors without modifying the downstream detector architecture.


## 1. Repository structure

```text
Disease_Representation_Learning/
├── configs/                         # Stage-1, Stage-2, and paper experiment configurations
├── data/                            # Dataset YAMLs, split files, or metadata when available
├── datasets/                        # Dataset loading code
├── losses/                          # FGCRL loss functions
├── models/                          # Stage-1 SSL model and YOLO-related modules
├── tools/                           # Dataset preparation, mask generation, evaluation, and analysis scripts
├── utils/                           # Checkpointing, config loading, seed control, and utility functions
├── runs_yolov8n/                    # YOLOv8n experiment outputs where available
├── runs_yolov9t/                    # YOLOv9t experiment outputs where available
├── runs_yolov10n/                   # YOLOv10n experiment outputs where available
├── runs_yolov11n/                   # YOLOv11n experiment outputs where available
├── runs_yolov8n_plantdoc_singleclass/
├── runs_yolov8n_plantdoc_multiclass/
├── runs_yolov8n_plantseg_singleclass/
├── runs_yolov8n_plantseg_multiclass/
├── environment.yml                  # Conda environment file for reproduction
├── environment_full.txt             # Full package list from the original Conda environment
├── train_stage1.py                  # Stage-1 foreground-prior-guided SSL pretraining
├── train_stage2.py                  # Stage-2 YOLO detector fine-tuning
├── LICENSE
└── README.md
```

---

## 2. Method overview

FGCRL follows a two-stage learning protocol.

### 2.1 Stage 1: foreground-prior-guided self-supervised pretraining

Given unlabeled diseased leaf images, SAM2-assisted coarse leaf masks are used as weak spatial priors. These priors are not lesion masks and are not used as supervised labels. They are used only during pretraining to:

1. construct foreground-aligned contrastive views;
2. preserve foreground-supported local structures;
3. stabilize prior-aware representation learning;
4. distil foreground-aware responses into the raw backbone.

After pretraining, the foreground priors, projection heads, momentum branch, and snapshot teacher are discarded. Only the learned online backbone is transferred to downstream detection.

### 2.2 Stage 2: YOLO-based downstream detection

The FGCRL-pretrained backbone is transferred to YOLO-based detectors and fine-tuned using standard supervised detection training. The downstream detector architecture is not modified in the main experiments.

---

## 3. Dataset preprocessing


The manuscript evaluates FGCRL on the following datasets. Users should download the public datasets from their official sources and follow the original licenses and usage restrictions.

| Dataset             | Annotation type    | Evaluation setting                                                                | Source / citation link                                                                                                              |
| ------------------- | ------------------ | --------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| Durian Leaf Disease | Bounding boxes     | In-domain detection                                                               | [Roboflow Universe](https://universe.roboflow.com/mintra/durian-leaf2)                                                              |
| PlantDoc            | Bounding boxes     | Cross-dataset single-class and multi-class detection                              | [paper DOI](https://doi.org/10.1145/3371158.3371196) |
| PlantSeg            | Segmentation masks | Cross-dataset single-class and multi-class detection after mask-to-box conversion | [dataset paper](https://www.nature.com/articles/s41597-025-06513-4)                                                                 |


### 4.2 PlantDoc preparation

PlantDoc annotations are converted into YOLO-compatible detection format. Two settings are supported:

* **Single-class setting:** all disease-region boxes are mapped to one class, `disease_region`.
* **Multi-class setting:** original disease categories are retained and remapped to continuous class IDs.

Example command:

```bash
python tools/build_plantdoc_datasets.py \
  --input_root /path/to/PlantDoc \
  --output_root data/plantdoc_yolo \
  --split_ratio 0.7 0.2 0.1 \
  --seed 42
```

If your local argument names differ, run:

```bash
python tools/build_plantdoc_datasets.py --help
```

### 4.3 PlantSeg preparation

PlantSeg provides pixel-level disease masks. For YOLO-based detection, each valid connected disease component is converted into a bounding box. The conversion follows the mask-to-box protocol used in the manuscript.

Example command:

```bash
python tools/build_plantseg_datasets.py \
  --input_root /path/to/PlantSeg \
  --output_root data/plantseg_yolo \
  --split_ratio 0.7 0.2 0.1 \
  --seed 42
```

The preprocessing protocol excludes:

* classes with fewer than 20 images;
* images containing more than 30 bounding boxes;
* records with missing, empty, or invalid annotation files;
* empty masks and degenerate connected components.

Filtering decisions should be recorded in audit files whenever possible.

### 4.4 Private orchard data

The private durian orchard images used for unlabeled pretraining and in-domain validation are not fully redistributed in this repository because of field-data ownership, farm collaboration agreements, and dataset-use restrictions. To improve reproducibility, we provide the most reproducible alternatives possible:

* preprocessing scripts;
* foreground-prior generation scripts;
* dataset format specifications;
* split construction protocol;
* configuration files;
* evaluation scripts;
* result folders and logs where available.

Researchers may reproduce the pipeline using their own unlabeled diseased leaf images and the same data organization described below.

### 4.5 Expected YOLO dataset format

```text
dataset_root/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
└── dataset.yaml
```

Example `dataset.yaml`:

```yaml
path: /absolute/path/to/dataset_root
train: images/train
val: images/val
test: images/test
names:
  0: disease_region
```

For multi-class detection, replace `names` with the retained disease categories.

---

## 5. Environment setup

The experiments were conducted in a Conda environment with Python 3.10, PyTorch 2.3.0, CUDA 11.8, and YOLO-related dependencies. The main experiments were run on an NVIDIA RTX 4090 GPU with 24 GB memory.

### 5.1 Clone the repository

```bash
git clone https://github.com/WendyCV/Disease_Representation_Learning.git
cd Disease_Representation_Learning
```

### 5.2 Create the Conda environment

We provide an `environment.yml` file for reproducing the software environment:

```bash
conda env create -f environment.yml
conda activate fgcrl
```

If dependency conflicts occur on a different CUDA or operating-system version, we recommend first creating a clean Python 3.10 environment and then installing PyTorch manually:

```bash
conda create -n fgcrl python=3.10 -y
conda activate fgcrl

pip install torch==2.3.0+cu118 torchvision==0.18.0+cu118 torchaudio==2.3.0+cu118 \
  --extra-index-url https://download.pytorch.org/whl/cu118

pip install albumentations==2.0.6 opencv-python-headless==4.10.0.84 \
  numpy==1.26.4 pandas==2.2.3 matplotlib==3.9.2 scikit-learn==1.6.1 \
  scikit-image==0.25.2 scipy==1.14.1 tqdm==4.66.5 pyyaml==6.0.2 \
  pillow==11.3.0 timm==1.0.15 transformers==4.50.3 hydra-core==1.3.2 \
  omegaconf==2.3.0 pytorch-metric-learning==2.9.0 torchmetrics==1.4.2 \
  grad-cam==1.5.4 pycocotools==2.0.8 roboflow==1.2.11 plantcv==4.6 \
  umap-learn==0.5.7 kneed==0.8.5 psutil==6.0.0 requests==2.32.3
```

### 5.3 Core software versions

The main environment used in our experiments includes:

| Package                 | Version      |
| ----------------------- | ------------ |
| Python                  | 3.10.15      |
| PyTorch                 | 2.3.0+cu118  |
| torchvision             | 0.18.0+cu118 |
| torchaudio              | 2.3.0+cu118  |
| NumPy                   | 1.26.4       |
| pandas                  | 2.2.3        |
| OpenCV                  | 4.10/4.12    |
| albumentations          | 2.0.6        |
| scikit-learn            | 1.6.1        |
| matplotlib              | 3.9.2        |
| timm                    | 1.0.15       |
| transformers            | 4.50.3       |
| grad-cam                | 1.5.4        |
| pytorch-metric-learning | 2.9.0        |
| pycocotools             | 2.0.8        |
| roboflow                | 1.2.11       |
| plantcv                 | 4.6          |

A full package list from the original Conda environment is provided in `environment_full.txt` for transparency.

---

## 6. Stage-1 FGCRL pretraining

Stage-1 pretraining learns foreground-sensitive representations from unlabeled leaf images and SAM2-assisted coarse foreground priors.

### 6.1 Prepare unlabeled images and foreground priors

Expected format:

```text
data/stage1/
├── images/
│   └── train/
└── masks/
    └── train/
```

The mask filename should correspond to the image filename unless otherwise specified in the config.

### 6.2 Generate SAM2-assisted foreground priors

If foreground masks are not available, generate them with:

```bash
python tools/generate_foreground_masks_with_qc.py \
  --image_root data/stage1/images/train \
  --output_root data/stage1/masks/train \
  --qc_output data/stage1/mask_qc
```

The generated masks are coarse foreground priors. They are not lesion annotations and should not be interpreted as ground-truth segmentation labels.

### 6.3 Run Stage-1 pretraining

```bash
python train_stage1.py --config configs/ssl_config.yaml
```

The key FGCRL settings include:

```yaml
raw_prior_distillation:
  enabled: true
  loss_type: smooth_l1
  weight: 0.15
  start_epoch: 6
  scale_weights: [0.30, 0.40, 0.30]
  student_temperature: 0.10
  teacher_temperature: 0.07
```

Training logs and checkpoints are saved under the configured run directory. Typical outputs include:

```text
train_log.csv
config_used.yaml
last.pth
best.pth
```

---

## 7. Stage-2 downstream YOLO detection

Stage-2 transfers the FGCRL-pretrained backbone to YOLO-based detectors.

### 7.1 Configure detector training

Edit `configs/det_config.yaml` and set:

```yaml
data:
  data_yaml: /path/to/dataset.yaml

stage1_init:
  enabled: true
  ckpt_path: /path/to/stage1/best.pth
```

### 7.2 Run detector fine-tuning

```bash
python train_stage2.py --config configs/det_config.yaml
```

Main training settings used in the manuscript:

```yaml
epochs: 100
batch_size: 16
image_size: 640
optimizer: AdamW
initial_learning_rate: 0.001
final_lr_factor: 0.01
momentum: 0.937
weight_decay: 0.0005
augmentation: RandAugment
```

The baseline and FGCRL-pretrained detectors should use the same detector configuration, input size, optimizer, augmentation, and evaluation procedure.

### 7.3 Checkpoint availability

Where technically feasible, pretrained checkpoints will be provided through GitHub Releases or Zenodo. If a checkpoint is not available, the corresponding `config_used.yaml`, training log, and reproduction command are provided so that users can retrain the model.

Recommended release structure:

```text
releases/
├── fgcrl_stage1_best.pth
├── yolov8n_fgcrl_durian_best.pt
├── yolov8n_fgcrl_plantdoc_singleclass_best.pt
├── yolov8n_fgcrl_plantdoc_multiclass_best.pt
├── yolov8n_fgcrl_plantseg_singleclass_best.pt
└── yolov8n_fgcrl_plantseg_multiclass_best.pt
```

---

## 8. Evaluation protocol

### 8.1 Standard detection metrics

The downstream detectors are evaluated using:

| Metric    | Meaning                                          |
| --------- | ------------------------------------------------ |
| Precision | Correct predictions among all predictions        |
| Recall    | Recovered GT objects among all GT objects        |
| mAP50     | Mean AP at IoU threshold 0.5                     |
| mAP50-95  | Mean AP averaged over IoU thresholds 0.5 to 0.95 |

### 8.2 Coarse-prior and feature-response analysis

The manuscript additionally reports diagnostic metrics to examine whether the learned backbone becomes more foreground- and lesion-aligned:

| Metric          | Purpose                                                        |
| --------------- | -------------------------------------------------------------- |
| Mask area ratio | Measures spatial extent of the coarse prior                    |
| BoxCov          | Measures lesion-box coverage by the coarse prior               |
| Cov50 / Cov70   | Proportion of GT boxes with prior coverage above 0.5 / 0.7     |
| Enrich          | Measures foreground enrichment relative to prior area          |
| BgCov           | Estimates background leakage                                   |
| Gap             | Difference between lesion-box coverage and background coverage |
| FG/BG           | Foreground/background feature-response ratio                   |
| R-GT            | GT/non-GT feature-response ratio                               |
| Top-20 FG lift  | Whether top-response regions concentrate in foreground         |
| IoU lift q70    | Overlap between high-response regions and foreground priors    |

Example scripts:

```bash
python tools/analyze_stage1_prior_foreground_alignment.py --help
python tools/analyze_stage1_mask_ratio.py --help
python tools/eval_best_models_on_split.py --help
```

---

## 9. Main results reported in the manuscript

### 9.1 Transfer across lightweight YOLO variants on Durian Leaf Disease

| Detector | Model            | Precision | Recall | mAP50 | mAP50-95 |
| -------- | ---------------- | --------: | -----: | ----: | -------: |
| YOLOv8n  | Baseline         |     91.41 |  91.45 | 92.89 |    78.49 |
| YOLOv8n  | FGCRL-pretrained |     95.24 |  91.77 | 93.18 |    77.86 |
| YOLOv9t  | Baseline         |     94.30 |  90.50 | 92.23 |    79.44 |
| YOLOv9t  | FGCRL-pretrained |     98.28 |  90.24 | 93.89 |    80.49 |
| YOLOv10n | Baseline         |     84.71 |  87.03 | 92.58 |    76.43 |
| YOLOv10n | FGCRL-pretrained |     87.91 |  91.30 | 93.22 |    78.02 |
| YOLOv11n | Baseline         |     93.28 |  90.42 | 93.06 |    78.84 |
| YOLOv11n | FGCRL-pretrained |     92.53 |  91.75 | 93.41 |    80.93 |

### 9.2 Cross-dataset transfer with YOLOv8n

| Dataset  | Setting      | Model            | Precision | Recall | mAP50 | mAP50-95 |
| -------- | ------------ | ---------------- | --------: | -----: | ----: | -------: |
| PlantDoc | Single-class | Baseline         |      76.2 |   66.9 |  74.2 |     45.1 |
| PlantDoc | Single-class | FGCRL-pretrained |      84.0 |   68.5 |  77.2 |     49.5 |
| PlantDoc | Multi-class  | Baseline         |      64.2 |   56.6 |  64.4 |     43.8 |
| PlantDoc | Multi-class  | FGCRL-pretrained |      71.6 |   67.5 |  70.5 |     47.0 |
| PlantSeg | Single-class | Baseline         |     47.70 |  43.54 | 42.42 |    22.67 |
| PlantSeg | Single-class | FGCRL-pretrained |     49.77 |  44.42 | 43.54 |    22.80 |
| PlantSeg | Multi-class  | Baseline         |     37.59 |  31.65 | 30.09 |    18.65 |
| PlantSeg | Multi-class  | FGCRL-pretrained |     35.49 |  34.95 | 31.24 |    19.33 |

The results should be interpreted together with the trade-off analysis in the manuscript. FGCRL improves mAP50 across all evaluated settings, but stricter localization and fine-grained category calibration remain challenging, especially for PlantSeg multi-class detection.

### 9.3 Ablation results

| Variant | FPG | PARM | RPD | FG/BG | Top-20 FG lift | IoU lift q70 | Precision | Recall | mAP50 | mAP50-95 |
| ------- | --- | ---- | --- | ----: | -------------: | -----------: | --------: | -----: | ----: | -------: |
| M0      | ✗   | ✗    | ✗   | 1.175 |          1.257 |        1.271 |     91.41 |  91.45 | 92.89 |    78.49 |
| M1      | ✓   | ✗    | ✗   | 1.177 |          1.229 |        1.265 |     91.52 |  91.68 | 92.60 |    78.55 |
| M2      | ✗   | ✓    | ✗   | 1.948 |          1.885 |        2.160 |     88.06 |  93.43 | 91.56 |    74.22 |
| M3      | ✓   | ✓    | ✗   | 2.052 |          2.010 |        2.303 |     93.55 |  91.48 | 92.86 |    77.74 |
| M4      | ✓   | ✓    | ✓   | 2.056 |          2.021 |        2.322 |     95.24 |  91.77 | 93.18 |    77.86 |

FPG denotes foreground-prior guidance, PARM denotes the prior-aware representation module, and RPD denotes raw prior distillation.

---

## 10. Reproduction checklist

To reproduce the main experimental pipeline:

1. Clone this repository.
2. Create the Conda environment using `environment.yml`.
3. Check `environment_full.txt` if a complete package record is needed.
4. Download the public datasets from their official sources.
5. Convert PlantDoc and PlantSeg into YOLO-compatible formats using the scripts in `tools/`.
6. Prepare unlabeled leaf images and generate SAM2-assisted coarse foreground priors.
7. Run Stage-1 FGCRL pretraining with `train_stage1.py`.
8. Transfer the pretrained backbone into a YOLO detector using `train_stage2.py`.
9. Evaluate the detector using the same split files and metrics.
10. Run prior-quality and feature-response analysis scripts for diagnostic validation.
11. Compare baseline and FGCRL-pretrained models under identical training settings.

---

## 11. Ethical and technical release limitations

Some data and models may not be fully released for ethical, privacy, copyright, or technical reasons.

| Item                                | Release status          | Reason / alternative                                        |
| ----------------------------------- | ----------------------- | ----------------------------------------------------------- |
| Public datasets                     | Not redistributed       | Users should download from official sources                 |
| Private orchard images              | Not publicly released   | Field-data ownership and collaboration agreements           |
| Raw annotations from private data   | Not fully redistributed | Dataset-use restrictions                                    |
| Processed metadata / split protocol | Provided where feasible | Enables reconstruction of the benchmark protocol            |
| Pretrained checkpoints              | Released where feasible | Large files may be hosted through GitHub Releases or Zenodo |
| Scripts and configs                 | Released                | Main reproducibility pathway                                |
| Conda environment                   | Released                | `environment.yml` and `environment_full.txt`                |

This repository aims to provide the most reproducible alternative possible when full raw data release is not feasible.

---

## 12. Permanent archive and DOI

A permanent archive will be created through GitHub Releases and Zenodo.

Current repository:

```text
https://github.com/WendyCV/Disease_Representation_Learning
```

Planned permanent archive:

```text
GitHub release: v1.0-tvc-submission
Zenodo DOI: to be added after Zenodo archiving
```

After creating the Zenodo archive, replace the placeholder above with the DOI, for example:

```text
https://doi.org/10.5281/zenodo.xxxxxxx
```

---

## 13. Citation

If you use this repository or build upon FGCRL, please cite the repository and the manuscript once it becomes available.

### Manuscript citation

```bibtex
@misc{liu2026fgcrl,
  title  = {FGCRL: Foreground-Prior-Guided Self-Supervised Pretraining for Lesion-Sensitive Visual Representation Learning in Complex Natural Scenes},
  author = {Liu, Wenjuan and Mohamed, Ahmad Sufril Azlan and Osman, Mohd Azam and Kie, Kim Hwa and Wong, Chow Jeng},
  year   = {2026},
  note   = {Manuscript submitted to The Visual Computer. Under review.}
}
```

### Code repository citation

```bibtex
@misc{liu2026fgcrl_code,
  title        = {Disease Representation Learning: Code for FGCRL},
  author       = {Liu, Wenjuan and Mohamed, Ahmad Sufril Azlan and Osman, Mohd Azam and Kie, Kim Hwa and Wong, Chow Jeng},
  year         = {2026},
  howpublished = {\url{https://github.com/WendyCV/Disease_Representation_Learning}},
  note         = {Code repository for the manuscript submitted to The Visual Computer. Zenodo DOI to be added.}
}
```

After the manuscript is accepted or published, please update the journal, volume, pages, DOI, and Zenodo DOI.

---

## 14. License

This project is released under the MIT License for academic research use. Please check the `LICENSE` file before using the code for commercial applications.

The YOLO base weights, public datasets, SAM2-related components, and third-party tools may be subject to their own licenses. Users are responsible for following the original licenses of external datasets, pretrained detectors, segmentation tools, and third-party dependencies.

---

## 15. Acknowledgements

This work is part of a research project on lesion-sensitive visual representation learning for diseased leaf detection in complex natural scenes. We thank the collaborators and contributors who supported field data collection, annotation checking, implementation, experimental validation, and manuscript preparation.

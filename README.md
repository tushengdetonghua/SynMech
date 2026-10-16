# SynMech

Code accompanying the manuscript:

**Synthetic data generation for instance segmentation of stacked mechanical parts**

SynMech is a synthetic-data generation and annotation pipeline for instance segmentation of stacked mechanical parts. The repository contains the main code used for synthetic scene generation, YOLO-compatible annotation conversion, YOLO11n-seg training, and model evaluation.

## Repository structure

```text
SynMech/
├── synthetic_generation.py
├── annotation_conversion.py
├── train.py
├── val.py
├── requirements.txt
└── README.md
```

### Main scripts

- `synthetic_generation.py`  
  Generates physically simulated stacked-part scenes with BlenderProc. It supports randomized part poses, materials, illumination, backgrounds/HDRIs, camera parameters, depth output, instance/category segmentation maps, and basic quality checks.

- `annotation_conversion.py`  
  Converts instance segmentation PNGs and corresponding JSON metadata into YOLO segmentation labels. The conversion preserves hole boundaries and connects disconnected visible regions of the same instance into a single YOLO-compatible polygon path.

- `train.py`  
  Trains a YOLO11 instance-segmentation model using Ultralytics.

- `val.py`  
  Evaluates a trained YOLO11 instance-segmentation model on the specified dataset split.

## Environment

The original experiments used a Python 3.9-based environment and Ultralytics 8.3.9. GPU acceleration can be used when a compatible PyTorch/CUDA installation is available.

Install PyTorch separately according to your CUDA/CPU environment, then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

BlenderProc scripts should be executed with the `blenderproc run` command so that Blender/Python-specific modules such as `bpy` and `mathutils` are available.

## Suggested asset layout

```text
assets/
├── parts/
│   ├── part_01.obj
│   ├── part_02.obj
│   └── ...
├── hdris/
├── backgrounds/
├── ground_texture/
├── container_texture/
└── part_texture/
```

The CAD models may use formats supported by the script, including OBJ, PLY, STL, FBX, GLB, and GLTF.

For `--background-mode hdris`, place HDR/EXR files in `assets/hdris/`.

For `--background-mode backgrounds`, place background images in `assets/backgrounds/`.

## 1. Generate synthetic data

Example:

```bash
blenderproc run synthetic_generation.py \
  --parts-dir assets/parts \
  --image-dir assets \
  --output-dir output \
  --dataset-name synmech_dataset \
  --num-images 100 \
  --background-mode hdris
```

Important options include:

```text
--min-parts / --max-parts
--background-mode {backgrounds,hdris}
--seed
--width / --height
--camera-radius-min / --camera-radius-max
--camera-elev-min / --camera-elev-max
--samples
--min-visible-pixels
--min-visible-instances
```

The generated dataset is written under:

```text
output/synmech_dataset/
```

Instance segmentation PNGs and JSON metadata are stored inside BOP-style chunk directories such as:

```text
output/synmech_dataset/train_pbr/000000/
├── rgb/
├── depth/
├── seg_category/
├── seg_instance/
└── seg_instance_labels/
```

## 2. Convert annotations to YOLO segmentation format

`annotation_conversion.py` expects a scene/chunk directory containing:

```text
seg_instance/
seg_instance_labels/
```

Example:

```bash
python annotation_conversion.py \
  --scene-root output/synmech_dataset/train_pbr/000000 \
  --output-dir converted_labels \
  --class-map 1:0,2:1,3:2 \
  --verify-ultralytics
```

The `--class-map` argument maps source category IDs to zero-based YOLO class IDs. Adjust it to match the category IDs in the generated dataset.

The output directory contains YOLO segmentation labels together with validation reports and optional previews.

## 3. Train YOLO11n-seg

Prepare an Ultralytics dataset YAML file that points to the training, validation, and test image/label directories.

Example:

```bash
python train.py \
  --model yolo11n-seg.pt \
  --data path/to/data.yaml \
  --epochs 200 \
  --batch 8 \
  --imgsz 640 \
  --optimizer SGD \
  --name synmech
```

Default training settings in `train.py` include:

```text
epochs = 200
batch = 8
imgsz = 640
optimizer = SGD
close_mosaic = 10
seed = 0
deterministic = True
AMP = True
```

## 4. Evaluate a trained model

Example:

```bash
python val.py \
  --model runs/train/synmech/weights/best.pt \
  --data path/to/data.yaml \
  --split test \
  --imgsz 640 \
  --batch 16
```

Valid values for `--split` are:

```text
train
val
test
```

## Notes on reproducibility

- Use the same CAD models, textures, HDRIs/backgrounds, dataset split, and category mapping when reproducing the reported experiments.
- The random seed can be fixed with `--seed` during synthetic generation and training.
- PyTorch/CUDA is intentionally not pinned in `requirements.txt`, because the appropriate build depends on the local GPU and CUDA environment.
- The cleaned public scripts do not contain user-specific absolute paths or personal information.
- Large datasets, trained weights, and third-party software packages are not included in this code repository.

## Third-party software

This project uses BlenderProc, Ultralytics, OpenCV, NumPy, and Pillow.

Please follow the licenses and terms of the corresponding third-party projects when redistributing or modifying their code.

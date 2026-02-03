# NIH Chest X-ray Disease Classification

A deep learning project for multi-label classification of 14 thoracic diseases from chest X-rays using the NIH Chest X-ray dataset.

## Overview

This project implements several CNN architectures to detect multiple diseases from a single chest X-ray image:

- **Custom CNN**: A 5-block convolutional neural network built from scratch
- **ResNet50**: Transfer learning with pre-trained ResNet50
- **DynamicCNN**: Configurable architecture for evolutionary/AutoML search

### Disease Classes (14)

Atelectasis, Cardiomegaly, Consolidation, Edema, Effusion, Emphysema, Fibrosis, Hernia, Infiltration, Mass, Nodule, Pleural_Thickening, Pneumonia, Pneumothorax

## Requirements

- Python 3.11+
- PyTorch
- AMD GPU with DirectML support (or NVIDIA GPU with CUDA)

## Installation

1. **Clone the repository**
2. **Create a virtual environment**
   ```bash
   python -m venv venv311
   venv311\Scripts\activate  # Windows
   # source venv311/bin/activate  # Linux/Mac
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

   For sklearn metrics support (AUC calculation), also install:
   ```bash
   pip install scikit-learn
   ```

## Dataset Setup

1. Download the NIH Chest X-ray dataset from [NIH Clinical Center](https://nihcc.app.box.com/v/ChestXray-NIHCC)

2. Extract the images into the `data/` directory with this structure:
   ```
   data/
   ├── images_001/
   │   └── images/
   │       └── *.png
   ├── images_002/
   │   └── images/
   │       └── *.png
   ...
   ├── images_012/
   ├── Data_Entry_2017.csv
   ├── train_val_list.txt
   └── test_list.txt
   ```

3. **Preprocess images** (recommended, one-time operation):
   ```bash
   python preprocess.py
   ```
   This resizes all images from 1024x1024 to 224x224 using multi-threading and saves them to `data_resized/`. This dramatically speeds up training by avoiding runtime resizing.

## Usage

### Standard Training

**Custom CNN:**
```bash
python train.py --model-name custom --epochs 10 --batch-size 16 --lr 1e-4 --save-model
```

**ResNet50 (Transfer Learning):**
```bash
python train.py --model-name resnet50 --epochs 20 --batch-size 32 --lr 1e-4 --save-model
```

**DenseNet121 (Transfer Learning):**
```bash
python train.py --model-name densenet121 --epochs 20 --batch-size 32 --lr 1e-4 --save-model
```

### Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model-name` | `custom` | Model architecture: `custom`, `resnet50`, `densenet121` |
| `--epochs` | `10` | Number of training epochs |
| `--batch-size` | `16` | Batch size |
| `--lr` | `1e-4` | Learning rate |
| `--data-dir` | `./data_resized` | Path to preprocessed images |
| `--csv-file` | `./data/Data_Entry_2017.csv` | Path to labels CSV |
| `--train-list` | `./data/train_val_list.txt` | Training split file |
| `--test-list` | `./data/test_list.txt` | Test/validation split file |
| `--output-file` | `training_log.xlsx` | Excel file for logging results |
| `--num-workers` | `4` | DataLoader worker threads |
| `--dry-run` | - | Quick test with 100 samples |
| `--save-model` | - | Save best model checkpoint |
| `--resume` | - | Resume from checkpoint |

### Quick Validation (Dry Run)

Test your setup with a small subset:
```bash
python train.py --dry-run
```

### Resume Training

Continue from a saved checkpoint:
```bash
python train.py --model-name resnet50 --epochs 50 --save-model --resume
```

## AutoML / Evolutionary Training

The `auto_learner.py` script runs an evolutionary search to find optimal architectures:

```bash
python auto_learner.py
```

**How it works:**
1. Starts with a baseline DynamicCNN configuration
2. Trains for 5 epochs per generation
3. Mutates the architecture if performance improves
4. Saves state automatically - can be interrupted with Ctrl+C and resumed

**Dry run:**
```bash
python auto_learner.py --dry-run
```

### AutoML State Files

- `auto_learner_state.pth` - Best configuration and metrics
- `current_experiment.pth` - Resume state for crash recovery
- `best_model_overall.pth` - Weights of best model found
- `auto_training_log.xlsx` - Experiment history

## Data Validation

Verify your dataset is correctly set up:
```bash
python debug_data.py
```

## Output Files

| File | Description |
|------|-------------|
| `training_log.xlsx` | Standard training metrics per epoch |
| `auto_training_log.xlsx` | AutoML experiment results |
| `{model}_best.pth` | Model checkpoint (e.g., `resnet50_best.pth`) |
| `best_model_overall.pth` | Best AutoML model weights |

## Project Structure

```
thorax-x-ray-cnn/
├── train.py              # Standard training script
├── auto_learner.py       # Evolutionary/AutoML training
├── preprocess.py         # Image resizing utility
├── debug_data.py         # Dataset validation
├── requirements.txt      # Python dependencies
├── src/
│   ├── dataset.py        # NIHChestXrayDataset class
│   ├── custom_model.py   # CustomCNN architecture
│   ├── dynamic_model.py  # DynamicCNN for AutoML
│   ├── evolution.py      # Mutation functions for AutoML
│   └── utils.py          # Logging utilities
├── data/                 # Raw dataset (1024x1024)
└── data_resized/         # Preprocessed images (224x224)
```

## Hardware Support

The project automatically detects available hardware:

1. **DirectML** (AMD GPUs) - Preferred on Windows with AMD
2. **CUDA** (NVIDIA GPUs) - For NVIDIA hardware
3. **CPU** - Fallback option

## Notes

- **Multi-label classification**: Each image can have multiple diseases. Uses `BCEWithLogitsLoss` instead of `CrossEntropyLoss`.
- **Image normalization**: Uses ImageNet mean/std values `[0.485, 0.456, 0.406]` / `[0.229, 0.224, 0.225]`
- **Preprocessing recommended**: Running `preprocess.py` once significantly speeds up training

## License

This project uses the NIH Chest X-ray dataset which is publicly available for research purposes.

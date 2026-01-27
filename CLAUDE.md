# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Medical image classification project for detecting 14 thoracic diseases from chest X-rays using the NIH Chest X-ray dataset. Multi-label classification (each image can have multiple diseases).

## Commands

### Environment Setup
```bash
pip install -r requirements.txt
```

### Preprocess Images (one-time, recommended)
```bash
python preprocess.py
```
Resizes images from 1024×1024 to 224×224, outputs to `./data_resized/`.

### Standard Training
```bash
# Custom CNN
python train.py --model-name custom --epochs 10 --batch-size 16 --lr 1e-4 --save-model

# Transfer learning
python train.py --model-name resnet50 --epochs 20 --batch-size 32 --save-model
python train.py --model-name densenet121 --epochs 20 --batch-size 32 --save-model

# Quick validation (100 samples)
python train.py --dry-run
```

### AutoML Evolutionary Training
```bash
python auto_learner.py           # Runs infinite evolutionary search (Ctrl+C to pause)
python auto_learner.py --dry-run # Single cycle test
```
State is saved automatically; re-running resumes from checkpoint.

### Data Validation
```bash
python debug_data.py
```

## Architecture

### Directory Structure
- `src/dataset.py` - `NIHChestXrayDataset` class, handles multi-label encoding
- `src/custom_model.py` - `CustomCNN`, 5-block CNN architecture
- `src/dynamic_model.py` - `DynamicCNN`, configurable architecture for evolutionary search
- `src/evolution.py` - `get_default_config()`, `mutate_config()` for architecture mutations
- `src/utils.py` - Excel logging utilities

### Data Organization
- Images split across `data/images_001/` through `data/images_012/`
- Labels in `data/Data_Entry_2017.csv` (pipe-separated, e.g., "Cardiomegaly|Emphysema")
- Splits defined by `data/train_val_list.txt` and `data/test_list.txt`

### Key Design Decisions
- **Multi-label classification**: Uses `BCEWithLogitsLoss`, not `CrossEntropyLoss`
- **Device priority**: DirectML (AMD) → CUDA → CPU
- **AutoML uses short cycles**: 5 epochs per configuration to enable faster exploration
- **DynamicCNN config format**:
  ```python
  {
      "blocks": [{"filters": 32, "kernel": 3, "pool": True}, ...],
      "fc_layers": [512],
      "dropout": 0.5,
      "lr": 0.0001,
      "batch_size": 64
  }
  ```

### Output Files
- `training_log.xlsx` - Standard training results
- `auto_training_log.xlsx` - AutoML results
- `auto_learner_state.pth` - Best config/metrics from AutoML
- `best_model_overall.pth` - Best weights from AutoML
- `{model_name}_best.pth` - Checkpoints from standard training

## Disease Classes (14)
Atelectasis, Cardiomegaly, Effusion, Infiltration, Mass, Nodule, Pneumonia, Pneumothorax, Consolidation, Edema, Emphysema, Fibrosis, Pleural_Thickening, Hernia
"""
Fine-tune the best performing model from AutoML evolution.

Loads the best model from best_model_overall.pth and continues training
with configurable parameters for further optimization.

Supports Ctrl+C interruption with automatic checkpoint saving and resume.

Usage:
    python finetune_best.py --epochs 20 --lr 1e-5
    python finetune_best.py --epochs 10 --lr 5e-5 --batch-size 32
    python finetune_best.py --dry-run  # Quick test
    python finetune_best.py --reset    # Start fresh, ignore checkpoint
"""

import os
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
import numpy as np

from src.dataset import NIHChestXrayDataset
from src.dynamic_model import DynamicCNN
from src.utils import log_experiment

# Try importing sklearn for AUC metrics
try:
    from sklearn.metrics import roc_auc_score
    sklearn_available = True
except ImportError:
    sklearn_available = False

# --- CONFIGURATION ---
DATA_DIR = "./data_resized"
CSV_FILE = "./data/Data_Entry_2017.csv"
TRAIN_LIST = "./data/train_val_list.txt"
TEST_LIST = "./data/test_list.txt"

STATE_FILE = "auto_learner_state.pth"
BEST_MODEL_PATH = "best_model_overall.pth"
FINETUNED_MODEL_PATH = "finetuned_best.pth"
CHECKPOINT_FILE = "finetune_checkpoint.pth"
LOG_FILE = "finetune_log.xlsx"

# Device selection
try:
    import torch_directml
    device = torch_directml.device()
    print("Using DirectML (AMD GPU)")
except ImportError:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")


def get_args():
    parser = argparse.ArgumentParser(description="Fine-tune the best AutoML model")

    # Training params
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs to train")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate (lower than initial training)")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size (default: use config from state)")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")

    # Learning rate scheduling
    parser.add_argument("--scheduler", type=str, default="plateau",
                        choices=["none", "plateau", "cosine"],
                        help="LR scheduler type")
    parser.add_argument("--patience", type=int, default=3, help="Patience for ReduceLROnPlateau")

    # Regularization
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay for Adam")

    # Flags
    parser.add_argument("--dry-run", action="store_true", help="Quick test with limited data")
    parser.add_argument("--no-save", action="store_true", help="Don't save model checkpoints")
    parser.add_argument("--reset", action="store_true", help="Ignore checkpoint and start fresh")

    return parser.parse_args()


def load_best_model_and_config():
    """Load the best config from state file and model weights."""
    if not os.path.exists(STATE_FILE):
        raise FileNotFoundError(f"State file not found: {STATE_FILE}. Run auto_learner.py first.")

    if not os.path.exists(BEST_MODEL_PATH):
        raise FileNotFoundError(f"Best model not found: {BEST_MODEL_PATH}. Run auto_learner.py first.")

    # Load state to get config
    state = torch.load(STATE_FILE, weights_only=False)
    config = state['best_config']
    best_val_loss = state['best_val_loss']
    generation = state['generation']

    print(f"Loaded config from generation {generation}")
    print(f"Previous best validation loss: {best_val_loss:.4f}")
    print(f"Model config: {config}")

    # Create model and load weights (load to CPU first to avoid DirectML gradient issues)
    model = DynamicCNN(config)
    state_dict = torch.load(BEST_MODEL_PATH, map_location='cpu', weights_only=False)
    model.load_state_dict(state_dict)

    # Ensure all parameters require gradients for training
    for param in model.parameters():
        param.requires_grad = True

    return model, config, best_val_loss


def save_checkpoint(model, optimizer, scheduler, epoch, best_val_loss, best_auc, config, args):
    """Save training checkpoint for resume capability."""
    checkpoint = {
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict() if scheduler else None,
        'best_val_loss': best_val_loss,
        'best_auc': best_auc,
        'config': config,
        'args': {
            'epochs': args.epochs,
            'lr': args.lr,
            'batch_size': args.batch_size,
            'scheduler': args.scheduler,
            'patience': args.patience,
            'weight_decay': args.weight_decay,
        }
    }
    torch.save(checkpoint, CHECKPOINT_FILE)


def load_checkpoint():
    """Load checkpoint if it exists."""
    if os.path.exists(CHECKPOINT_FILE):
        return torch.load(CHECKPOINT_FILE, map_location='cpu', weights_only=False)
    return None


def get_dataloaders(batch_size, num_workers=4, dry_run=False):
    """Create train and validation dataloaders."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_ds = NIHChestXrayDataset(DATA_DIR, CSV_FILE, TRAIN_LIST, transform=transform)
    val_ds = NIHChestXrayDataset(DATA_DIR, CSV_FILE, TEST_LIST, transform=transform)

    if dry_run:
        print(">> DRY RUN: Limiting dataset to 100 images.")
        train_ds = Subset(train_ds, range(100))
        val_ds = Subset(val_ds, range(100))

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None
    )
    return train_loader, val_loader


def compute_auc(y_true, y_pred):
    """Compute mean AUC across all classes."""
    if not sklearn_available:
        return -1.0
    try:
        aucs = []
        for i in range(y_true.shape[1]):
            try:
                score = roc_auc_score(y_true[:, i], y_pred[:, i])
                aucs.append(score)
            except ValueError:
                pass  # Class not present in batch
        return np.mean(aucs) if aucs else 0.0
    except Exception as e:
        print(f"Warning: AUC calculation failed: {e}")
        return 0.0


def train(args):
    # Check for existing checkpoint
    checkpoint = None if args.reset else load_checkpoint()

    if checkpoint and not args.dry_run:
        print("\n" + "=" * 60)
        print("RESUMING FROM CHECKPOINT")
        print("=" * 60)
        config = checkpoint['config']
        start_epoch = checkpoint['epoch']
        best_val_loss = checkpoint['best_val_loss']
        best_auc = checkpoint['best_auc']

        # Load autolearner state to get prev_best_loss
        state = torch.load(STATE_FILE, weights_only=False)
        prev_best_loss = state['best_val_loss']

        # Recreate model and load checkpoint weights
        model = DynamicCNN(config)
        model.load_state_dict(checkpoint['model_state'])
        for param in model.parameters():
            param.requires_grad = True
        model = model.to(device)

        print(f"Resumed from epoch {start_epoch}")
        print(f"Best val loss so far: {best_val_loss:.4f}")
    else:
        # Fresh start
        if args.reset and os.path.exists(CHECKPOINT_FILE):
            os.remove(CHECKPOINT_FILE)
            print("Checkpoint cleared, starting fresh.")

        model, config, prev_best_loss = load_best_model_and_config()
        model = model.to(device)
        start_epoch = 0
        best_val_loss = float('inf')
        best_auc = 0.0

    # Use batch size from config if not specified
    batch_size = args.batch_size if args.batch_size else config.get('batch_size', 64)

    # Setup data
    train_loader, val_loader = get_dataloaders(batch_size, args.num_workers, args.dry_run)
    print(f"\nTraining samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")

    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Learning rate scheduler+

    scheduler = None
    if args.scheduler == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=args.patience, verbose=True
        )
    elif args.scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Restore optimizer and scheduler state if resuming
    if checkpoint and not args.dry_run:
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        if scheduler and checkpoint['scheduler_state']:
            scheduler.load_state_dict(checkpoint['scheduler_state'])

    print(f"\nFine-tuning configuration:")
    print(f"  Epochs: {start_epoch + 1} to {args.epochs}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Batch size: {batch_size}")
    print(f"  Scheduler: {args.scheduler}")
    print(f"  Weight decay: {args.weight_decay}")

    # Training loop with interrupt handling
    start_time = time.time()
    interrupted = False
    train_loss = 0.0  # Initialize for logging

    try:
        for epoch in range(start_epoch, args.epochs):
            epoch_start = time.time()

            # Training phase
            model.train()
            train_loss = 0.0

            for i, (images, labels) in enumerate(train_loader):
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

                if i % 50 == 0:
                    current_lr = optimizer.param_groups[0]['lr']
                    print(f"Epoch {epoch+1}/{args.epochs} | Batch {i}/{len(train_loader)} | "
                          f"Loss: {loss.item():.4f} | LR: {current_lr:.2e}")

            train_loss /= len(train_loader)

            # Validation phase
            model.eval()
            val_loss = 0.0
            all_labels = []
            all_preds = []

            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()

                    if sklearn_available:
                        all_labels.append(labels.cpu().numpy())
                        all_preds.append(torch.sigmoid(outputs).cpu().numpy())

            val_loss /= len(val_loader)

            # Calculate AUC
            auc_score = 0.0
            if sklearn_available and len(all_labels) > 0:
                all_labels = np.concatenate(all_labels)
                all_preds = np.concatenate(all_preds)
                auc_score = compute_auc(all_labels, all_preds)

            epoch_time = time.time() - epoch_start
            print(f"\n>>> Epoch {epoch+1} Complete ({epoch_time:.1f}s)")
            print(f"    Train Loss: {train_loss:.4f}")
            print(f"    Val Loss: {val_loss:.4f} (prev best from AutoML: {prev_best_loss:.4f})")
            print(f"    Val AUC: {auc_score:.4f}")

            # Update LR scheduler
            if scheduler:
                if args.scheduler == "plateau":
                    scheduler.step(val_loss)
                else:
                    scheduler.step()

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_auc = auc_score
                if not args.no_save and not args.dry_run:
                    torch.save(model.state_dict(), FINETUNED_MODEL_PATH)
                    print(f"    >> Saved new best model to {FINETUNED_MODEL_PATH}")

            # Save checkpoint after each epoch
            if not args.no_save and not args.dry_run:
                save_checkpoint(model, optimizer, scheduler, epoch, best_val_loss, best_auc, config, args)
                print(f"    >> Checkpoint saved (epoch {epoch+1})")

            print()

    except KeyboardInterrupt:
        interrupted = True
        print("\n" + "=" * 60)
        print("TRAINING INTERRUPTED")
        print("=" * 60)
        if not args.no_save and not args.dry_run:
            # Save current state
            save_checkpoint(model, optimizer, scheduler, epoch, best_val_loss, best_auc, config, args)
            print(f"Checkpoint saved at epoch {epoch+1}.")
            print(f"Run again to resume training.")
        print("=" * 60)
        return

    total_time = time.time() - start_time

    # Training completed successfully - clean up checkpoint
    if os.path.exists(CHECKPOINT_FILE) and not args.dry_run:
        os.remove(CHECKPOINT_FILE)
        print("Training complete, checkpoint file removed.")

    # Summary
    print("=" * 60)
    print("FINE-TUNING COMPLETE")
    print("=" * 60)
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best validation AUC: {best_auc:.4f}")
    print(f"Previous best (AutoML): {prev_best_loss:.4f}")

    improvement = prev_best_loss - best_val_loss
    if improvement > 0:
        print(f"Improvement: {improvement:.4f} ({100*improvement/prev_best_loss:.2f}%)")
    else:
        print(f"No improvement over AutoML baseline")

    # Log results
    if not args.dry_run:
        metrics = {
            'Train Loss': train_loss,
            'Val Loss': best_val_loss,
            'Val AUC': best_auc,
            'Prev Best Loss': prev_best_loss,
            'Improvement': improvement,
            'Duration (s)': total_time
        }
        hyperparams = {
            'Epochs': args.epochs,
            'Batch Size': batch_size,
            'LR': args.lr,
            'Scheduler': args.scheduler,
            'Weight Decay': args.weight_decay,
            'Model Config': str(config)
        }
        log_experiment(LOG_FILE, "Finetune-DynamicCNN", hyperparams, metrics)
        print(f"\nResults logged to {LOG_FILE}")


if __name__ == "__main__":
    args = get_args()
    train(args)

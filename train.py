import os
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models
import numpy as np
from src.dataset import NIHChestXrayDataset
from src.utils import log_experiment
from src.custom_model import CustomCNN

# Try importing torch_directml for AMD GPU support
try:
    import torch_directml
    dml_available = True
except ImportError:
    dml_available = False

# Try importing sklearn for metrics
try:
    from sklearn.metrics import roc_auc_score
    sklearn_available = True
except ImportError:
    sklearn_available = False

def get_args():
    parser = argparse.ArgumentParser(description="Train CNN on NIH Chest X-ray Dataset")
    
    # Paths
    parser.add_argument("--data-dir", type=str, default="./data_resized", help="Root data directory")
    parser.add_argument("--csv-file", type=str, default="./data/Data_Entry_2017.csv", help="Path to Data Entry CSV")
    parser.add_argument("--train-list", type=str, default="./data/train_val_list.txt", help="Path to train/val list")
    parser.add_argument("--test-list", type=str, default="./data/test_list.txt", help="Path to test list")
    parser.add_argument("--output-file", type=str, default="training_log.xlsx", help="Excel file to log results")
    
    # Training Params
    parser.add_argument("--model-name", type=str, default="custom", choices=["custom", "resnet50", "densenet121"], help="Model architecture")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of dataloader workers")
    
    # Flags
    parser.add_argument("--dry-run", action="store_true", help="Run a fast development run with limited data")
    parser.add_argument("--save-model", action="store_true", help="Save the best model checkpoint")
    parser.add_argument("--resume", action="store_true", help="Resume training from checkpoint")

    return parser.parse_args()

def get_device():
    if dml_available:
        print("Using DirectML (AMD GPU)")
        return torch_directml.device()
    elif torch.cuda.is_available():
        print("Using CUDA")
        return torch.device("cuda")
    else:
        print("Using CPU")
        return torch.device("cpu")

def compute_auc(y_true, y_pred):
    if not sklearn_available:
        return -1.0
    try:
        # Calculate AUC per class and average
        aucs = []
        for i in range(y_true.shape[1]):
            try:
                score = roc_auc_score(y_true[:, i], y_pred[:, i])
                aucs.append(score)
            except ValueError:
                pass # Can happen if a class is not present in the batch
        return np.mean(aucs) if aucs else 0.0
    except Exception as e:
        print(f"Warning: AUC calculation failed: {e}")
        return 0.0

def train(args):
    device = get_device()
    print(f"Device: {device}")

    # Transforms (images in data_resized/ are already 224x224)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Datasets
    print("Initializing Datasets...")
    train_dataset = NIHChestXrayDataset(
        data_dir=args.data_dir,
        csv_file=args.csv_file,
        split_list_file=args.train_list,
        transform=transform,
        fast_dev_run=args.dry_run
    )
    
    # For validation, we can use the test list or a split of train_val
    # Here using the provided test_list.txt as validation for simplicity, 
    # or you can split train_dataset. 
    val_dataset = NIHChestXrayDataset(
        data_dir=args.data_dir,
        csv_file=args.csv_file,
        split_list_file=args.test_list,
        transform=transform,
        fast_dev_run=args.dry_run
    )

    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=2 if args.num_workers > 0 else None
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=2 if args.num_workers > 0 else None
    )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # Model
    print(f"Building Model: {args.model_name}")
    if args.model_name == "custom":
        model = CustomCNN(num_classes=14)
    elif args.model_name == "resnet50":
        model = models.resnet50(weights='DEFAULT')
        model.fc = nn.Linear(model.fc.in_features, 14)
    elif args.model_name == "densenet121":
        model = models.densenet121(weights='DEFAULT')
        model.classifier = nn.Linear(model.classifier.in_features, 14)
    
    model = model.to(device)

    # Loss and Optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Resume from checkpoint if requested
    start_epoch = 0
    best_val_loss = float('inf')
    checkpoint_path = f"{args.model_name}_best.pth"

    if args.resume and os.path.exists(checkpoint_path):
        print(f"Resuming from checkpoint: {checkpoint_path}")
        # Load to CPU first to avoid DirectML compatibility issues with map_location
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # Move optimizer state tensors to device (needed for momentum buffers etc.)
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
        print(f"Resumed from epoch {start_epoch}, best val loss: {best_val_loss:.4f}")
    elif args.resume:
        print(f"No checkpoint found at {checkpoint_path}, starting fresh")

    # Training Loop
    print("Starting Training...")
    start_time = time.time()

    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_loss = 0.0
        
        # Training Step
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if i % 10 == 0:
                print(f"Epoch [{epoch+1}/{args.epochs}] Batch [{i}/{len(train_loader)}] Loss: {loss.item():.4f}")

        train_loss /= len(train_loader)

        # Validation Step
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
                
                # Store for AUC
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

        print(f"Epoch [{epoch+1}/{args.epochs}] Train Loss: {train_loss:.4f} Val Loss: {val_loss:.4f} Val AUC: {auc_score:.4f}")

        # Log epoch results
        epoch_metrics = {
            'Epoch': epoch + 1,
            'Train Loss': train_loss,
            'Val Loss': val_loss,
            'Val AUC': auc_score,
        }
        epoch_hyperparams = {
            'Batch Size': args.batch_size,
            'LR': args.lr,
            'Dry Run': args.dry_run
        }
        log_experiment(args.output_file, args.model_name, epoch_hyperparams, epoch_metrics)

        # Save Best Model
        if args.save_model and val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
            }
            torch.save(checkpoint, f"{args.model_name}_best.pth")
            print("Saved Best Model")

    total_time = time.time() - start_time
    print(f"Training Complete. Total Time: {total_time:.2f}s")

if __name__ == "__main__":
    args = get_args()
    train(args)

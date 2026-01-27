import os
import torch
import torch.nn as nn
import torch.optim as optim
import time
import json
import copy
import argparse
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from src.dataset import NIHChestXrayDataset
from src.dynamic_model import DynamicCNN
from src.utils import log_experiment
from src.evolution import get_default_config, mutate_config

# --- CONFIGURATION ---
DATA_DIR = "./data_resized"  # Use the optimized data
CSV_FILE = "./data/Data_Entry_2017.csv"
TRAIN_LIST = "./data/train_val_list.txt"
TEST_LIST = "./data/test_list.txt"

STATE_FILE = "auto_learner_state.pth"   # Stores the overall meta-state (best score, best config)
CURRENT_EXP_FILE = "current_experiment.pth" # Stores the detailed Resume state (model weights, optimizer)
BEST_MODEL_PATH = "best_model_overall.pth"
LOG_FILE = "auto_training_log.xlsx"

# Try importing torch_directml
try:
    import torch_directml
    device = torch_directml.device()
    if __name__ == "__main__":
        print("Using DirectML (AMD GPU)")
except:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if __name__ == "__main__":
        print(f"Using device: {device}")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Run a fast test cycle")
    return parser.parse_args()

def get_dataloaders(batch_size, num_workers=4, dry_run=False):
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

    # Persistent workers optimization
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True, 
        persistent_workers=(num_workers > 0), prefetch_factor=2 if num_workers > 0 else None
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=True,
        persistent_workers=(num_workers > 0), prefetch_factor=2 if num_workers > 0 else None
    )
    return train_loader, val_loader

def save_experiment_state(model, optimizer, epoch, config, val_loss):
    """Saves the specific state of the CURRENT running experiment for resume."""
    state = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "config": config,
        "val_loss": val_loss
    }
    torch.save(state, CURRENT_EXP_FILE)

def load_overall_state():
    """Loads the best-so-far tracking."""
    if os.path.exists(STATE_FILE):
        return torch.load(STATE_FILE)
    return {"best_val_loss": float('inf'), "best_config": get_default_config(), "generation": 0}

def save_overall_state(best_val_loss, best_config, generation):
    state = {
        "best_val_loss": best_val_loss,
        "best_config": best_config,
        "generation": generation
    }
    torch.save(state, STATE_FILE)

def run_training_cycle(args):
    # 1. Load Global State (Best config so far)
    global_state = load_overall_state()
    best_config = global_state['best_config']
    best_val_loss_overall = global_state['best_val_loss']
    generation = global_state['generation']
    
    current_config = best_config
    start_epoch = 0
    
    # 2. Check for Resume (Crash Recovery)
    resume_mode = False
    if os.path.exists(CURRENT_EXP_FILE) and not args.dry_run:
        print(">> FOUND INTERRUPTED EXPERIMENT. RESUMING...")
        checkpoint = torch.load(CURRENT_EXP_FILE)
        current_config = checkpoint['config']
        start_epoch = checkpoint['epoch'] + 1 # Resume from next epoch
        resume_mode = True
    else:
        # New Cycle: Decide whether to mutate
        if generation > 0:
            print(f">> Starting Generation {generation + 1}. Mutating...")
            current_config = mutate_config(best_config)
        else:
            print(">> Starting Generation 1 (Baseline).")

    # 3. Setup Model & Training
    model = DynamicCNN(current_config).to(device)
    optimizer = optim.Adam(model.parameters(), lr=current_config['lr'])
    criterion = nn.BCEWithLogitsLoss()

    if resume_mode:
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])

    train_loader, val_loader = get_dataloaders(current_config['batch_size'], dry_run=args.dry_run)
    
    print(f"Model Config: {json.dumps(current_config, indent=2)}")
    
    # Run for 5 epochs per experiment (short cycles for evolution)
    TARGET_EPOCHS = 1 if args.dry_run else 5
    
    best_val_loss_this_run = float('inf')

    # --- TRAINING LOOP ---
    try:
        for epoch in range(start_epoch, TARGET_EPOCHS):
            model.train()
            train_loss = 0.0
            
            # Train Batch
            for i, (images, labels) in enumerate(train_loader):
                images, labels = images.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                if i % 50 == 0:
                    print(f"Gen {generation}|Ep {epoch+1}|Batch {i} - Loss: {loss.item():.4f}")

            train_loss /= len(train_loader)

            # Validation
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            print(f"--> Epoch {epoch+1} Finished. Train: {train_loss:.4f}, Val: {val_loss:.4f}")
            
            # Update best for this specific run
            if val_loss < best_val_loss_this_run:
                best_val_loss_this_run = val_loss

            # SAVE RESUME STATE (Every Epoch)
            if not args.dry_run:
                save_experiment_state(model, optimizer, epoch, current_config, val_loss)

    except KeyboardInterrupt:
        print("\nTraining interrupted! State saved. Run again to resume.")
        return # Exit cleanly, state is already saved

    # --- END OF EXPERIMENT EVALUATION ---
    print(f"Experiment Finished. Best Val Loss: {best_val_loss_this_run:.4f}")
    
    # Compare with Overall Best
    improved = False
    if best_val_loss_this_run < best_val_loss_overall:
        print(f"!!! NEW RECORD !!! Previous Best: {best_val_loss_overall:.4f} -> New: {best_val_loss_this_run:.4f}")
        best_val_loss_overall = best_val_loss_this_run
        best_config = current_config
        
        # Save weights as the new "Gold Standard"
        if not args.dry_run:
            torch.save(model.state_dict(), BEST_MODEL_PATH)
        improved = True
    else:
        print(f"Result did not beat baseline ({best_val_loss_overall:.4f}). Reverting config.")
    
    # Log to Excel
    metrics = {'Val Loss': best_val_loss_this_run, 'Improvement': improved}
    log_experiment(LOG_FILE, "DynamicCNN", current_config, metrics)

    # Clean up resume file because run finished successfully
    if os.path.exists(CURRENT_EXP_FILE) and not args.dry_run:
        os.remove(CURRENT_EXP_FILE)

    # Save Global State for next run
    if not args.dry_run:
        save_overall_state(best_val_loss_overall, best_config, generation + 1)
    
    print(">> Restarting cycle...\n")
    
    # For dry run, exit after one cycle
    if args.dry_run:
        print("Dry run complete.")
        return

    # Recursively call to loop forever
    run_training_cycle(args)

if __name__ == "__main__":
    args = get_args()
    run_training_cycle(args)

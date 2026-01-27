import os
import pandas as pd
import torch
import torch_directml
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

# --- STEP 1: SETUP DEVICE & PATHS ---
print("Device Setup")
device = torch_directml.device()
data_dir = "./data"

# Build the path map for the 12 subfolders
image_path_map = {}
for i in range(1, 13):
    folder_path = os.path.join(data_dir, f"images_{i:03d}", "images")
    if os.path.isdir(folder_path):
        for filename in os.listdir(folder_path):
            image_path_map[filename] = os.path.join(folder_path, filename)

# --- STEP 2: DEFINE DATASET ---
print("Dataset Setup start")
class NIHChestXrayDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        # The 14 disease categories from the dataset [cite: 2, 14]
        self.labels = [
            "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", 
            "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema", 
            "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia"
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx, 0] # Image Index [cite: 19]
        img_path = image_path_map[img_name]
        
        image = Image.open(img_path).convert("RGB")
        
        # Multi-label parsing: Check if disease name is in the 'Finding Labels' string [cite: 13, 19]
        label_str = str(self.data.iloc[idx, 1])
        label_vector = [1.0 if l in label_str else 0.0 for l in self.labels]
        
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label_vector)

print("Dataset Setup complete")
print("Init data and models")
# --- STEP 3: INITIALIZE DATA & MODEL ---
# Standard ResNet transformation for medical imaging
transform = transforms.Compose([
    transforms.Resize((224, 224)), # Resize from 1024x1024 to save VRAM [cite: 19]
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

dataset = NIHChestXrayDataset(csv_file="./data/Data_Entry_2017.csv", transform=transform)
train_loader = DataLoader(
    dataset, 
    batch_size=64,      # Try increasing this if VRAM allows
    shuffle=True, 
    num_workers=8,      # <--- This is the key. Try 4, 8, or 12
    pin_memory=True     # Speeds up data transfer to GPU
)

# Load a pre-trained model and modify the final layer for 14 classes [cite: 2]
model = models.resnet50(weights='DEFAULT')
model.fc = torch.nn.Linear(model.fc.in_features, 14)
model.to(device)

# Multi-label loss function
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# --- STEP 4: TRAINING LOOP ---
print("Starting Training...")
model.train()
for images, labels in train_loader:
    images, labels = images.to(device), labels.to(device)
    
    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    
    print(f"Batch Loss: {loss.item():.4f}")
    # break # Remove this break to train on the whole dataset!
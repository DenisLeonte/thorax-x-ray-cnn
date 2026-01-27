import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image

class NIHChestXrayDataset(Dataset):
    def __init__(self, data_dir, csv_file, split_list_file=None, transform=None, fast_dev_run=False):
        """
        Args:
            data_dir (str): Path to the data directory containing images_0xx folders.
            csv_file (str): Path to the Data_Entry_2017.csv file.
            split_list_file (str, optional): Path to train_val_list.txt or test_list.txt. 
                                             If None, uses all data.
            transform (callable, optional): Optional transform to be applied on a sample.
            fast_dev_run (bool): If True, only load a small subset of the data for testing.
        """
        self.data_dir = data_dir
        self.transform = transform
        self.image_path_map = self._build_image_path_map()
        
        # Load CSV
        self.df = pd.read_csv(csv_file)
        
        # Filter by split list if provided
        if split_list_file:
            with open(split_list_file, 'r') as f:
                split_filenames = set(x.strip() for x in f.readlines())
            self.df = self.df[self.df['Image Index'].isin(split_filenames)]
            
        if fast_dev_run:
            print("WARNING: Fast dev run enabled. Limiting dataset to 100 samples.")
            self.df = self.df.iloc[:100]

        self.labels_list = [
            "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", 
            "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema", 
            "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia"
        ]
        
    def _build_image_path_map(self):
        """Scans the data directory for all images in subfolders."""
        image_map = {}
        # Assuming folders are named images_001 to images_012
        for i in range(1, 13):
            folder_name = f"images_{i:03d}"
            folder_path = os.path.join(self.data_dir, folder_name, "images")
            if os.path.exists(folder_path):
                for filename in os.listdir(folder_path):
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                        image_map[filename] = os.path.join(folder_path, filename)
        return image_map

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.df.iloc[idx]['Image Index']
        
        # Handle case where image might not be found (though it should be)
        if img_name not in self.image_path_map:
            # Fallback or error? For now, let's error to be safe, or skip
            raise FileNotFoundError(f"Image {img_name} not found in data directories.")

        img_path = self.image_path_map[img_name]
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
             print(f"Error loading image {img_path}: {e}")
             # Return a black image or handle gracefully? 
             # For training stability, raising error is usually better than bad data
             raise e

        if self.transform:
            image = self.transform(image)

        # Labels
        # Finding Labels are separated by '|', e.g., "Cardiomegaly|Emphysema"
        label_str = self.df.iloc[idx]['Finding Labels']
        
        # Create multi-hot vector
        label_vec = torch.zeros(len(self.labels_list), dtype=torch.float32)
        
        if label_str != "No Finding":
            for i, label in enumerate(self.labels_list):
                if label in label_str:
                    label_vec[i] = 1.0

        return image, label_vec

import os
import time
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

# Configuration
SOURCE_DIR = os.path.abspath("./data")
DEST_DIR = os.path.abspath("./data_resized")
TARGET_SIZE = (224, 224)
NUM_THREADS = os.cpu_count() or 4

def process_single_image(file_info):
    """Function to be called by threads to resize one image."""
    src_path, dest_path = file_info
    
    # Create the output folder for this specific image if it doesn't exist
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    
    if os.path.exists(dest_path):
        return
        
    try:
        with Image.open(src_path) as img:
            # Handle grayscale/RGBA
            img = img.convert('RGB')
            # Resize
            img = img.resize(TARGET_SIZE, Image.Resampling.LANCZOS)
            # Save
            img.save(dest_path, optimize=True, quality=95)
    except Exception as e:
        print(f"\nError processing {src_path}: {e}")

def main():
    print("--- NIH Chest X-ray Preprocessor ---")
    print(f"Source: {SOURCE_DIR}")
    print(f"Destination: {DEST_DIR}")
    
    if not os.path.exists(SOURCE_DIR):
        print(f"ERROR: Source directory {SOURCE_DIR} not found.")
        return

    start_time = time.time()
    tasks = []

    print("Scanning for images...")
    # Walk through ALL subdirectories of 'data'
    for root, dirs, files in os.walk(SOURCE_DIR):
        for filename in files:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                src_path = os.path.join(root, filename)
                
                # Create the corresponding path in the destination folder
                rel_path = os.path.relpath(src_path, SOURCE_DIR)
                dest_path = os.path.join(DEST_DIR, rel_path)
                
                tasks.append((src_path, dest_path))

    total_images = len(tasks)
    print(f"Found {total_images} images.")
    
    if total_images == 0:
        print("No images found. Please check your 'data' folder structure.")
        return

    print(f"Starting resize with {NUM_THREADS} threads...")
    
    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        for i, _ in enumerate(executor.map(process_single_image, tasks)):
            if i % 100 == 0 or i == total_images - 1:
                print(f"Progress: {i+1}/{total_images} ({(i+1)/total_images*100:.1f}%)", end='\r')

    end_time = time.time()
    print(f"\n\nFinished! Total time: {end_time - start_time:.2f} seconds.")
    print(f"All images are now in: {DEST_DIR}")

if __name__ == "__main__":
    main()

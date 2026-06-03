import os
import cv2
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# Define paths
INPUT_DIR = "C:\\Users\\STUDENT\\Documents\\Spring 2026\\Research Project - MS\\dataset_split"         
OUTPUT_DIR = "C:\\Users\\STUDENT\\Documents\\Spring 2026\\Research Project - MS\\ degraded_split"       

def apply_degradation(src_path, dst_path):
    """
    Applies realistic physical degradations to synthetic microstructures.
    """
    img = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return

    # 1. Gaussian Blurring
    blurred = cv2.GaussianBlur(img, (9, 9), 0)

    # 2. Morphological Operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    wobbly = cv2.morphologyEx(blurred, cv2.MORPH_OPEN, kernel)
    wobbly = cv2.morphologyEx(wobbly, cv2.MORPH_CLOSE, kernel)

    # 3. Masking ("Holes of Information")
    degraded_x = wobbly.copy()
    h, w = degraded_x.shape
    
    num_holes = 5
    for _ in range(num_holes):
        hole_w = np.random.randint(15, 30)
        hole_h = np.random.randint(15, 30)
        x = np.random.randint(0, max(1, w - hole_w))
        y = np.random.randint(0, max(1, h - hole_h))
        degraded_x[y:y+hole_h, x:x+hole_w] = 0

    # 4. Save the final flawed image
    cv2.imwrite(dst_path, degraded_x)

def process_nested_dataset():
    valid_extensions = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')
    tasks = []

    print("Scanning directory structure...")
    
    # Traverse the entire directory tree
    for root, dirs, files in os.walk(INPUT_DIR):
        for file in files:
            if file.lower().endswith(valid_extensions):
                src_path = os.path.join(root, file)
                
                # Figure out the relative path to maintain the folder structure
                rel_path = os.path.relpath(src_path, INPUT_DIR)
                dst_path = os.path.join(OUTPUT_DIR, rel_path)
                
                # Ensure the exact subfolder exists in the output directory before saving
                os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                
                tasks.append((src_path, dst_path))
                
    print(f"Found {len(tasks)} images across all splits and classes.")
    
    max_workers = os.cpu_count()
    print(f"Processing images using {max_workers} CPU workers...")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        list(tqdm(executor.map(lambda p: apply_degradation(*p), tasks), total=len(tasks)))

if __name__ == "__main__":
    process_nested_dataset()
    print("Dataset degradation complete! Mirrored folder structure is ready.")
import numpy as np
import os
import torch
from PIL import Image
import matplotlib.pyplot as plt

def check_and_visualize_outputs(output_dir):
    """
    Check dimensions of .npy files and create visualizations
    
    Args:
        output_dir: Directory containing the .npy files (e.g., /home/hamscher/datasets/Cityscapes/FGSM_untargeted4/)
    """
    
    # Find all .npy files
    npy_files = [f for f in os.listdir(output_dir) if f.endswith('.npy')]
    
    if not npy_files:
        print(f"No .npy files found in {output_dir}")
        return
    
    print(f"Found {len(npy_files)} .npy files")
    
    # Create output directory for images
    img_output_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(img_output_dir, exist_ok=True)
    
    # Check first file to understand the format
    sample_file = npy_files[0]
    sample_path = os.path.join(output_dir, sample_file)
    sample_data = np.load(sample_path)
    
    print(f"\nSample file: {sample_file}")
    print(f"Shape: {sample_data.shape}")
    print(f"Data type: {sample_data.dtype}")
    print(f"Min value: {sample_data.min():.6f}")
    print(f"Max value: {sample_data.max():.6f}")
    print(f"Mean value: {sample_data.mean():.6f}")
    
    # Determine if this is segmentation probabilities or RGB image
    if len(sample_data.shape) == 3:
        channels, height, width = sample_data.shape
        print(f"Format: {channels} channels, {height}x{width} resolution")
        
        if channels == 3:
            print("This appears to be RGB image format")
            process_as_rgb(output_dir, npy_files, img_output_dir)
        elif channels > 3:
            print(f"This appears to be segmentation probability maps ({channels} classes)")
            process_as_segmentation(output_dir, npy_files, img_output_dir, channels)
        else:
            print("Unknown format")
    else:
        print("Unexpected shape format")

def process_as_rgb(output_dir, npy_files, img_output_dir):
    """Process files as RGB images"""
    print("\nProcessing as RGB images...")
    
    for i, filename in enumerate(npy_files[:5]):  # Process first 5 files
        filepath = os.path.join(output_dir, filename)
        data = np.load(filepath)
        
        # Convert from (C, H, W) to (H, W, C)
        img_array = np.transpose(data, (1, 2, 0))
        
        # Normalize to 0-255 if needed
        if img_array.max() <= 1.0:
            img_array = (img_array * 255).astype(np.uint8)
        else:
            img_array = img_array.astype(np.uint8)
        
        # Save as PNG
        img = Image.fromarray(img_array)
        output_path = os.path.join(img_output_dir, filename.replace('.npy', '_rgb.png'))
        img.save(output_path)
        print(f"Saved: {output_path}")

def process_as_segmentation(output_dir, npy_files, img_output_dir, num_classes):
    """Process files as segmentation probability maps"""
    print(f"\nProcessing as segmentation maps with {num_classes} classes...")
    
    # Cityscapes color palette
    cityscapes_colors = [
        (128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156), (190, 153, 153),
        (153, 153, 153), (250, 170, 30), (220, 220, 0), (107, 142, 35), (152, 251, 152),
        (0, 130, 180), (220, 20, 60), (255, 0, 0), (0, 0, 142), (0, 0, 70),
        (0, 60, 100), (0, 80, 100), (0, 0, 230), (119, 11, 32)
    ]
    
    for i, filename in enumerate(npy_files[:5]):  # Process first 5 files
        filepath = os.path.join(output_dir, filename)
        data = np.load(filepath)  # Shape: (num_classes, H, W)
        
        # Get predicted class for each pixel (argmax across classes)
        pred_mask = np.argmax(data, axis=0)  # Shape: (H, W)
        
        # Create colored segmentation map
        height, width = pred_mask.shape
        colored_mask = np.zeros((height, width, 3), dtype=np.uint8)
        
        for class_id in range(min(num_classes, len(cityscapes_colors))):
            mask = (pred_mask == class_id)
            colored_mask[mask] = cityscapes_colors[class_id]
        
        # Save colored segmentation
        img = Image.fromarray(colored_mask)
        output_path = os.path.join(img_output_dir, filename.replace('.npy', '_segmentation.png'))
        img.save(output_path)
        
        # Also save probability visualization for first few classes
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'Probability Maps - {filename}')
        
        for j in range(min(6, num_classes)):
            row, col = j // 3, j % 3
            axes[row, col].imshow(data[j], cmap='hot', vmin=0, vmax=1)
            axes[row, col].set_title(f'Class {j}')
            axes[row, col].axis('off')
        
        # Hide unused subplots
        for j in range(6, 6):
            if j < 6:
                axes[j // 3, j % 3].axis('off')
        
        prob_output_path = os.path.join(img_output_dir, filename.replace('.npy', '_probabilities.png'))
        plt.tight_layout()
        plt.savefig(prob_output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved segmentation: {output_path}")
        print(f"Saved probabilities: {prob_output_path}")

if __name__ == "__main__":
    # Set your output directory here
    output_directory = "/home/hamscher/datasets/Cityscapes/FGSM_untargeted4/"
    
    if os.path.exists(output_directory):
        check_and_visualize_outputs(output_directory)
    else:
        print(f"Directory not found: {output_directory}")
        print("Please update the output_directory variable with the correct path")
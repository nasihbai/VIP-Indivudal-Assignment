"""
CDS6334 Visual Information Processing - Utility Functions
Trimester 2530

Helper functions for visualization and reporting.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os


def create_comparison_grid(images_dict: dict, output_path: str, 
                           titles: list = None, figsize: tuple = (15, 10)):
    """
    Create a grid of images for comparison.
    
    Args:
        images_dict: Dictionary with image names as keys and image arrays as values
        output_path: Path to save the comparison image
        titles: List of titles for each image
        figsize: Figure size
    """
    n_images = len(images_dict)
    cols = min(3, n_images)
    rows = (n_images + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if rows == 1 and cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for idx, (name, img) in enumerate(images_dict.items()):
        # Convert BGR to RGB for display
        if len(img.shape) == 3:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = img
        
        axes[idx].imshow(img_rgb)
        axes[idx].set_title(titles[idx] if titles else name)
        axes[idx].axis('off')
    
    # Hide extra subplots
    for idx in range(n_images, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Comparison saved to: {output_path}")


def generate_sample_visualizations(raw_folder: str, gt_folder: str, 
                                   enhanced_folder: str, output_folder: str,
                                   num_samples: int = 5):
    """
    Generate sample before/after visualizations.
    
    Args:
        raw_folder: Folder with raw images
        gt_folder: Folder with ground truth images
        enhanced_folder: Folder with enhanced images
        output_folder: Folder to save visualizations
        num_samples: Number of sample images to visualize
    """
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    # Get image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_files = [f for f in os.listdir(enhanced_folder) 
                  if Path(f).suffix.lower() in image_extensions]
    
    # Select random samples
    import random
    samples = random.sample(image_files, min(num_samples, len(image_files)))
    
    for filename in samples:
        raw_path = os.path.join(raw_folder, filename)
        gt_path = os.path.join(gt_folder, filename)
        enhanced_path = os.path.join(enhanced_folder, filename)
        
        if all(os.path.exists(p) for p in [raw_path, gt_path, enhanced_path]):
            raw = cv2.imread(raw_path)
            gt = cv2.imread(gt_path)
            enhanced = cv2.imread(enhanced_path)
            
            images = {
                'Raw': raw,
                'Ground Truth': gt,
                'Student Enhanced': enhanced
            }
            
            output_path = os.path.join(output_folder, 
                                      f"{Path(filename).stem}_comparison.png")
            create_comparison_grid(images, output_path, 
                                  ['Raw Input', 'Ground Truth', 'My Enhancement'])


def calculate_image_statistics(image: np.ndarray) -> dict:
    """
    Calculate various statistics for an image.
    
    Args:
        image: Input image
        
    Returns:
        Dictionary with statistics
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    stats = {
        'mean': np.mean(gray),
        'std': np.std(gray),
        'min': np.min(gray),
        'max': np.max(gray),
        'median': np.median(gray),
        'entropy': calculate_entropy(gray)
    }
    
    return stats


def calculate_entropy(image: np.ndarray) -> float:
    """
    Calculate Shannon entropy of an image.
    
    Args:
        image: Input image (grayscale)
        
    Returns:
        Entropy value
    """
    # Calculate histogram
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    hist = hist.flatten()
    
    # Normalize
    hist = hist / hist.sum()
    
    # Remove zeros
    hist = hist[hist > 0]
    
    # Calculate entropy
    entropy = -np.sum(hist * np.log2(hist))
    
    return entropy


def visualize_color_distribution(image: np.ndarray, output_path: str, 
                                 title: str = "Color Distribution"):
    """
    Visualize color distribution in 3D RGB space.
    
    Args:
        image: Input image
        output_path: Path to save visualization
        title: Plot title
    """
    from mpl_toolkits.mplot3d import Axes3D
    
    # Reshape image
    pixels = image.reshape(-1, 3)
    
    # Sample pixels for visualization (use all if small, sample if large)
    if len(pixels) > 10000:
        indices = np.random.choice(len(pixels), 10000, replace=False)
        pixels = pixels[indices]
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot
    ax.scatter(pixels[:, 2], pixels[:, 1], pixels[:, 0], 
              c=pixels/255.0, marker='.', s=1, alpha=0.5)
    
    ax.set_xlabel('Red')
    ax.set_ylabel('Green')
    ax.set_zlabel('Blue')
    ax.set_title(title)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_detailed_report_images(raw_path: str, gt_path: str, 
                                  enhanced_path: str, output_folder: str):
    """
    Create detailed visual report for a single image.
    
    Args:
        raw_path: Path to raw image
        gt_path: Path to ground truth
        enhanced_path: Path to enhanced image
        output_folder: Folder to save reports
    """
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    # Read images
    raw = cv2.imread(raw_path)
    gt = cv2.imread(gt_path)
    enhanced = cv2.imread(enhanced_path)
    
    filename = Path(raw_path).stem
    
    # 1. Side-by-side comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    axes[0].imshow(cv2.cvtColor(raw, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Raw Input', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(cv2.cvtColor(gt, cv2.COLOR_BGR2RGB))
    axes[1].set_title('Ground Truth', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    axes[2].imshow(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))
    axes[2].set_title('My Enhancement', fontsize=14, fontweight='bold')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f"{filename}_comparison.png"),
                dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Histogram comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    
    colors = ('b', 'g', 'r')
    labels = ('Blue', 'Green', 'Red')
    
    for idx, (img, title) in enumerate([(raw, 'Raw'), 
                                         (gt, 'Ground Truth'), 
                                         (enhanced, 'My Enhancement')]):
        for i, (color, label) in enumerate(zip(colors, labels)):
            hist = cv2.calcHist([img], [i], None, [256], [0, 256])
            axes[idx].plot(hist, color=color, alpha=0.7, label=label)
        
        axes[idx].set_title(f'{title} - Histogram', fontsize=12)
        axes[idx].set_xlabel('Pixel Value')
        axes[idx].set_ylabel('Frequency')
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f"{filename}_histograms.png"),
                dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Detailed report created for {filename}")


def batch_create_reports(raw_folder: str, gt_folder: str, 
                         enhanced_folder: str, report_folder: str,
                         num_samples: int = 10):
    """
    Create detailed reports for multiple images.
    
    Args:
        raw_folder: Folder with raw images
        gt_folder: Folder with ground truth
        enhanced_folder: Folder with enhanced images
        report_folder: Folder to save reports
        num_samples: Number of images to create reports for
    """
    Path(report_folder).mkdir(parents=True, exist_ok=True)
    
    # Get image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_files = [f for f in os.listdir(enhanced_folder) 
                  if Path(f).suffix.lower() in image_extensions]
    
    # Select samples
    import random
    samples = random.sample(image_files, min(num_samples, len(image_files)))
    
    print(f"\nCreating detailed reports for {len(samples)} images...")
    
    for filename in samples:
        raw_path = os.path.join(raw_folder, filename)
        gt_path = os.path.join(gt_folder, filename)
        enhanced_path = os.path.join(enhanced_folder, filename)
        
        if all(os.path.exists(p) for p in [raw_path, gt_path, enhanced_path]):
            create_detailed_report_images(raw_path, gt_path, enhanced_path, 
                                         report_folder)


def print_summary_table(summary_dict: dict):
    """
    Print a formatted summary table.

    Args:
        summary_dict: Dictionary with summary statistics
    """
    print("\n" + "="*70)
    print(f"{'Metric':<35} {'Value':>15}")
    print("="*70)

    for key, value in summary_dict.items():
        if isinstance(value, float):
            print(f"{key:<35} {value:>15.4f}")
        else:
            print(f"{key:<35} {str(value):>15}")

    print("="*70 + "\n")


def get_sparse_neighbor(p: int, n: int, m: int):
    """
    Get the neighbors of pixel p in a flattened image for sparse matrix construction.

    This function returns the 4-connected neighbors (up, down, left, right) of a pixel
    at position p in a flattened n x m image array.

    Args:
        p: pixel position in flattened array (0 to n*m-1)
        n: number of rows in the image
        m: number of columns in the image

    Returns:
        dict: Dictionary mapping neighbor positions to (row, col, is_horizontal) tuples
              where is_horizontal is True for left/right neighbors, False for up/down
    """
    i, j = p // m, p % m
    neighbors = {}

    # Right neighbor
    if j + 1 < m:
        neighbors[p + 1] = (i, j, True)

    # Left neighbor
    if j - 1 >= 0:
        neighbors[p - 1] = (i, j, True)

    # Down neighbor
    if i + 1 < n:
        neighbors[p + m] = (i, j, False)

    # Up neighbor
    if i - 1 >= 0:
        neighbors[p - m] = (i, j, False)

    return neighbors


if __name__ == "__main__":
    print("This is a utility module. Import it in your main scripts.")
    print("Available functions:")
    print("  - create_comparison_grid()")
    print("  - generate_sample_visualizations()")
    print("  - calculate_image_statistics()")
    print("  - visualize_color_distribution()")
    print("  - create_detailed_report_images()")
    print("  - batch_create_reports()")
    print("  - print_summary_table()")

"""
CDS6334 Visual Information Processing - Evaluation Script
Trimester 2530

This script evaluates enhanced images using:
1. Full-reference metrics: PSNR and SSIM
2. No-reference metrics: Sharpness (Variance of Laplacian) and Contrast (Std Dev)
"""

import cv2
import numpy as np
import os
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


class ImageEvaluator:
    """
    Evaluates image enhancement results using multiple metrics.
    """
    
    def __init__(self):
        """Initialize the evaluator."""
        self.results = {
            'hazy': [],
            'lowlight': []
        }
    
    # ==================== FULL-REFERENCE METRICS ====================
    
    def calculate_psnr(self, gt_image: np.ndarray, 
                       enhanced_image: np.ndarray) -> float:
        """
        Calculate Peak Signal-to-Noise Ratio between two images.
        
        Args:
            gt_image: Ground truth image
            enhanced_image: Enhanced image
            
        Returns:
            PSNR value in dB
        """
        return psnr(gt_image, enhanced_image)
    
    def calculate_ssim(self, gt_image: np.ndarray, 
                       enhanced_image: np.ndarray) -> float:
        """
        Calculate Structural Similarity Index between two images.
        
        Args:
            gt_image: Ground truth image
            enhanced_image: Enhanced image
            
        Returns:
            SSIM value (0-1)
        """
        # Convert to grayscale for SSIM calculation
        if len(gt_image.shape) == 3:
            gt_gray = cv2.cvtColor(gt_image, cv2.COLOR_BGR2GRAY)
            enhanced_gray = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2GRAY)
        else:
            gt_gray = gt_image
            enhanced_gray = enhanced_image
        
        return ssim(gt_gray, enhanced_gray, data_range=255)
    
    # ==================== NO-REFERENCE METRICS ====================
    
    def calculate_sharpness(self, image: np.ndarray) -> float:
        """
        Calculate sharpness using Variance of Laplacian.
        
        Higher values indicate sharper images.
        
        Args:
            image: Input image
            
        Returns:
            Sharpness score
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Calculate Laplacian
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        
        # Calculate variance
        sharpness = laplacian.var()
        
        return sharpness
    
    def calculate_contrast(self, image: np.ndarray) -> float:
        """
        Calculate contrast using Standard Deviation of Intensity.
        
        Higher values indicate better contrast.
        
        Args:
            image: Input image
            
        Returns:
            Contrast score
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Calculate standard deviation
        contrast = np.std(gray)
        
        return contrast
    
    # ==================== HISTOGRAM ANALYSIS ====================
    
    def plot_histogram_comparison(self, original: np.ndarray, 
                                   enhanced: np.ndarray,
                                   output_path: str,
                                   title: str = "Histogram Comparison"):
        """
        Plot before and after histograms side by side.
        
        Args:
            original: Original image
            enhanced: Enhanced image
            output_path: Path to save the plot
            title: Plot title
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(title, fontsize=16)
        
        # Original image
        axes[0, 0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Enhanced image
        axes[0, 1].imshow(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))
        axes[0, 1].set_title('Enhanced Image')
        axes[0, 1].axis('off')
        
        # Original histogram
        colors = ('b', 'g', 'r')
        for i, color in enumerate(colors):
            hist = cv2.calcHist([original], [i], None, [256], [0, 256])
            axes[1, 0].plot(hist, color=color, alpha=0.7)
        axes[1, 0].set_title('Original Histogram')
        axes[1, 0].set_xlabel('Pixel Value')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Enhanced histogram
        for i, color in enumerate(colors):
            hist = cv2.calcHist([enhanced], [i], None, [256], [0, 256])
            axes[1, 1].plot(hist, color=color, alpha=0.7)
        axes[1, 1].set_title('Enhanced Histogram')
        axes[1, 1].set_xlabel('Pixel Value')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    # ==================== EVALUATION PIPELINE ====================
    
    def evaluate_image(self, original_path: str, gt_path: str, 
                       enhanced_path: str, image_name: str,
                       category: str) -> Dict:
        """
        Evaluate a single image with all metrics.
        
        Args:
            original_path: Path to original (raw) image
            gt_path: Path to ground truth enhanced image
            enhanced_path: Path to student's enhanced image
            image_name: Name of the image
            category: 'hazy' or 'lowlight'
            
        Returns:
            Dictionary containing all metrics
        """
        # Read images
        original = cv2.imread(original_path)
        gt = cv2.imread(gt_path)
        enhanced = cv2.imread(enhanced_path)
        
        if original is None or gt is None or enhanced is None:
            print(f"Error reading images for {image_name}")
            return None
        
        # Ensure images are the same size
        if gt.shape != enhanced.shape:
            enhanced = cv2.resize(enhanced, (gt.shape[1], gt.shape[0]))
        
        # Calculate metrics
        results = {
            'Image': image_name,
            'Category': category,
            
            # Full-reference metrics
            'PSNR': self.calculate_psnr(gt, enhanced),
            'SSIM': self.calculate_ssim(gt, enhanced),
            
            # No-reference metrics for GT
            'Sharpness_GT': self.calculate_sharpness(gt),
            'Contrast_GT': self.calculate_contrast(gt),
            
            # No-reference metrics for Student output
            'Sharpness_Student': self.calculate_sharpness(enhanced),
            'Contrast_Student': self.calculate_contrast(enhanced)
        }
        
        return results
    
    def evaluate_folder(self, raw_folder: str, gt_folder: str, 
                        enhanced_folder: str, category: str,
                        visualization_folder: str = None) -> pd.DataFrame:
        """
        Evaluate all images in a folder.
        
        Args:
            raw_folder: Folder with raw input images
            gt_folder: Folder with ground truth images
            enhanced_folder: Folder with student-enhanced images
            category: 'hazy' or 'lowlight'
            visualization_folder: Optional folder to save visualizations
            
        Returns:
            DataFrame with all results
        """
        if not os.path.exists(enhanced_folder):
            print(f"Error: Enhanced folder not found: {enhanced_folder}")
            return pd.DataFrame()
        
        # Get all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = [f for f in os.listdir(enhanced_folder) 
                      if Path(f).suffix.lower() in image_extensions]
        
        print(f"\n{'='*60}")
        print(f"Evaluating {category.upper()} images")
        print(f"Found {len(image_files)} enhanced images")
        print(f"{'='*60}\n")
        
        results = []
        
        for filename in image_files:
            # Construct paths
            raw_path = os.path.join(raw_folder, filename)
            gt_path = os.path.join(gt_folder, filename)
            enhanced_path = os.path.join(enhanced_folder, filename)
            
            # Check if all files exist
            if not all(os.path.exists(p) for p in [raw_path, gt_path, enhanced_path]):
                print(f"⚠ Missing file for {filename}, skipping...")
                continue
            
            # Evaluate
            result = self.evaluate_image(raw_path, gt_path, enhanced_path, 
                                        filename, category)
            if result:
                results.append(result)
                print(f"✓ Evaluated: {filename}")
                
                # Generate visualization if requested
                if visualization_folder:
                    viz_path = os.path.join(visualization_folder, 
                                           f"{Path(filename).stem}_comparison.png")
                    Path(visualization_folder).mkdir(parents=True, exist_ok=True)
                    
                    original = cv2.imread(raw_path)
                    enhanced = cv2.imread(enhanced_path)
                    self.plot_histogram_comparison(original, enhanced, viz_path,
                                                   f"{filename} - Before/After")
        
        df = pd.DataFrame(results)
        self.results[category] = results
        
        return df
    
    def generate_summary_report(self, df: pd.DataFrame, category: str) -> Dict:
        """
        Generate summary statistics for a category.
        
        Args:
            df: DataFrame with evaluation results
            category: 'hazy' or 'lowlight'
            
        Returns:
            Dictionary with summary statistics
        """
        if df.empty:
            return {}
        
        summary = {
            'Category': category,
            'Total Images': len(df),
            
            # Average metrics
            'Avg_PSNR': df['PSNR'].mean(),
            'Avg_SSIM': df['SSIM'].mean(),
            'Avg_Sharpness_GT': df['Sharpness_GT'].mean(),
            'Avg_Sharpness_Student': df['Sharpness_Student'].mean(),
            'Avg_Contrast_GT': df['Contrast_GT'].mean(),
            'Avg_Contrast_Student': df['Contrast_Student'].mean(),
            
            # Standard deviations
            'Std_PSNR': df['PSNR'].std(),
            'Std_SSIM': df['SSIM'].std(),
            
            # Min/Max
            'Min_PSNR': df['PSNR'].min(),
            'Max_PSNR': df['PSNR'].max(),
            'Min_SSIM': df['SSIM'].min(),
            'Max_SSIM': df['SSIM'].max()
        }
        
        return summary
    
    def save_results_to_excel(self, output_path: str):
        """
        Save all results to an Excel file with multiple sheets.

        Args:
            output_path: Path to save Excel file
        """
        # Check if there are any results to save
        if not self.results['hazy'] and not self.results['lowlight']:
            print(f"\n⚠ No results to save. Skipping Excel file creation.")
            return

        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Hazy results
            if self.results['hazy']:
                df_hazy = pd.DataFrame(self.results['hazy'])
                df_hazy.to_excel(writer, sheet_name='Hazy_Details', index=False)

                summary_hazy = self.generate_summary_report(df_hazy, 'hazy')
                pd.DataFrame([summary_hazy]).to_excel(writer,
                                                       sheet_name='Hazy_Summary',
                                                       index=False)

            # Low-light results
            if self.results['lowlight']:
                df_lowlight = pd.DataFrame(self.results['lowlight'])
                df_lowlight.to_excel(writer, sheet_name='LowLight_Details',
                                     index=False)

                summary_lowlight = self.generate_summary_report(df_lowlight,
                                                                 'lowlight')
                pd.DataFrame([summary_lowlight]).to_excel(writer,
                                                          sheet_name='LowLight_Summary',
                                                          index=False)

        print(f"\n✓ Results saved to: {output_path}")


def main():
    """
    Main evaluation function.
    """
    # Initialize evaluator
    evaluator = ImageEvaluator()
    
    # Define paths - ADJUST THESE TO YOUR FOLDER STRUCTURE
    base_dataset = r"C:\Users\muham\Documents\VIP Indivudal Assignment\Datset"
    base_output = "output"
    
    # Hazy images paths
    hazy_raw = os.path.join(base_dataset, "01. Hazy - Raw")
    hazy_gt = os.path.join(base_dataset, "01. Hazy - Enhanced (GT)")
    hazy_enhanced = os.path.join(base_output, "hazy-student-enhanced")
    hazy_viz = os.path.join(base_output, "visualizations", "hazy")
    
    # Low-light images paths
    lowlight_raw = os.path.join(base_dataset, "02. Low Light - Raw")
    lowlight_gt = os.path.join(base_dataset, "02. Low Light - Enhanced (GT)")
    lowlight_enhanced = os.path.join(base_output, "lowlight-student-enhanced")
    lowlight_viz = os.path.join(base_output, "visualizations", "lowlight")
    
    print("\n" + "="*60)
    print("CDS6334 Image Enhancement - Evaluation")
    print("="*60)
    
    # Evaluate hazy images
    if os.path.exists(hazy_enhanced):
        df_hazy = evaluator.evaluate_folder(hazy_raw, hazy_gt, hazy_enhanced, 
                                            'hazy', hazy_viz)
        
        if not df_hazy.empty:
            print("\n" + "-"*60)
            print("HAZY IMAGES - Summary Statistics")
            print("-"*60)
            summary_hazy = evaluator.generate_summary_report(df_hazy, 'hazy')
            for key, value in summary_hazy.items():
                if isinstance(value, float):
                    print(f"{key:30s}: {value:.4f}")
                else:
                    print(f"{key:30s}: {value}")
    else:
        print(f"Warning: Hazy enhanced folder not found: {hazy_enhanced}")
    
    # Evaluate low-light images
    if os.path.exists(lowlight_enhanced):
        df_lowlight = evaluator.evaluate_folder(lowlight_raw, lowlight_gt, 
                                                lowlight_enhanced, 'lowlight',
                                                lowlight_viz)
        
        if not df_lowlight.empty:
            print("\n" + "-"*60)
            print("LOW-LIGHT IMAGES - Summary Statistics")
            print("-"*60)
            summary_lowlight = evaluator.generate_summary_report(df_lowlight, 
                                                                  'lowlight')
            for key, value in summary_lowlight.items():
                if isinstance(value, float):
                    print(f"{key:30s}: {value:.4f}")
                else:
                    print(f"{key:30s}: {value}")
    else:
        print(f"Warning: Low-light enhanced folder not found: {lowlight_enhanced}")
    
    # Save all results to Excel
    excel_path = os.path.join(base_output, "evaluation_results.xlsx")
    evaluator.save_results_to_excel(excel_path)
    
    print("\n" + "="*60)
    print("✓ Evaluation complete!")
    print("="*60)


if __name__ == "__main__":
    main()

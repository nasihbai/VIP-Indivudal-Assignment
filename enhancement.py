"""
CDS6334 Visual Information Processing - Image Enhancement Assignment
Student: [Your Name]
Trimester 2530

This script implements traditional (non-deep learning) image enhancement techniques
for both hazy and low-light images using classical image processing methods.
"""

import cv2
import numpy as np
import os
from pathlib import Path
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class ImageEnhancer:
    """
    A class to handle image enhancement using traditional image processing techniques.
    Supports both hazy image dehazing and low-light image enhancement.
    """
    
    def __init__(self):
        """Initialize the ImageEnhancer with default parameters."""
        self.hazy_params = {
            'omega': 0.95,  # Dehazing strength
            'radius': 15,   # Dark channel radius
            't0': 0.1,      # Minimum transmission
            'guided_r': 60, # Guided filter radius
            'guided_eps': 0.001  # Guided filter regularization
        }
        
        self.lowlight_params = {
            'gamma': 1.5,   # Gamma correction
            'alpha': 1.3,   # Contrast enhancement
            'beta': 10,     # Brightness adjustment
            'clahe_clip': 2.5,  # CLAHE clip limit
            'clahe_grid': (8, 8)  # CLAHE grid size
        }
    
    # ==================== HAZY IMAGE ENHANCEMENT ====================
    
    def enhance_hazy_image(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance a hazy image using Dark Channel Prior-based dehazing.
        
        This method implements a simplified version of the Dark Channel Prior
        algorithm combined with guided filtering for refined transmission estimation.
        
        Args:
            image: Input hazy image (BGR format)
            
        Returns:
            Enhanced dehazed image
        """
        # Normalize image to [0, 1]
        img_normalized = image.astype(np.float64) / 255.0
        
        # Step 1: Estimate atmospheric light
        atmospheric_light = self._estimate_atmospheric_light(img_normalized)
        
        # Step 2: Compute dark channel
        dark_channel = self._get_dark_channel(img_normalized, 
                                               self.hazy_params['radius'])
        
        # Step 3: Estimate transmission map
        transmission = self._estimate_transmission(img_normalized, 
                                                     atmospheric_light, 
                                                     dark_channel)
        
        # Step 4: Refine transmission using guided filter
        transmission_refined = self._guided_filter(img_normalized[:, :, 0], 
                                                     transmission,
                                                     self.hazy_params['guided_r'],
                                                     self.hazy_params['guided_eps'])
        
        # Step 5: Recover scene radiance (dehaze)
        dehazed = self._recover_scene_radiance(img_normalized, 
                                                 transmission_refined, 
                                                 atmospheric_light)
        
        # Step 6: Post-processing enhancement
        dehazed = self._post_process_hazy(dehazed)
        
        # Convert back to uint8
        result = np.clip(dehazed * 255, 0, 255).astype(np.uint8)
        
        return result
    
    def _get_dark_channel(self, image: np.ndarray, radius: int) -> np.ndarray:
        """
        Compute the dark channel of an image.
        
        The dark channel is defined as the minimum pixel value in each color channel
        within a local patch.
        
        Args:
            image: Input image (normalized to [0, 1])
            radius: Patch radius
            
        Returns:
            Dark channel map
        """
        # Get minimum across color channels
        min_channel = np.min(image, axis=2)
        
        # Apply minimum filter (erosion)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (radius, radius))
        dark_channel = cv2.erode(min_channel, kernel)
        
        return dark_channel
    
    def _estimate_atmospheric_light(self, image: np.ndarray, 
                                     top_percent: float = 0.001) -> np.ndarray:
        """
        Estimate atmospheric light from the haziest region.
        
        Args:
            image: Input image (normalized)
            top_percent: Percentage of brightest pixels to consider
            
        Returns:
            Atmospheric light value (3-channel)
        """
        # Get dark channel
        dark_channel = self._get_dark_channel(image, 15)
        
        # Find top brightest pixels in dark channel
        num_pixels = int(dark_channel.size * top_percent)
        flat_dark = dark_channel.flatten()
        indices = np.argpartition(flat_dark, -num_pixels)[-num_pixels:]
        
        # Get corresponding pixels in original image
        h, w = dark_channel.shape
        indices_2d = np.unravel_index(indices, (h, w))
        
        # Find pixel with maximum intensity among top bright pixels
        atmospheric_light = np.zeros(3)
        max_intensity = 0
        
        for i in range(len(indices)):
            y, x = indices_2d[0][i], indices_2d[1][i]
            intensity = np.sum(image[y, x, :])
            if intensity > max_intensity:
                max_intensity = intensity
                atmospheric_light = image[y, x, :]
        
        return atmospheric_light
    
    def _estimate_transmission(self, image: np.ndarray, 
                                atmospheric_light: np.ndarray,
                                dark_channel: np.ndarray) -> np.ndarray:
        """
        Estimate transmission map using dark channel prior.
        
        Args:
            image: Input image
            atmospheric_light: Estimated atmospheric light
            dark_channel: Pre-computed dark channel
            
        Returns:
            Transmission map
        """
        omega = self.hazy_params['omega']
        
        # Normalize by atmospheric light
        normalized = np.zeros_like(image)
        for i in range(3):
            normalized[:, :, i] = image[:, :, i] / atmospheric_light[i]
        
        # Compute transmission
        transmission = 1 - omega * self._get_dark_channel(normalized, 
                                                           self.hazy_params['radius'])
        
        return transmission
    
    def _guided_filter(self, guide: np.ndarray, src: np.ndarray, 
                       radius: int, eps: float) -> np.ndarray:
        """
        Apply guided filter for edge-preserving smoothing.
        
        Args:
            guide: Guidance image
            src: Source image to filter
            radius: Filter radius
            eps: Regularization parameter
            
        Returns:
            Filtered image
        """
        mean_guide = cv2.boxFilter(guide, cv2.CV_64F, (radius, radius))
        mean_src = cv2.boxFilter(src, cv2.CV_64F, (radius, radius))
        corr_guide = cv2.boxFilter(guide * guide, cv2.CV_64F, (radius, radius))
        corr_guide_src = cv2.boxFilter(guide * src, cv2.CV_64F, (radius, radius))
        
        var_guide = corr_guide - mean_guide * mean_guide
        cov_guide_src = corr_guide_src - mean_guide * mean_src
        
        a = cov_guide_src / (var_guide + eps)
        b = mean_src - a * mean_guide
        
        mean_a = cv2.boxFilter(a, cv2.CV_64F, (radius, radius))
        mean_b = cv2.boxFilter(b, cv2.CV_64F, (radius, radius))
        
        return mean_a * guide + mean_b
    
    def _recover_scene_radiance(self, image: np.ndarray, 
                                 transmission: np.ndarray,
                                 atmospheric_light: np.ndarray) -> np.ndarray:
        """
        Recover the scene radiance (dehaze the image).
        
        Args:
            image: Input hazy image
            transmission: Transmission map
            atmospheric_light: Atmospheric light
            
        Returns:
            Dehazed image
        """
        t0 = self.hazy_params['t0']
        
        # Ensure minimum transmission
        transmission = np.maximum(transmission, t0)
        
        # Recover radiance
        result = np.zeros_like(image)
        for i in range(3):
            result[:, :, i] = (image[:, :, i] - atmospheric_light[i]) / \
                              transmission + atmospheric_light[i]
        
        return result
    
    def _post_process_hazy(self, image: np.ndarray) -> np.ndarray:
        """
        Post-process dehazed image for better visual quality.
        
        Args:
            image: Dehazed image
            
        Returns:
            Enhanced image
        """
        # Clip values
        image = np.clip(image, 0, 1)
        
        # Convert to uint8 for CLAHE
        img_uint8 = (image * 255).astype(np.uint8)
        
        # Apply CLAHE for contrast enhancement
        lab = cv2.cvtColor(img_uint8, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        lab = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # Convert back to float
        enhanced = enhanced.astype(np.float64) / 255.0
        
        # Slight sharpening
        enhanced = self._apply_sharpening(enhanced, strength=0.5)
        
        return enhanced
    
    # ==================== LOW-LIGHT IMAGE ENHANCEMENT ====================
    
    def enhance_lowlight_image(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance a low-light image using multiple traditional techniques.
        
        This method combines:
        1. Adaptive histogram equalization (CLAHE)
        2. Gamma correction
        3. Bilateral filtering for noise reduction
        4. Color correction
        5. Sharpening
        
        Args:
            image: Input low-light image (BGR format)
            
        Returns:
            Enhanced bright image
        """
        # Step 1: Convert to LAB color space for better processing
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Step 2: Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=self.lowlight_params['clahe_clip'],
                                tileGridSize=self.lowlight_params['clahe_grid'])
        l_enhanced = clahe.apply(l)
        
        # Merge back
        lab_enhanced = cv2.merge([l_enhanced, a, b])
        enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
        
        # Step 3: Apply gamma correction for brightness
        enhanced = self._apply_gamma_correction(enhanced, 
                                                 self.lowlight_params['gamma'])
        
        # Step 4: Bilateral filter for noise reduction while preserving edges
        enhanced = cv2.bilateralFilter(enhanced, d=9, sigmaColor=75, sigmaSpace=75)
        
        # Step 5: Enhance contrast
        enhanced = self._enhance_contrast(enhanced,
                                          alpha=self.lowlight_params['alpha'],
                                          beta=self.lowlight_params['beta'])
        
        # Step 6: Color correction to restore natural colors
        enhanced = self._correct_color_cast(enhanced)
        
        # Step 7: Apply sharpening
        enhanced = self._apply_sharpening(enhanced, strength=0.8)
        
        # Step 8: Final normalization
        enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
        
        return enhanced
    
    def _apply_gamma_correction(self, image: np.ndarray, 
                                 gamma: float) -> np.ndarray:
        """
        Apply gamma correction to adjust brightness.
        
        Args:
            image: Input image
            gamma: Gamma value (>1 brightens, <1 darkens)
            
        Returns:
            Gamma-corrected image
        """
        # Build lookup table
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 
                         for i in range(256)]).astype(np.uint8)
        
        # Apply lookup table
        return cv2.LUT(image, table)
    
    def _enhance_contrast(self, image: np.ndarray, 
                          alpha: float = 1.3, 
                          beta: int = 10) -> np.ndarray:
        """
        Enhance contrast using linear transformation.
        
        Args:
            image: Input image
            alpha: Contrast control (1.0-3.0)
            beta: Brightness control
            
        Returns:
            Contrast-enhanced image
        """
        adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        return adjusted
    
    def _correct_color_cast(self, image: np.ndarray) -> np.ndarray:
        """
        Correct color cast using Gray World assumption.
        
        Args:
            image: Input image
            
        Returns:
            Color-corrected image
        """
        result = image.astype(np.float64)
        
        # Calculate mean of each channel
        avg_b = np.mean(result[:, :, 0])
        avg_g = np.mean(result[:, :, 1])
        avg_r = np.mean(result[:, :, 2])
        
        # Calculate gray reference
        avg_gray = (avg_b + avg_g + avg_r) / 3
        
        # Scale each channel
        if avg_b > 0:
            result[:, :, 0] = result[:, :, 0] * (avg_gray / avg_b)
        if avg_g > 0:
            result[:, :, 1] = result[:, :, 1] * (avg_gray / avg_g)
        if avg_r > 0:
            result[:, :, 2] = result[:, :, 2] * (avg_gray / avg_r)
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def _apply_sharpening(self, image: np.ndarray, 
                          strength: float = 1.0) -> np.ndarray:
        """
        Apply unsharp masking for image sharpening.
        
        Args:
            image: Input image (float or uint8)
            strength: Sharpening strength
            
        Returns:
            Sharpened image
        """
        # Determine if input is float or uint8
        is_float = image.dtype == np.float64 or image.dtype == np.float32
        
        if is_float:
            img_process = (image * 255).astype(np.uint8)
        else:
            img_process = image
        
        # Create Gaussian blur
        blurred = cv2.GaussianBlur(img_process, (0, 0), 3)
        
        # Unsharp mask
        sharpened = cv2.addWeighted(img_process, 1.0 + strength, 
                                     blurred, -strength, 0)
        
        if is_float:
            return sharpened.astype(np.float64) / 255.0
        else:
            return sharpened
    
    # ==================== UTILITY FUNCTIONS ====================
    
    def process_image(self, input_path: str, output_path: str, 
                      image_type: str) -> bool:
        """
        Process a single image file.
        
        Args:
            input_path: Path to input image
            output_path: Path to save enhanced image
            image_type: 'hazy' or 'lowlight'
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Read image
            image = cv2.imread(input_path)
            if image is None:
                print(f"Error: Could not read image {input_path}")
                return False
            
            # Enhance based on type
            if image_type == 'hazy':
                enhanced = self.enhance_hazy_image(image)
            elif image_type == 'lowlight':
                enhanced = self.enhance_lowlight_image(image)
            else:
                print(f"Error: Unknown image type {image_type}")
                return False
            
            # Save result
            cv2.imwrite(output_path, enhanced)
            print(f"✓ Processed: {os.path.basename(input_path)}")
            return True
            
        except Exception as e:
            print(f"✗ Error processing {input_path}: {str(e)}")
            return False
    
    def process_folder(self, input_folder: str, output_folder: str, 
                       image_type: str) -> int:
        """
        Process all images in a folder.
        
        Args:
            input_folder: Input folder path
            output_folder: Output folder path
            image_type: 'hazy' or 'lowlight'
            
        Returns:
            Number of successfully processed images
        """
        # Create output folder if it doesn't exist
        Path(output_folder).mkdir(parents=True, exist_ok=True)
        
        # Get all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = [f for f in os.listdir(input_folder) 
                      if Path(f).suffix.lower() in image_extensions]
        
        print(f"\n{'='*60}")
        print(f"Processing {image_type.upper()} images")
        print(f"Found {len(image_files)} images in {input_folder}")
        print(f"{'='*60}\n")
        
        success_count = 0
        for filename in image_files:
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            
            if self.process_image(input_path, output_path, image_type):
                success_count += 1
        
        print(f"\n{'='*60}")
        print(f"Completed: {success_count}/{len(image_files)} images processed")
        print(f"{'='*60}\n")
        
        return success_count


def main():
    """
    Main function to process all images.
    """
    # Initialize enhancer
    enhancer = ImageEnhancer()
    
    # Define paths - ADJUST THESE TO YOUR DATASET LOCATION
    base_input_path = "dataset"  # Adjust this to your dataset folder
    base_output_path = "output"
    
    # Hazy images
    hazy_input = os.path.join(base_input_path, "01. Hazy - Raw")
    hazy_output = os.path.join(base_output_path, "hazy-student-enhanced")
    
    # Low-light images
    lowlight_input = os.path.join(base_input_path, "02. Low Light - Raw")
    lowlight_output = os.path.join(base_output_path, "lowlight-student-enhanced")
    
    print("\n" + "="*60)
    print("CDS6334 Image Enhancement - Traditional Methods")
    print("="*60)
    
    # Process hazy images
    if os.path.exists(hazy_input):
        enhancer.process_folder(hazy_input, hazy_output, 'hazy')
    else:
        print(f"Warning: Hazy input folder not found: {hazy_input}")
    
    # Process low-light images
    if os.path.exists(lowlight_input):
        enhancer.process_folder(lowlight_input, lowlight_output, 'lowlight')
    else:
        print(f"Warning: Low-light input folder not found: {lowlight_input}")
    
    print("\n✓ All processing complete!")
    print(f"Output saved to: {base_output_path}")


if __name__ == "__main__":
    main()

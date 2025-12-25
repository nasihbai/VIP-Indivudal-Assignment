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
from scipy.spatial import distance
from scipy.ndimage import convolve
from scipy.sparse import diags, csr_matrix
from scipy.sparse.linalg import spsolve
from utils import get_sparse_neighbor

warnings.filterwarnings('ignore')


class ImageEnhancer:
    """
    A class to handle image enhancement using traditional image processing techniques.
    Supports both hazy image dehazing and low-light image enhancement.
    """
    
    def __init__(self):
        """Initialize the ImageEnhancer with default parameters."""
        self.hazy_params = {
            'omega': 0.95,         # Dehazing strength (amount of haze to remove, 0-1)
            'radius': 15,          # Dark channel patch size (MUST match reference: 15)
            't0': 0.1,             # Minimum transmission threshold
            'guided_r': 20,        # Guided filter radius (soft matting)
            'guided_eps': 0.01,    # Guided filter regularization (10e-3)
            'color_mode': 'LAB',   # Color enhancement: 'LAB', 'YUV', or 'BOTH'
            'adaptive': True,      # Enable adaptive parameter tuning per image
            'saturation_scale': 0.9  # Saturation multiplier (0-2: <1 reduces, >1 increases, 1=unchanged)
        }
        
        self.lowlight_params = {
            'gamma': 0.55,     # Gamma correction for illumination refinement (lower = brighter)
            'lambda_': 0.15,   # Balance coefficient for optimization
            'sigma': 3,        # Spatial standard deviation for Gaussian weights
            'dual': True,      # Use DUAL method (True) or LIME method (False)
            'bc': 1,           # Mertens contrast measure weight
            'bs': 1,           # Mertens saturation measure weight
            'be': 1,           # Mertens well-exposedness measure weight
            'eps': 1e-3,       # Small constant for stability
            'post_gamma': 1.35,  # Post-processing brightness boost
            'denoise': False,  # Disabled - filtering doesn't help
            'median_kernel': 5,  # Median filter kernel size (3, 5, or 7)
            'bilateral_d': 9,    # Bilateral filter diameter
            'bilateral_sigma': 10  # Bilateral filter sigma
        }
    
    # ==================== HAZY IMAGE ENHANCEMENT ====================
    
    def enhance_hazy_image(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance a hazy image using adaptive Dark Channel Prior with guided filtering.

        This method implements the complete Dark Channel Prior algorithm with:
        1. Adaptive parameter selection based on haze level (optional)
        2. Dark channel prior computation
        3. Atmospheric light estimation
        4. Transmission map estimation
        5. Guided filter refinement (soft matting)
        6. Scene radiance recovery
        7. Flexible color enhancement (LAB/YUV/Both)
        8. Saturation adjustment (reduces oversaturation)

        Args:
            image: Input hazy image (BGR format)

        Returns:
            Enhanced dehazed image
        """
        # Step 0: Adaptive parameter tuning (if enabled)
        if self.hazy_params['adaptive']:
            params = self._adapt_hazy_parameters(image)
        else:
            params = self.hazy_params.copy()

        # Step 1: Compute dark channel
        dark_channel = self._compute_dark_channel_optimized(
            image,
            params['radius']
        )

        # Step 2: Estimate atmospheric light
        atmospheric_light = self._estimate_atmospheric_light_optimized(
            image,
            dark_channel,
            topPercent=0.001
        )

        # Step 3: Estimate transmission map
        transmission_map = self._estimate_transmission_optimized(
            image,
            atmospheric_light,
            params['omega'],
            params['radius']
        )

        # Step 4: Refine transmission using guided filter (soft matting)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        refined_transmission = self._guided_filter(
            gray,
            transmission_map,
            params['guided_r'],
            params['guided_eps']
        )

        # Step 5: Recover scene radiance
        radiance_map = self._recover_radiance_optimized(
            image,
            atmospheric_light,
            refined_transmission,
            params['t0']
        )

        # Step 6: Color enhancement (LAB, YUV, or both)
        enhanced = self._apply_color_enhancement(radiance_map, params['color_mode'])

        # Step 7: Adjust saturation (if scale != 1.0)
        if params['saturation_scale'] != 1.0:
            enhanced = self._adjust_saturation(enhanced, params['saturation_scale'])

        return enhanced
    
    def _compute_dark_channel_optimized(self, image: np.ndarray, patchSize: int) -> np.ndarray:
        """
        Compute the dark channel prior (OPTIMIZED - vectorized).

        Args:
            image: Input image (BGR, uint8 or float32)
            patchSize: Patch size for local minimum

        Returns:
            Dark channel map (float32)
        """
        # Convert to float32 if needed
        if image.dtype == np.uint8:
            img_float = image.astype(np.float32)
        else:
            img_float = image

        height, width = img_float.shape[:2]

        # Step 1: Compute minimum across color channels for each pixel
        min_channels = np.min(img_float, axis=2)

        # Step 2: Apply minimum filter over local patches
        pad = patchSize // 2
        padded = np.pad(min_channels, ((pad, pad), (pad, pad)), mode='edge')

        dark_channel = np.zeros((height, width), dtype=np.float32)
        for i in range(height):
            for j in range(width):
                patch = padded[i:i + patchSize, j:j + patchSize]
                dark_channel[i, j] = np.min(patch)

        return dark_channel
    
    def _estimate_atmospheric_light_optimized(self, image: np.ndarray,
                                              dark_channel: np.ndarray,
                                              topPercent: float = 0.001) -> np.ndarray:
        """
        Estimate atmospheric light (OPTIMIZED).

        Args:
            image: Input image (BGR, uint8)
            dark_channel: Dark channel prior
            topPercent: Top percentage of brightest pixels

        Returns:
            Atmospheric light (3-channel array)
        """
        height, width = image.shape[:2]
        flat_img = image.reshape(height * width, 3)
        flat_dark = dark_channel.flatten()

        # Get top percent brightest pixels in dark channel
        num_pixels = int(height * width * topPercent)
        sorted_indices = np.argsort(-flat_dark)  # Sort in descending order
        top_indices = sorted_indices[:num_pixels]

        # Find max intensity for each channel
        atmospheric_light = np.zeros(3, dtype=np.float32)
        for idx in top_indices:
            pixel = flat_img[idx]
            for c in range(3):
                if pixel[c] > atmospheric_light[c]:
                    atmospheric_light[c] = pixel[c]

        return atmospheric_light
    
    def _estimate_transmission_optimized(self, image: np.ndarray,
                                         atmospheric_light: np.ndarray,
                                         omega: float,
                                         patchSize: int) -> np.ndarray:
        """
        Estimate transmission map (OPTIMIZED).

        Args:
            image: Input image (BGR, uint8)
            atmospheric_light: Atmospheric light
            omega: Parameter (0-1) for haze retention
            patchSize: Patch size

        Returns:
            Transmission map (float32)
        """
        # Normalize image by atmospheric light to range [0, 1]
        normalized_img = image.astype(np.float32) / (atmospheric_light + 1e-6)

        # Compute dark channel of normalized image (stays in float32)
        dark_channel = self._compute_dark_channel_optimized(
            normalized_img,
            patchSize
        )

        # Transmission estimation (eq. 12 from paper)
        # dark_channel is already in [0, 1] range since normalized_img is [0, 1]
        transmission_map = 1 - omega * dark_channel
        transmission_map = transmission_map.astype(np.float32)

        return transmission_map
    
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
    
    def _recover_radiance_optimized(self, image: np.ndarray,
                                    atmospheric_light: np.ndarray,
                                    transmission: np.ndarray,
                                    t0: float) -> np.ndarray:
        """
        Recover scene radiance (OPTIMIZED).

        Args:
            image: Input hazy image (BGR, uint8)
            atmospheric_light: Atmospheric light
            transmission: Refined transmission map
            t0: Minimum transmission threshold

        Returns:
            Dehazed image (uint8)
        """
        height, width = transmission.shape
        num_channels = image.shape[2]

        # Clip transmission to avoid division by zero
        transmission_clipped = np.clip(transmission, t0, 1.0)

        # Expand transmission to 3 channels
        transmission_3d = np.zeros_like(image, dtype=np.float32)
        for c in range(num_channels):
            transmission_3d[:, :, c] = transmission_clipped

        # Recover radiance: J(x) = (I(x) - A) / t(x) + A
        radiance = (image.astype(np.float32) - atmospheric_light) / transmission_3d + atmospheric_light

        # Clip to valid range
        radiance = np.clip(radiance, 0, 255).astype(np.uint8)

        return radiance
    
    def _detect_haze_level(self, image: np.ndarray, dark_channel: np.ndarray = None) -> float:
        """
        Detect the haze level in an image.

        Args:
            image: Input image (BGR, uint8)
            dark_channel: Pre-computed dark channel (optional)

        Returns:
            Haze level (0-1, higher = more haze)
        """
        if dark_channel is None:
            dark_channel = self._compute_dark_channel_optimized(image, 15)

        # Haze level is estimated by average dark channel value
        # Low dark channel = high haze
        avg_dark = np.mean(dark_channel) / 255.0
        haze_level = 1.0 - avg_dark

        return haze_level

    def _adapt_hazy_parameters(self, image: np.ndarray) -> dict:
        """
        Adaptively tune parameters based on image haze level.

        Args:
            image: Input hazy image

        Returns:
            Adapted parameters dictionary
        """
        params = self.hazy_params.copy()

        # Detect haze level
        dark_channel = self._compute_dark_channel_optimized(image, 15)
        haze_level = self._detect_haze_level(image, dark_channel)

        # Adapt parameters based on haze level
        if haze_level > 0.7:  # Heavy haze
            params['omega'] = 0.98
            params['guided_r'] = 25
            params['guided_eps'] = 0.02
        elif haze_level > 0.5:  # Moderate haze
            params['omega'] = 0.95
            params['guided_r'] = 20
            params['guided_eps'] = 0.01
        else:  # Light haze
            params['omega'] = 0.90
            params['guided_r'] = 15
            params['guided_eps'] = 0.005

        return params

    def _enhance_lab_color(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance image using LAB color space with histogram equalization.

        Args:
            image: Input image (BGR, uint8)

        Returns:
            Enhanced image (uint8)
        """
        # Convert to LAB color space
        lab_img = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

        # Apply histogram equalization to L channel
        lab_img[:, :, 0] = cv2.equalizeHist(lab_img[:, :, 0])

        # Convert back to BGR
        enhanced = cv2.cvtColor(lab_img, cv2.COLOR_LAB2BGR)

        return enhanced

    def _enhance_yuv_color(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance image using YUV color space with histogram equalization.

        Args:
            image: Input image (BGR, uint8)

        Returns:
            Enhanced image (uint8)
        """
        # Convert to YUV color space
        yuv_img = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)

        # Apply histogram equalization to Y channel (luminance)
        yuv_img[:, :, 0] = cv2.equalizeHist(yuv_img[:, :, 0])

        # Convert back to BGR
        enhanced = cv2.cvtColor(yuv_img, cv2.COLOR_YUV2BGR)

        return enhanced

    def _apply_color_enhancement(self, image: np.ndarray, mode: str = 'LAB') -> np.ndarray:
        """
        Apply color enhancement based on selected mode.

        Args:
            image: Input image (BGR, uint8)
            mode: Enhancement mode ('LAB', 'YUV', or 'BOTH')

        Returns:
            Enhanced image (uint8)
        """
        if mode == 'LAB':
            return self._enhance_lab_color(image)
        elif mode == 'YUV':
            return self._enhance_yuv_color(image)
        elif mode == 'BOTH':
            # Apply both and blend
            lab_enhanced = self._enhance_lab_color(image)
            yuv_enhanced = self._enhance_yuv_color(image)
            # Weighted blend (60% LAB, 40% YUV)
            blended = cv2.addWeighted(lab_enhanced, 0.6, yuv_enhanced, 0.4, 0)
            return blended
        else:
            # Default to LAB
            return self._enhance_lab_color(image)

    def _adjust_saturation(self, image: np.ndarray, scale: float) -> np.ndarray:
        """
        Adjust color saturation in HSV color space.

        Args:
            image: Input image (BGR, uint8)
            scale: Saturation multiplier (0-2)
                   < 1.0 = reduce saturation (more grayscale)
                   = 1.0 = no change
                   > 1.0 = increase saturation (more vibrant)

        Returns:
            Image with adjusted saturation (uint8)
        """
        # Convert BGR to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)

        # Adjust saturation (S channel)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * scale, 0, 255)

        # Convert back to BGR
        hsv = hsv.astype(np.uint8)
        adjusted = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        return adjusted

    # ==================== LOW-LIGHT IMAGE ENHANCEMENT ====================
    # Using DUAL/LIME Method (Retinex-based illumination correction)

    def enhance_lowlight_image(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance a low-light image using DUAL or LIME method with adaptive parameters.

        This method implements the Retinex-based approach with:
        1. Automatic brightness level detection
        2. Adaptive parameter selection based on image darkness
        3. Illumination map estimation and refinement
        4. Multi-exposure fusion (DUAL) or simple correction (LIME)

        Args:
            image: Input low-light image (BGR format)

        Returns:
            Enhanced bright image
        """
        # Use DUAL/LIME enhancement
        enhanced = self._enhance_image_exposure(
            image,
            gamma=self.lowlight_params['gamma'],
            lambda_=self.lowlight_params['lambda_'],
            dual=self.lowlight_params['dual'],
            sigma=self.lowlight_params['sigma'],
            bc=self.lowlight_params['bc'],
            bs=self.lowlight_params['bs'],
            be=self.lowlight_params['be'],
            eps=self.lowlight_params['eps']
        )

        # Apply uniform post-processing brightness boost
        enhanced = self._apply_post_gamma(enhanced, self.lowlight_params['post_gamma'])

        # Apply adaptive noise reduction if enabled
        if self.lowlight_params['denoise']:
            enhanced = self._adaptive_noise_reduction(
                enhanced,
                median_kernel=self.lowlight_params['median_kernel'],
                bilateral_d=self.lowlight_params['bilateral_d'],
                bilateral_sigma=self.lowlight_params['bilateral_sigma']
            )

        return enhanced

    def _apply_post_gamma(self, image: np.ndarray, gamma: float) -> np.ndarray:
        """
        Apply gamma correction as post-processing.

        Args:
            image: Input image (uint8)
            gamma: Gamma value (>1 brightens, <1 darkens)

        Returns:
            Gamma-corrected image
        """
        # Normalize to [0, 1]
        img_normalized = image.astype(np.float32) / 255.0

        # Apply gamma correction
        corrected = np.power(img_normalized, 1.0 / gamma)

        # Convert back to uint8
        return np.clip(corrected * 255, 0, 255).astype(np.uint8)

    def _estimate_noise_level(self, image: np.ndarray) -> float:
        """
        Estimate noise level in the image using Laplacian variance method.

        Args:
            image: Input image (uint8)

        Returns:
            Noise estimate (higher = more noise)
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Compute Laplacian variance (edge detection)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()

        return variance

    def _adaptive_noise_reduction(self, image: np.ndarray, median_kernel: int = 5,
                                  bilateral_d: int = 9, bilateral_sigma: int = 75) -> np.ndarray:
        """
        Apply adaptive noise reduction with median and bilateral filtering.

        This method:
        1. Uses median filter to remove salt & pepper noise
        2. Applies bilateral filter to reduce noise while preserving edges
        3. Blends with original to maintain sharpness

        Args:
            image: Input image (uint8)
            median_kernel: Kernel size for median filter (3, 5, or 7)
            bilateral_d: Bilateral filter diameter
            bilateral_sigma: Bilateral filter sigma for color and space

        Returns:
            Denoised image
        """
        # Step 1: Median filter for salt & pepper noise
        # Higher kernel = more noise removal but potential blur
        median_filtered = cv2.medianBlur(image, median_kernel)

        # Step 2: Bilateral filter for edge-preserving smoothing
        # This maintains sharp edges while reducing noise in smooth areas
        bilateral_filtered = cv2.bilateralFilter(
            median_filtered,
            d=bilateral_d,
            sigmaColor=bilateral_sigma,
            sigmaSpace=bilateral_sigma
        )

        # Step 3: Adaptive blending based on noise level
        noise_level = self._estimate_noise_level(image)

        # If noise is high, use more filtering; if low, preserve original
        if noise_level < 100:  # Low noise
            alpha = 0.3  # 30% filtered, 70% original
        elif noise_level < 500:  # Medium noise
            alpha = 0.6  # 60% filtered, 40% original
        else:  # High noise
            alpha = 0.85  # 85% filtered, 15% original

        # Blend filtered and original for sharpness preservation
        denoised = cv2.addWeighted(bilateral_filtered, alpha, image, 1 - alpha, 0)

        return denoised.astype(np.uint8)

    def _create_spacial_affinity_kernel(self, spatial_sigma: float, size: int = 15) -> np.ndarray:
        """
        Create a kernel (size * size matrix) for spatial affinity based Gaussian weights.

        Args:
            spatial_sigma: Spatial standard deviation
            size: Size of the kernel

        Returns:
            Kernel matrix
        """
        kernel = np.zeros((size, size))
        for i in range(size):
            for j in range(size):
                kernel[i, j] = np.exp(-0.5 * (distance.euclidean((i, j), (size // 2, size // 2)) ** 2) / (spatial_sigma ** 2))

        return kernel

    def _compute_smoothness_weights(self, L: np.ndarray, x: int, kernel: np.ndarray, eps: float = 1e-3) -> np.ndarray:
        """
        Compute smoothness weights for illumination map refinement.

        Args:
            L: Initial illumination map
            x: Direction (1 for horizontal, 0 for vertical)
            kernel: Spatial affinity matrix
            eps: Small constant for stability

        Returns:
            Smoothness weights
        """
        Lp = cv2.Sobel(L, cv2.CV_64F, int(x == 1), int(x == 0), ksize=1)
        T = convolve(np.ones_like(L), kernel, mode='constant')
        T = T / (np.abs(convolve(Lp, kernel, mode='constant')) + eps)
        return T / (np.abs(Lp) + eps)

    def _refine_illumination_map_linear(self, L: np.ndarray, gamma: float,
                                        lambda_: float, kernel: np.ndarray,
                                        eps: float = 1e-3) -> np.ndarray:
        """
        Refine illumination map using optimization (LIME method).

        Args:
            L: Initial illumination map
            gamma: Gamma correction factor
            lambda_: Balance coefficient
            kernel: Spatial affinity matrix
            eps: Small constant for stability

        Returns:
            Refined illumination map
        """
        # Compute smoothness weights
        wx = self._compute_smoothness_weights(L, x=1, kernel=kernel, eps=eps)
        wy = self._compute_smoothness_weights(L, x=0, kernel=kernel, eps=eps)

        n, m = L.shape
        L_1d = L.copy().flatten()

        # Compute five-point spatially inhomogeneous Laplacian matrix
        row, column, data = [], [], []
        for p in range(n * m):
            diag = 0
            for q, (k, l, x) in get_sparse_neighbor(p, n, m).items():
                weight = wx[k, l] if x else wy[k, l]
                row.append(p)
                column.append(q)
                data.append(-weight)
                diag += weight
            row.append(p)
            column.append(p)
            data.append(diag)
        F = csr_matrix((data, (row, column)), shape=(n * m, n * m))

        # Solve linear system
        Id = diags([np.ones(n * m)], [0])
        A = Id + lambda_ * F
        L_refined = spsolve(csr_matrix(A), L_1d, permc_spec=None, use_umfpack=True).reshape((n, m))

        # Gamma correction
        L_refined = np.clip(L_refined, eps, 1) ** gamma

        return L_refined

    def _correct_underexposure(self, im: np.ndarray, gamma: float,
                               lambda_: float, kernel: np.ndarray,
                               eps: float = 1e-3) -> np.ndarray:
        """
        Correct underexposure using Retinex-based algorithm.

        Args:
            im: Input image (normalized)
            gamma: Gamma correction factor
            lambda_: Balance coefficient
            kernel: Spatial affinity matrix
            eps: Small constant for stability

        Returns:
            Corrected image
        """
        # Initial illumination map estimation
        L = np.max(im, axis=-1)

        # Refine illumination map
        L_refined = self._refine_illumination_map_linear(L, gamma, lambda_, kernel, eps)

        # Correct underexposure
        L_refined_3d = np.repeat(L_refined[..., None], 3, axis=-1)
        im_corrected = im / L_refined_3d

        return im_corrected

    def _fuse_multi_exposure_images(self, im: np.ndarray, under_ex: np.ndarray,
                                    over_ex: np.ndarray, bc: float = 1,
                                    bs: float = 1, be: float = 1.5) -> np.ndarray:
        """
        Fuse multi-exposure images using Mertens method (DUAL).

        Args:
            im: Original image
            under_ex: Under-exposure corrected image
            over_ex: Over-exposure corrected image
            bc: Contrast measure weight
            bs: Saturation measure weight
            be: Well-exposedness measure weight

        Returns:
            Fused image
        """
        merge_mertens = cv2.createMergeMertens(bc, bs, be)
        images = [np.clip(x * 255, 0, 255).astype("uint8") for x in [im, under_ex, over_ex]]
        fused_images = merge_mertens.process(images)
        return fused_images

    def _enhance_image_exposure(self, im: np.ndarray, gamma: float, lambda_: float,
                                dual: bool = True, sigma: int = 3, bc: float = 1,
                                bs: float = 1, be: float = 1.5, eps: float = 1e-3) -> np.ndarray:
        """
        Main enhancement function using DUAL or LIME method.

        Args:
            im: Input image
            gamma: Gamma correction factor
            lambda_: Balance coefficient
            dual: Use DUAL (True) or LIME (False)
            sigma: Spatial standard deviation
            bc: Contrast measure weight
            bs: Saturation measure weight
            be: Well-exposedness measure weight
            eps: Small constant for stability

        Returns:
            Enhanced image
        """
        # Create spatial affinity kernel
        kernel = self._create_spacial_affinity_kernel(sigma)

        # Normalize image
        im_normalized = im.astype(float) / 255.0

        # Correct underexposure
        under_corrected = self._correct_underexposure(im_normalized, gamma, lambda_, kernel, eps)

        if dual:
            # DUAL method: also correct overexposure and fuse
            inv_im_normalized = 1 - im_normalized
            over_corrected = 1 - self._correct_underexposure(inv_im_normalized, gamma, lambda_, kernel, eps)
            # Fuse images
            im_corrected = self._fuse_multi_exposure_images(im_normalized, under_corrected, over_corrected, bc, bs, be)
        else:
            # LIME method: only underexposure correction
            im_corrected = under_corrected

        # Convert back to uint8
        return np.clip(im_corrected * 255, 0, 255).astype("uint8")
    
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
            print(f"[OK] Processed: {os.path.basename(input_path)}")
            return True

        except Exception as e:
            print(f"[ERROR] Error processing {input_path}: {str(e)}")
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
    base_input_path = r"C:\Users\muham\Documents\VIP Indivudal Assignment_lama\Datset"
    base_output_path = r"C:\Users\muham\Documents\VIP Indivudal Assignment_lama\output"
    
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
    
    print("\n[OK] All processing complete!")
    print(f"Output saved to: {base_output_path}")


if __name__ == "__main__":
    main()

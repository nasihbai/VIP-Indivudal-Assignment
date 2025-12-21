# CDS6334 Visual Information Processing - Image Enhancement Assignment

## Student Information
- **Name**: [Your Name]
- **ID**: [Your Student ID]
- **Course**: CDS6334 Visual Information Processing
- **Trimester**: 2530
- **Assignment**: Individual Assignment (20%)

---

## Overview

This project implements traditional (non-deep learning) image processing techniques to enhance:
1. **Hazy images** - Using Dark Channel Prior-based dehazing
2. **Low-light images** - Using CLAHE, gamma correction, and bilateral filtering

**No deep learning, pretrained models, or AI-based tools are used.**

---

## Project Structure

```
project/
├── enhancement.py          # Main enhancement pipeline
├── evaluation.py           # Evaluation metrics (PSNR, SSIM, Sharpness, Contrast)
├── utils.py               # Utility functions for visualization
├── README.md              # This file
├── requirements.txt       # Python dependencies
│
├── dataset/               # Your dataset folder (not included)
│   ├── 01. Hazy - Raw/
│   ├── 01. Hazy - Enhanced (GT)/
│   ├── 02. Low Light - Raw/
│   └── 02. Low Light - Enhanced (GT)/
│
└── output/                # Generated output (created by scripts)
    ├── hazy-student-enhanced/
    ├── lowlight-student-enhanced/
    ├── visualizations/
    └── evaluation_results.xlsx
```

---

## Installation

### 1. Install Required Packages

```bash
pip install opencv-python numpy pandas scikit-image matplotlib openpyxl
```

Or use the requirements file:

```bash
pip install -r requirements.txt
```

### 2. Prepare Dataset

Place your dataset in a folder named `dataset/` with the following structure:

```
dataset/
├── 01. Hazy - Raw/
├── 01. Hazy - Enhanced (GT)/
├── 02. Low Light - Raw/
└── 02. Low Light - Enhanced (GT)/
```

---

## Usage

### Step 1: Run Image Enhancement

```bash
python enhancement.py
```

This will:
- Process all hazy images from `dataset/01. Hazy - Raw/`
- Process all low-light images from `dataset/02. Low Light - Raw/`
- Save enhanced images to `output/hazy-student-enhanced/` and `output/lowlight-student-enhanced/`

### Step 2: Run Evaluation

```bash
python evaluation.py
```

This will:
- Calculate PSNR and SSIM for all enhanced images
- Calculate Sharpness (Variance of Laplacian) and Contrast (Std Dev)
- Generate before/after histogram comparisons
- Save results to `output/evaluation_results.xlsx`
- Print summary statistics to console

### Step 3: Generate Visualizations (Optional)

```python
from utils import batch_create_reports

# Create detailed reports for sample images
batch_create_reports(
    raw_folder="dataset/01. Hazy - Raw",
    gt_folder="dataset/01. Hazy - Enhanced (GT)",
    enhanced_folder="output/hazy-student-enhanced",
    report_folder="output/reports/hazy",
    num_samples=10
)
```

---

## Methodology

### Hazy Image Enhancement

The hazy image enhancement pipeline uses the **Dark Channel Prior** algorithm:

1. **Atmospheric Light Estimation**
   - Find the haziest region using dark channel
   - Extract atmospheric light value

2. **Dark Channel Computation**
   - Calculate minimum intensity across color channels
   - Apply morphological erosion

3. **Transmission Estimation**
   - Estimate light transmission through haze
   - Apply guided filtering for refinement

4. **Scene Radiance Recovery**
   - Recover clear image using transmission map
   - Formula: `J(x) = (I(x) - A) / max(t(x), t0) + A`

5. **Post-Processing**
   - CLAHE for contrast enhancement
   - Unsharp masking for sharpening

**Key Parameters:**
- `omega = 0.95` - Dehazing strength
- `radius = 15` - Dark channel radius
- `t0 = 0.1` - Minimum transmission threshold
- `guided_r = 60` - Guided filter radius

### Low-Light Image Enhancement

The low-light enhancement pipeline combines multiple techniques:

1. **CLAHE (Contrast Limited Adaptive Histogram Equalization)**
   - Applied to L channel in LAB color space
   - Clip limit: 2.5
   - Grid size: 8×8

2. **Gamma Correction**
   - Brightens dark regions
   - Gamma value: 1.5

3. **Bilateral Filtering**
   - Reduces noise while preserving edges
   - Parameters: d=9, sigmaColor=75, sigmaSpace=75

4. **Contrast Enhancement**
   - Linear transformation: `α * I + β`
   - α = 1.3, β = 10

5. **Color Correction**
   - Gray World assumption for white balance

6. **Sharpening**
   - Unsharp masking for detail enhancement

---

## Evaluation Metrics

### Full-Reference Metrics (Compare with Ground Truth)

1. **PSNR (Peak Signal-to-Noise Ratio)**
   - Measures pixel-level similarity
   - Higher is better (typically 20-40 dB)

2. **SSIM (Structural Similarity Index)**
   - Measures structural similarity
   - Range: 0-1 (1 is perfect match)

### No-Reference Metrics (Computed Separately)

3. **Sharpness (Variance of Laplacian)**
   - Measures image sharpness
   - Higher values = sharper image

4. **Contrast (Standard Deviation of Intensity)**
   - Measures overall contrast
   - Higher values = better contrast

---

## Output Files

After running the scripts, you will have:

1. **Enhanced Images**
   - `output/hazy-student-enhanced/` - Your enhanced hazy images
   - `output/lowlight-student-enhanced/` - Your enhanced low-light images

2. **Evaluation Results**
   - `output/evaluation_results.xlsx` - Excel file with:
     - Detailed metrics for each image
     - Summary statistics for each category

3. **Visualizations**
   - `output/visualizations/hazy/` - Before/after histograms for hazy images
   - `output/visualizations/lowlight/` - Before/after histograms for low-light images

---

## Customization

### Adjusting Enhancement Parameters

#### For Hazy Images (in `enhancement.py`):

```python
self.hazy_params = {
    'omega': 0.95,        # Increase for stronger dehazing (0.8-1.0)
    'radius': 15,         # Dark channel patch size (7-21)
    't0': 0.1,           # Minimum transmission (0.05-0.2)
    'guided_r': 60,      # Guided filter radius (30-100)
    'guided_eps': 0.001  # Guided filter regularization
}
```

#### For Low-Light Images (in `enhancement.py`):

```python
self.lowlight_params = {
    'gamma': 1.5,         # Brightness adjustment (1.2-2.0)
    'alpha': 1.3,         # Contrast multiplier (1.1-1.5)
    'beta': 10,           # Brightness offset (0-20)
    'clahe_clip': 2.5,    # CLAHE clip limit (2.0-4.0)
    'clahe_grid': (8, 8)  # CLAHE grid size
}
```

### Changing Dataset Paths

Edit the paths in `main()` function of each script:

```python
base_input_path = "your_dataset_folder"
base_output_path = "your_output_folder"
```

---

## Troubleshooting

### Issue: "Could not read image"
- Check if dataset paths are correct
- Ensure images are in supported formats (jpg, png, bmp, tiff)

### Issue: Images look too bright/dark
- Adjust gamma value in `lowlight_params`
- Modify alpha/beta for contrast adjustment

### Issue: Hazy images still look hazy
- Increase `omega` value (closer to 1.0)
- Adjust `guided_r` for better transmission refinement

### Issue: Low PSNR/SSIM scores
- This is normal if your approach differs from ground truth
- Focus on visual quality improvement
- Explain your design choices in the report

---

## Report Guidelines

Your report should include:

1. **Abstract & Introduction**
   - Brief overview of the task
   - Dataset description

2. **Methodology**
   - Detailed explanation of chosen techniques
   - Parameter justification
   - Pipeline diagram (optional but recommended)

3. **Results & Analysis**
   - Before/after images (at least 3-5 examples per category)
   - Before/after histograms
   - Tables with PSNR, SSIM, Sharpness, Contrast
   - Average metrics for both categories

4. **Discussion**
   - Strengths of your approach
   - Limitations and challenges
   - Comparison with ground truth

5. **Suggestions for Improvement**
   - What could be done better
   - Alternative techniques to try

6. **Conclusion**
   - Summary of achievements
   - Key takeaways

---

## Tips for Better Results

1. **Visual Inspection First**
   - Look at your enhanced images
   - Adjust parameters based on what you see
   - Don't just optimize for metrics

2. **Test on Sample Images**
   - Start with 2-3 images
   - Fine-tune parameters
   - Then process all images

3. **Compare with Ground Truth**
   - Analyze what makes GT images good
   - Try to replicate those characteristics
   - Document your observations

4. **Iterate and Experiment**
   - Try different parameter combinations
   - Keep notes on what works
   - Save your best configurations

---

## References

### Key Papers and Resources

1. **Dark Channel Prior**
   - He, K., Sun, J., & Tang, X. (2010). "Single image haze removal using dark channel prior." IEEE TPAMI.

2. **CLAHE**
   - Zuiderveld, K. (1994). "Contrast Limited Adaptive Histogram Equalization."

3. **Guided Filter**
   - He, K., Sun, J., & Tang, X. (2013). "Guided image filtering." IEEE TPAMI.

4. **Image Quality Metrics**
   - Wang, Z., et al. (2004). "Image quality assessment: from error visibility to structural similarity." IEEE TIP.

---

## License

This code is submitted as part of academic coursework for CDS6334. 
Please adhere to your institution's academic integrity policies.

---

## Contact

For questions about this implementation:
- **Student**: [Your Name]
- **Email**: [Your Email]
- **Submission Date**: 26th December 2025

---

**Good luck with your assignment! 🚀**

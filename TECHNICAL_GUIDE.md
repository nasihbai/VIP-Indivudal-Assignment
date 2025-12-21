# Technical Guide: Traditional Image Enhancement Techniques

## Table of Contents
1. [Hazy Image Enhancement](#hazy-image-enhancement)
2. [Low-Light Image Enhancement](#low-light-image-enhancement)
3. [Evaluation Metrics](#evaluation-metrics)
4. [Parameter Tuning Guide](#parameter-tuning-guide)

---

## Hazy Image Enhancement

### Overview
Hazy images suffer from reduced visibility due to atmospheric scattering. Our approach uses the **Dark Channel Prior** method, which is based on the observation that in most outdoor haze-free images, at least one color channel has very low intensity in most local regions.

### Key Techniques

#### 1. Dark Channel Prior
**Concept**: The dark channel of an image is the minimum intensity value across color channels in local patches.

**Formula**:
```
J_dark(x) = min_{y∈Ω(x)} (min_{c∈{r,g,b}} J^c(y))
```
where Ω(x) is a local patch centered at x.

**Implementation**:
```python
min_channel = np.min(image, axis=2)  # Min across RGB
dark_channel = cv2.erode(min_channel, kernel)  # Local minimum
```

**Why it works**: In haze-free outdoor images, the dark channel is close to zero. In hazy images, it's influenced by airlight.

---

#### 2. Atmospheric Light Estimation
**Purpose**: Estimate the global atmospheric light (the color of the sky/haze).

**Method**:
1. Find brightest pixels in the dark channel (typically sky/haze)
2. Among these, select pixel with highest intensity in original image
3. Use this as atmospheric light value

**Code**:
```python
# Get top 0.1% brightest pixels in dark channel
num_pixels = int(dark_channel.size * 0.001)
indices = np.argpartition(flat_dark, -num_pixels)[-num_pixels:]

# Find max intensity among these
for i in indices:
    intensity = np.sum(image[i])
    if intensity > max_intensity:
        atmospheric_light = image[i]
```

---

#### 3. Transmission Estimation
**Purpose**: Estimate how much light reaches the camera through the haze.

**Haze Model**:
```
I(x) = J(x)t(x) + A(1-t(x))

where:
I(x) = observed hazy image
J(x) = scene radiance (what we want)
t(x) = transmission map
A = atmospheric light
```

**Transmission Formula**:
```
t(x) = 1 - ω * dark_channel(I/A)
```

**Parameters**:
- ω (omega): Controls dehazing strength (typically 0.9-0.95)
  - Higher ω = stronger dehazing
  - Lower ω = more conservative dehazing

---

#### 4. Guided Filter Refinement
**Purpose**: Refine transmission map to avoid halo artifacts.

**Why needed**: Initial transmission map is blocky due to patch-based processing.

**Guided Filter Properties**:
- Edge-preserving smoothing
- Linear time complexity
- Uses image structure as guidance

**Parameters**:
- Radius (r): Larger = smoother (typical: 30-100)
- Regularization (ε): Controls edge preservation (typical: 0.001-0.01)

**Formula**:
```
q_i = a_k I_i + b_k    for all i in window ω_k

where:
a_k = (cov(I,p) / (var(I) + ε))
b_k = mean(p) - a_k * mean(I)
```

---

#### 5. Scene Radiance Recovery
**Purpose**: Recover the haze-free image.

**Formula**:
```
J(x) = (I(x) - A) / max(t(x), t0) + A

where:
t0 = minimum transmission threshold (prevents division by zero)
```

**Why max(t, t0)**:
- Prevents over-enhancement in very dense haze
- Maintains some haze for distant objects (looks natural)
- Typical t0: 0.05-0.2

---

#### 6. Post-Processing

**a) CLAHE (Contrast Limited Adaptive Histogram Equalization)**
- Enhances local contrast
- Prevents over-amplification of noise
- Applied to L channel in LAB color space

**b) Unsharp Masking**
- Enhances edges and details
- Formula: `sharpened = original + α(original - blurred)`
- Typical α: 0.3-1.0

---

## Low-Light Image Enhancement

### Overview
Low-light images suffer from poor visibility, low contrast, and high noise. Our pipeline combines multiple techniques to address these issues while preserving natural appearance.

### Pipeline Stages

#### 1. CLAHE (Contrast Limited Adaptive Histogram Equalization)
**Purpose**: Enhance local contrast while preventing noise amplification.

**How it works**:
1. Divide image into tiles (e.g., 8×8 grid)
2. Compute histogram for each tile
3. Clip histogram at limit (prevents over-amplification)
4. Redistribute clipped pixels
5. Apply histogram equalization
6. Interpolate at tile boundaries

**Parameters**:
- `clipLimit`: Controls contrast enhancement (2.0-4.0)
  - Higher = stronger contrast but more noise
  - Lower = gentler enhancement
- `tileGridSize`: Size of tiles (4×4 to 16×16)
  - Smaller = more local adaptation
  - Larger = smoother result

**Why LAB color space**:
- Separates luminance (L) from color (a, b)
- Enhancing L preserves color information
- Prevents color distortion

---

#### 2. Gamma Correction
**Purpose**: Adjust overall brightness non-linearly.

**Formula**:
```
Output = 255 * (Input/255)^(1/γ)

where:
γ > 1: brightens image (expands dark values)
γ < 1: darkens image (compresses dark values)
```

**Typical values**:
- Low-light enhancement: γ = 1.5-2.0
- Monitor calibration: γ = 2.2

**Advantages**:
- Preserves relative contrast
- Brightens shadows more than highlights
- Fast computation (lookup table)

**Implementation**:
```python
inv_gamma = 1.0 / gamma
table = np.array([((i / 255.0) ** inv_gamma) * 255 
                 for i in range(256)]).astype(np.uint8)
corrected = cv2.LUT(image, table)
```

---

#### 3. Bilateral Filtering
**Purpose**: Reduce noise while preserving edges.

**How it works**:
- Combines spatial proximity and intensity similarity
- Nearby pixels with similar intensity = high weight
- Nearby pixels with different intensity = low weight

**Formula**:
```
BF[I]_p = (1/W_p) * Σ_q G_σs(||p-q||) * G_σr(|I_p - I_q|) * I_q

where:
G_σs = spatial Gaussian (distance weight)
G_σr = range Gaussian (intensity weight)
W_p = normalization factor
```

**Parameters**:
- `d`: Diameter of pixel neighborhood
  - Typical: 5-15
  - Larger = more smoothing but slower
- `sigmaColor`: Filter sigma in color space
  - Typical: 50-100
  - Larger = more colors mixed
- `sigmaSpace`: Filter sigma in coordinate space
  - Typical: 50-100
  - Larger = more spatial smoothing

---

#### 4. Contrast Enhancement
**Purpose**: Stretch intensity range for better visibility.

**Linear Transformation**:
```
Output = α * Input + β

where:
α = contrast multiplier (typically 1.1-1.5)
β = brightness offset (typically 0-20)
```

**Effect**:
- α > 1: Increases contrast
- α < 1: Decreases contrast
- β > 0: Increases brightness
- β < 0: Decreases brightness

---

#### 5. Color Correction (Gray World Assumption)
**Purpose**: Remove color cast caused by lighting conditions.

**Assumption**: Average color in natural scenes is gray.

**Algorithm**:
1. Calculate mean of each color channel
2. Calculate gray reference = (R_avg + G_avg + B_avg) / 3
3. Scale each channel:
   ```
   R' = R * (gray_ref / R_avg)
   G' = G * (gray_ref / G_avg)
   B' = B * (gray_ref / B_avg)
   ```

**When it works best**:
- Images with diverse colors
- Uniform lighting
- No dominant single color

**Limitations**:
- Fails for images with naturally dominant colors
- May overcorrect sunset/sunrise scenes

---

#### 6. Sharpening (Unsharp Masking)
**Purpose**: Enhance edges and fine details.

**Algorithm**:
1. Create blurred version: `Blurred = GaussianBlur(Image)`
2. Find detail: `Detail = Image - Blurred`
3. Add back scaled detail: `Sharpened = Image + strength * Detail`

**Alternative formula**:
```
Sharpened = (1 + strength) * Image - strength * Blurred
```

**Parameters**:
- `strength`: How much to sharpen (0.3-1.5)
  - Too high = halos and noise
  - Too low = no visible effect
- Gaussian kernel size: Controls detail scale
  - Small (3-5): Fine details
  - Large (7-15): Coarse details

---

## Evaluation Metrics

### Full-Reference Metrics

#### PSNR (Peak Signal-to-Noise Ratio)
**Formula**:
```
PSNR = 10 * log10(MAX²/MSE)

where:
MAX = maximum possible pixel value (255 for 8-bit)
MSE = mean squared error
```

**Interpretation**:
- Higher is better
- Typical range: 20-40 dB
- <30 dB: Poor quality
- 30-40 dB: Good quality
- >40 dB: Excellent quality

**Limitations**:
- Doesn't match human perception well
- Sensitive to small shifts
- Same PSNR ≠ same perceived quality

---

#### SSIM (Structural Similarity Index)
**Purpose**: Measure perceived quality based on structural information.

**Components**:
1. Luminance comparison: `l(x,y) = (2μ_x μ_y + C1) / (μ_x² + μ_y² + C1)`
2. Contrast comparison: `c(x,y) = (2σ_x σ_y + C2) / (σ_x² + σ_y² + C2)`
3. Structure comparison: `s(x,y) = (σ_xy + C3) / (σ_x σ_y + C3)`

**Final formula**:
```
SSIM(x,y) = [l(x,y)]^α * [c(x,y)]^β * [s(x,y)]^γ

Typically α = β = γ = 1
```

**Interpretation**:
- Range: -1 to 1 (usually 0 to 1)
- 1 = perfect structural similarity
- >0.9: Excellent
- 0.8-0.9: Good
- 0.7-0.8: Fair
- <0.7: Poor

**Advantages**:
- Better matches human perception
- Considers structure, not just error

---

### No-Reference Metrics

#### Sharpness (Variance of Laplacian)
**Purpose**: Measure edge strength (focus quality).

**Formula**:
```
Laplacian = [0  1  0]
            [1 -4  1]
            [0  1  0]

Sharpness = Var(Image ⊗ Laplacian)
```

**Interpretation**:
- Higher = sharper
- Typical range: 0-500+
- Very low (<10): Blurry
- Medium (10-100): Acceptable
- High (>100): Sharp

---

#### Contrast (Standard Deviation of Intensity)
**Purpose**: Measure overall contrast/dynamic range.

**Formula**:
```
Contrast = σ(Intensity) = sqrt(E[(I - μ)²])
```

**Interpretation**:
- Higher = better contrast
- Range: 0-255
- Low (<30): Flat, low contrast
- Medium (30-60): Normal
- High (>60): High contrast

---

## Parameter Tuning Guide

### For Hazy Images

| Parameter | Default | Increase to... | Decrease to... |
|-----------|---------|----------------|----------------|
| omega | 0.95 | Stronger dehazing | More conservative |
| radius | 15 | Smoother dark channel | More detailed |
| t0 | 0.1 | Preserve more haze | Stronger dehazing |
| guided_r | 60 | Smoother transmission | More detailed edges |
| guided_eps | 0.001 | More smoothing | Sharper edges |

**Common adjustments**:
- Sky too bright → Reduce omega
- Still hazy → Increase omega
- Halo artifacts → Increase guided_r
- Details lost → Decrease radius

---

### For Low-Light Images

| Parameter | Default | Increase to... | Decrease to... |
|-----------|---------|----------------|----------------|
| gamma | 1.5 | Brighter | Darker |
| alpha | 1.3 | More contrast | Less contrast |
| beta | 10 | Brighter | Darker |
| clahe_clip | 2.5 | Stronger contrast | Less noise |
| clahe_grid | (8,8) | Larger patches | More local |

**Common adjustments**:
- Too dark → Increase gamma/beta
- Washed out → Decrease alpha/clahe_clip
- Too noisy → Decrease clahe_clip, increase bilateral sigma
- Unnatural colors → Adjust color correction strength

---

## Best Practices

### General Guidelines
1. **Start with defaults** - Observe results first
2. **Adjust one parameter at a time** - Understand individual effects
3. **Visual inspection trumps metrics** - Images should look good
4. **Test on diverse samples** - Parameters that work for one may not work for all
5. **Document your changes** - Keep notes on what works

### Image-Specific Tips

**For very hazy images**:
- Increase omega to 0.98
- Increase guided_r to reduce artifacts
- May need post-processing contrast boost

**For slightly hazy images**:
- Reduce omega to 0.85-0.90
- Use gentler post-processing
- Focus on color restoration

**For very dark images**:
- Increase gamma to 1.8-2.0
- Use stronger CLAHE (clip=3.0-4.0)
- Consider noise reduction first

**For moderately dark images**:
- Keep gamma around 1.5
- Balance CLAHE and bilateral filtering
- Preserve some natural darkness

---

## Troubleshooting Common Issues

### Hazy Images

**Problem**: Sky is over-enhanced/too bright
- **Solution**: Reduce omega or atmospheric light estimation threshold

**Problem**: Halo artifacts around objects
- **Solution**: Increase guided filter radius

**Problem**: Colors look unnatural
- **Solution**: Adjust color correction or reduce dehazing strength

**Problem**: Details are lost
- **Solution**: Reduce dark channel radius, add sharpening

---

### Low-Light Images

**Problem**: Image is too noisy
- **Solution**: Reduce CLAHE clip limit, increase bilateral filtering

**Problem**: Colors are washed out
- **Solution**: Reduce alpha, adjust CLAHE parameters

**Problem**: Uneven brightness
- **Solution**: Reduce CLAHE tile size, adjust gamma

**Problem**: Loss of shadow detail
- **Solution**: Reduce beta, use gentler gamma

---

## References

1. He, K., Sun, J., & Tang, X. (2010). Single image haze removal using dark channel prior. IEEE TPAMI.
2. He, K., Sun, J., & Tang, X. (2013). Guided image filtering. IEEE TPAMI.
3. Zuiderveld, K. (1994). Contrast Limited Adaptive Histogram Equalization.
4. Tomasi, C., & Manduchi, R. (1998). Bilateral filtering for gray and color images. ICCV.
5. Wang, Z., et al. (2004). Image quality assessment: from error visibility to structural similarity. IEEE TIP.

---

**End of Technical Guide**

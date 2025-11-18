# Complete Restoration Flow Explanation

## Overview
This document explains the exact flow when you provide input for restoration and details all types of restorations in the system.

---

## 🔄 Complete Restoration Pipeline Flow

When you run `run.py` with an input image, the system goes through **4 main stages**:

### **Stage 1: Global Restoration (Overall Quality Improvement)**

**Location**: `Global/test.py`

This is the first and most critical stage that handles overall image quality restoration.

#### **Path A: Images WITHOUT Scratches** (`--Quality_restore`)

1. **Input Processing**:
   - Image is loaded and converted to RGB
   - Image is resized/transformed (based on `--test_mode`):
     - `Full`: Resizes to dimensions divisible by 4 (maintains aspect ratio)
     - `Scale`: Scales to 256px on the smaller dimension
     - `Crop`: Center crops to 256x256
   - Image is normalized to [-1, 1] range
   - Empty mask (zeros) is created (no scratch mask needed)

2. **Model Architecture**:
   - Uses **Triplet Domain Translation Network**:
     - **VAE_A (Encoder)**: Encodes degraded old photo → latent space
     - **Mapping Network**: Translates from "old photo domain" → "clean photo domain" 
     - **VAE_B (Decoder)**: Decodes from latent space → restored clean photo
   - Model: `mapping_quality` checkpoint
   - Uses pretrained VAEs: `VAE_A_quality` and `VAE_B_quality`

3. **Inference Process**:
   ```python
   generated = model.inference(input, mask)
   ```
   - Input image is encoded to latent features
   - Features are mapped from degraded domain to clean domain
   - Mapped features are decoded to generate restored image

4. **Output**:
   - Saved to `output/stage_1_restore_output/restored_image/`
   - Also saves original input and processed input for reference

#### **Path B: Images WITH Scratches** (`--Scratch_and_Quality_restore`)

**Step 1: Scratch Detection** (`Global/detection.py`)

1. **Detection Model**:
   - Uses a **U-Net** architecture (4 depth levels, 6 width factor)
   - Converts image to grayscale
   - Normalizes to [-0.5, 0.5] range
   - Scales image (maintains aspect ratio, dimensions divisible by 16)

2. **Scratch Detection Process**:
   - U-Net predicts scratch probability map
   - Applies sigmoid activation
   - Thresholds at 0.4 to create binary mask
   - Mask is interpolated back to original image size

3. **Output**:
   - Binary mask saved to `masks/mask/`
   - Preprocessed input saved to `masks/input/`

**Step 2: Scratch and Quality Restoration** (`Global/test.py`)

1. **Mask Processing**:
   - Mask is loaded and dilated (if `--HR` flag, dilation=3 iterations)
   - Original image is combined with mask using `irregular_hole_synthesize()`:
     - Scratched regions are filled with white (255) pixels
     - This creates a "hole" for inpainting

2. **Model Architecture**:
   - Uses **Non-Local Feature Mapping Network**:
     - Same VAE encoder/decoder structure
     - **Enhanced Mapping Network** with:
       - Non-local attention blocks (`Setting_42`)
       - Spectral normalization
       - Correlation renormalization
       - Mask-aware processing
   - For HR images: Uses `mapping_Patch_Attention` with multi-scale patch attention
   - Model: `mapping_scratch` checkpoint (or `mapping_Patch_Attention` for HR)
   - Uses pretrained VAEs: `VAE_A_quality` and `VAE_B_scratch`

3. **Inference Process**:
   - Image with white-filled scratches is encoded
   - Mask guides the non-local attention to focus on scratch regions
   - Features are mapped from degraded domain to clean domain
   - Decoder generates restored image with scratches removed

4. **Output**:
   - Restored image saved to `output/stage_1_restore_output/restored_image/`

**Fallback**: If no faces are detected later, Stage 1 output is copied directly to final output.

---

### **Stage 2: Face Detection**

**Location**: `Face_Detection/detect_all_dlib.py` (or `detect_all_dlib_HR.py` for HR)

1. **Input**: Stage 1 restored images

2. **Face Detection Process**:
   - Uses **dlib** face detector (`get_frontal_face_detector()`)
   - Detects all faces in the image
   - For each detected face:
     - Uses 68-point facial landmark predictor
     - Extracts 5 key landmarks:
       - Left eye center (average of landmarks 36, 39)
       - Right eye center (average of landmarks 42, 45)
       - Nose tip (landmark 30)
       - Left mouth corner (landmark 48)
       - Right mouth corner (landmark 54)

3. **Face Alignment**:
   - Computes similarity transformation matrix
   - Aligns face to standard 256x256 template (scale factor 1.3)
   - Warps face region to aligned position
   - Saves aligned face as `{image_name}_{face_id}.png`

4. **Output**:
   - Aligned faces saved to `output/stage_2_detection_output/`
   - If no faces detected, warning is printed and image is skipped

---

### **Stage 3: Face Enhancement**

**Location**: `Face_Enhancement/test_face.py`

1. **Input**: Aligned faces from Stage 2

2. **Model Architecture**:
   - Uses **SPADE (Spatially-Adaptive Normalization) Generator**
   - Progressive generator that refines face regions
   - Can use VAE for random variations (if enabled)
   - Uses semantic parsing maps (if available) or degraded face directly

3. **Enhancement Process**:
   - Face images are loaded and preprocessed:
     - Resized to `load_size` (256 for standard, 512 for HR)
     - Normalized to [-1, 1]
   - Generator processes the face:
     - Starts from low-resolution latent/feature map
     - Progressively upsamples through multiple SPADE blocks
     - Each SPADE block uses spatially-adaptive normalization:
       - Normalizes activations
       - Generates scale (gamma) and bias (beta) parameters from semantic map
       - Applies adaptive normalization: `out = normalized * (1 + gamma) + beta`
   - Final output is generated face at target resolution

4. **Model Checkpoints**:
   - Standard: `Setting_9_epoch_100`
   - HR: `FaceSR_512`

5. **Output**:
   - Enhanced faces saved to `output/stage_3_face_output/each_img/`

---

### **Stage 4: Face Warping and Blending**

**Location**: `Face_Detection/align_warp_back_multiple_dlib.py` (or `align_warp_back_multiple_dlib_HR.py`)

1. **Input**:
   - Original restored image from Stage 1
   - Enhanced faces from Stage 3

2. **Warping Process**:
   - For each face in the original image:
     - Detects face and landmarks again (same as Stage 2)
     - Computes forward transformation (face → aligned)
     - Computes **inverse transformation** (aligned → original position)
     - Loads corresponding enhanced face
     - **Histogram Matching**: Matches color histogram of enhanced face to original aligned face
       - Prevents color mismatch between enhanced face and original image
       - Uses CDF (Cumulative Distribution Function) matching per RGB channel
     - Warps enhanced face back to original position using inverse transformation

3. **Blending Process**:
   - Creates mask from forward transformation
   - Warps mask back to original image coordinates
   - Uses **Gaussian blur blending**:
     - Erodes mask slightly (9x9 kernel, 3 iterations)
     - Applies Gaussian blur (25x25 kernel)
     - Blends enhanced face with original image using blurred mask
     - Formula: `blended = enhanced_face * mask_blur + original * (1 - mask_blur)`

4. **Output**:
   - Final restored image with enhanced faces saved to `output/final_output/`

---

## 🎯 Types of Restoration Explained

### **1. Quality Restoration (No Scratches)**

**Purpose**: Restores overall image quality, color, contrast, and removes general degradation.

**How it works**:
- Uses **Triplet Domain Translation**:
  - **Domain A (Old Photos)**: Encoded to latent space via VAE_A
  - **Mapping Network**: Translates features from old photo domain to clean photo domain
  - **Domain B (Clean Photos)**: Decoded from latent space via VAE_B
- The mapping network learns to translate degraded features to clean features
- Handles:
  - Color fading/yellowing
  - Low contrast
  - Noise
  - General age-related degradation

**Model**: `mapping_quality`
**VAEs**: `VAE_A_quality`, `VAE_B_quality`

---

### **2. Scratch Detection**

**Purpose**: Identifies and locates scratches in old photos.

**How it works**:
- **U-Net Architecture**: Encoder-decoder with skip connections
- Processes grayscale version of image
- Outputs probability map of scratch locations
- Binary threshold at 0.4 creates final mask
- Handles:
  - Linear scratches
  - Irregular scratches
  - Various scratch widths

**Model**: `FT_Epoch_latest.pt` (U-Net)

---

### **3. Scratch and Quality Restoration**

**Purpose**: Removes scratches AND improves overall quality simultaneously.

**How it works**:
- **Two-step process**:
  1. **Scratch Detection**: Identifies scratch locations (creates mask)
  2. **Inpainting + Quality Restoration**: 
     - Fills scratch regions with white (creates "holes")
     - Uses **Non-Local Attention** mechanism:
       - Attention blocks can look at distant regions
       - Helps fill scratches using similar regions from the image
     - Mask guides attention to focus on scratch regions
     - Simultaneously improves overall quality

**Key Features**:
- **Non-Local Blocks**: Allow long-range dependencies
- **Mask-Aware Processing**: Focuses attention on damaged regions
- **Spectral Normalization**: Stabilizes training
- **Correlation Renormalization**: Improves feature relationships

**Models**:
- Standard: `mapping_scratch`
- HR: `mapping_Patch_Attention` (multi-scale patch attention for high-res)

**VAEs**: `VAE_A_quality`, `VAE_B_scratch`

---

### **4. Face Enhancement**

**Purpose**: Specifically enhances face regions with higher quality than global restoration.

**How it works**:
- **SPADE Generator**: Spatially-Adaptive Normalization
- **Progressive Upsampling**: Starts low-res, progressively increases resolution
- **Adaptive Normalization**: 
  - Each spatial location gets custom normalization parameters
  - Parameters generated from semantic parsing map (or degraded face)
  - Allows fine-grained control over face features
- Handles:
  - Face details (eyes, nose, mouth)
  - Skin texture
  - Facial features sharpness
  - Better than global restoration for faces

**Model**: `Setting_9_epoch_100` (standard) or `FaceSR_512` (HR)

---

## 🔬 Technical Deep Dive

### **Domain Translation Architecture**

The core restoration uses a **triplet domain translation** approach:

```
Old Photo → [VAE_A Encoder] → Latent Features (Domain A)
                                    ↓
                            [Mapping Network]
                                    ↓
                            Latent Features (Domain B) → [VAE_B Decoder] → Clean Photo
```

**Why this works**:
- Separates encoding/decoding from domain translation
- Allows learning domain-specific features separately
- Mapping network focuses only on translating between domains

### **Non-Local Attention for Scratches**

For scratch removal, the mapping network uses **Non-Local Attention**:

- **Attention Mechanism**: Each location can attend to all other locations
- **Mask Guidance**: Attention is weighted by scratch mask
- **Long-Range Dependencies**: Can use information from distant similar regions
- **Multi-Scale (HR)**: Uses patch attention at multiple scales for high-resolution images

### **SPADE for Face Enhancement**

**Spatially-Adaptive Normalization**:
- Traditional normalization: `(x - mean) / std`
- SPADE: `normalized * (1 + gamma) + beta`
- `gamma` and `beta` are **learned per spatial location** from semantic map
- Allows different normalization for different face regions (eyes, skin, etc.)

---

## 📊 Complete Flow Diagram

```
Input Image
    ↓
┌─────────────────────────────────────┐
│  Stage 1: Global Restoration        │
│  ┌───────────────────────────────┐  │
│  │ Has Scratches?                │  │
│  └───────────────────────────────┘  │
│           ↓              ↓           │
│      YES │              │ NO         │
│          ↓              ↓            │
│  ┌──────────────┐  ┌──────────────┐ │
│  │ Scratch      │  │ Quality     │ │
│  │ Detection    │  │ Restoration │ │
│  │ (U-Net)      │  │ (Mapping)   │ │
│  └──────────────┘  └──────────────┘ │
│          ↓              ↓            │
│          └──────┬───────┘            │
│                 ↓                    │
│         Scratch + Quality           │
│         Restoration                 │
│         (Non-Local Mapping)         │
└─────────────────────────────────────┘
    ↓
Restored Global Image
    ↓
┌─────────────────────────────────────┐
│  Stage 2: Face Detection            │
│  (dlib + 68 landmarks)              │
│  → Extract & Align Faces            │
└─────────────────────────────────────┘
    ↓
Aligned Face Patches
    ↓
┌─────────────────────────────────────┐
│  Stage 3: Face Enhancement           │
│  (SPADE Generator)                  │
│  → Enhance Each Face                │
└─────────────────────────────────────┘
    ↓
Enhanced Faces
    ↓
┌─────────────────────────────────────┐
│  Stage 4: Warp & Blend               │
│  → Inverse Transform                 │
│  → Histogram Matching                │
│  → Gaussian Blur Blending           │
└─────────────────────────────────────┘
    ↓
Final Restored Image
```

---

## 🎛️ Command Line Options

### **Basic Usage**:
```bash
# No scratches
python run.py --input_folder ./test_images/old --output_folder ./output --GPU 0

# With scratches
python run.py --input_folder ./test_images/old_w_scratch --output_folder ./output --GPU 0 --with_scratch

# High-resolution with scratches
python run.py --input_folder ./test_images/old_w_scratch --output_folder ./output --GPU 0 --with_scratch --HR
```

### **Key Parameters**:
- `--input_folder`: Input images directory
- `--output_folder`: Output directory (use absolute path)
- `--GPU`: GPU IDs (e.g., "0" or "0,1,2")
- `--with_scratch`: Enable scratch detection and removal
- `--HR`: High-resolution mode (uses different models)
- `--checkpoint_name`: Face enhancement checkpoint (default: `Setting_9_epoch_100`)

---

## 📁 Output Structure

```
output/
├── stage_1_restore_output/
│   ├── input_image/          # Preprocessed inputs
│   ├── origin/              # Original images
│   ├── restored_image/      # Stage 1 output (global restoration)
│   └── masks/               # (if --with_scratch)
│       ├── input/          # Preprocessed inputs for scratch removal
│       └── mask/           # Scratch detection masks
├── stage_2_detection_output/  # Detected and aligned faces
├── stage_3_face_output/
│   └── each_img/            # Enhanced faces
└── final_output/             # Final restored images with enhanced faces
```

---

## 🔍 Key Insights

1. **Two-Stage Approach**: Global restoration first, then face-specific enhancement
2. **Domain Translation**: Uses learned mappings between old and clean photo domains
3. **Attention Mechanisms**: Non-local attention for scratch removal
4. **Adaptive Normalization**: SPADE for fine-grained face control
5. **Seamless Blending**: Histogram matching + Gaussian blur for natural face integration

This architecture allows the system to handle both structured degradation (scratches) and unstructured degradation (general quality loss) while providing specialized face enhancement.


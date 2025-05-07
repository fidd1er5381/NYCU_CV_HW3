
# NYCU Computer Vision 2025 Spring - Homework 3

StudentID: 313553023  
Name: 褚敏匡

## Introduction

This repository contains an implementation of an enhanced Mask R-CNN model for cell instance segmentation in medical images. The system can accurately detect and segment four different types of cells in medical images, with specialized architectural improvements and post-processing techniques to achieve superior performance within the 200MB model size constraint.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/fidd1er5381/NYCU_CV_HW3.git
cd NYCU_CV_HW3
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset Preparation

1. Organize your dataset in the following structure:
```
hw3-data-release/
├── train/
│   ├── folder1/
│   │   ├── image.tif
│   │   ├── class1.tif
│   │   ├── class2.tif
│   │   ├── class3.tif
│   │   └── class4.tif
│   ├── folder2/
│   │   └── ...
│   └── ...
├── test_release/
│   ├── test1.tif
│   ├── test2.tif
│   └── ...
└── test_image_name_to_ids.json
```

## Train and Predict

To train the model:
```bash
python train.py
```

To run prediction on test data:
```bash
python predict.py --model best_model.pth --threshold 0.25 --use_tta --enhance_boundaries
```

This will generate:
- `test-results.json`: Cell instance segmentation results in the required format

## Performance Snapshot

### Model Architecture
- **Backbone**: ResNet-50 with FPN
- **Enhanced Heads**:
  - Custom Box Predictor with efficient hidden layers (512 dim)
  - Enhanced Mask Predictor with attention mechanism and grouped convolutions
- **Size-optimized**: <200MB total model size
- **Input Resolution**: 512×800

### Training Configuration
- **Optimizer**: SGD with momentum 0.9
- **Learning Rates**: Layerwise (0.0001-0.001)
- **Loss Weighting**: Classification (1.0), BBox Reg (2.0), Mask (3.0)
- **Scheduler**: Cosine Annealing with T_max=40
- **Batch Size**: 2
- **Epochs**: 60

### Performance Metrics
- mAP@50: 0.29 on validation set

## Outputs

- Best Model: `best_model.pth` (<200MB)
- Prediction Results: `test-results.json`

## Enhanced Post-Processing Pipeline

The system utilizes a comprehensive post-processing pipeline to maximize segmentation quality:

1. **Test-Time Augmentation (TTA)**:
   - Multiple transformations (flips, rotations) combined via soft-NMS
   - Improves robustness to orientation and reflection variations

2. **Adaptive Thresholding**:
   - Dynamic threshold selection based on prediction histograms
   - Class-specific thresholds for balanced precision-recall

3. **Class-Specific Morphological Operations**:
   - Tailored processing for each cell type's unique characteristics
   - Size-dependent morphological operations (erosion, dilation, etc.)

4. **Boundary Enhancement**:
   - Watershed-based edge refinement using original image gradients
   - Improves separation of adjacent cells

5. **Connected Component Analysis**:
   - Intelligent handling of mask fragments
   - Adaptive area thresholding for small instance preservation

These techniques together significantly improve segmentation quality, especially for challenging cases like small cells, overlapping cells, and cells with irregular shapes.

## Model Size Optimization

To meet the 200MB constraint, the model utilizes several parameter-efficient techniques:
- Grouped convolutions in the mask head
- Bottleneck attention modules
- Reduced hidden layer dimensions
- Efficient network depth

These optimizations reduced the model size by approximately 20% while maintaining segmentation performance.

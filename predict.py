import os
import glob
import json
import math
import argparse

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import skimage.io as sio
from scipy import ndimage
from skimage import measure, segmentation, morphology, filters
import cv2
from PIL import Image
from tqdm import tqdm

from utils import encode_mask
from train import (
    get_improved_maskrcnn_model,
    set_seed,
    EnhancedMaskPredictor,
    EnhancedBoxPredictor
)

# Set random seed for reproducibility
set_seed()


class TestDataset(Dataset):
    """Dataset for loading and preprocessing test images.

    Args:
        data_dir (str): Directory containing test images
    """

    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.image_files = sorted(glob.glob(os.path.join(data_dir, '*.tif')))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        """Load and preprocess a test image.

        Args:
            idx (int): Index of the image

        Returns:
            dict: Contains processed image tensor and metadata
        """
        image_path = self.image_files[idx]
        image_name = os.path.basename(image_path)

        # Read image
        image = sio.imread(image_path)

        # Ensure image has 3 channels
        if len(image.shape) == 2:
            # Convert single-channel to three-channel
            image = np.stack([image, image, image], axis=2)
        elif len(image.shape) == 3 and image.shape[2] == 4:
            # Convert RGBA to RGB
            image = image[:, :, :3]
        elif len(image.shape) == 3 and image.shape[2] != 3:
            # Handle other non-3-channel cases
            print(f"Warning: Image {image_name} has {image.shape[2]} channels, "
                  f"converting to 3 channels")
            if image.shape[2] < 3:
                # Replicate existing channels if fewer than 3
                channels = [image[:, :, i % image.shape[2]] for i in range(3)]
                image = np.stack(channels, axis=2)
            else:
                # Keep only first 3 channels if more than 3
                image = image[:, :, :3]

        # Convert to tensor and normalize
        image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0

        return {
            'image': image,
            'image_name': image_name,
            'image_path': image_path
        }


def apply_test_time_augmentation(model, image, device):
    """Apply test-time augmentation to improve prediction quality.

    Args:
        model: The model for prediction
        image: Input image tensor
        device: Device to run model on (CPU/GPU)

    Returns:
        dict: Merged prediction results
    """
    original_image = image.clone()

    # Define augmentations
    augmentations = [
        lambda img: img,  # Original
        lambda img: torch.flip(img, dims=[2]),  # Horizontal flip
        lambda img: torch.flip(img, dims=[1]),  # Vertical flip
        lambda img: torch.rot90(img, k=1, dims=[1, 2]),  # 90 degrees
        lambda img: torch.rot90(img, k=2, dims=[1, 2]),  # 180 degrees
        lambda img: torch.rot90(img, k=3, dims=[1, 2]),  # 270 degrees
    ]

    reverse_augmentations = [
        lambda pred: pred,  # Original
        lambda pred: {
            'boxes': flip_boxes_horizontal(pred['boxes'], image.shape[2]),
            'scores': pred['scores'],
            'labels': pred['labels'],
            'masks': torch.flip(pred['masks'], dims=[3])
        },  # Horizontal flip
        lambda pred: {
            'boxes': flip_boxes_vertical(pred['boxes'], image.shape[1]),
            'scores': pred['scores'],
            'labels': pred['labels'],
            'masks': torch.flip(pred['masks'], dims=[2])
        },  # Vertical flip
        lambda pred: {
            'boxes': rotate_boxes_90(pred['boxes'], image.shape[1:3]),
            'scores': pred['scores'],
            'labels': pred['labels'],
            'masks': torch.rot90(pred['masks'], k=-1, dims=[2, 3])
        },  # 90 degrees
        lambda pred: {
            'boxes': rotate_boxes_180(pred['boxes'], image.shape[1:3]),
            'scores': pred['scores'],
            'labels': pred['labels'],
            'masks': torch.rot90(pred['masks'], k=-2, dims=[2, 3])
        },  # 180 degrees
        lambda pred: {
            'boxes': rotate_boxes_270(pred['boxes'], image.shape[1:3]),
            'scores': pred['scores'],
            'labels': pred['labels'],
            'masks': torch.rot90(pred['masks'], k=-3, dims=[2, 3])
        },  # 270 degrees
    ]

    # Run model with each augmentation
    all_predictions = []
    with torch.no_grad():
        for aug_func, reverse_func in zip(augmentations, reverse_augmentations):
            # Apply augmentation
            aug_image = aug_func(original_image)

            # Get prediction
            aug_prediction = model([aug_image.to(device)])[0]

            # Reverse augmentation on predictions
            reversed_prediction = reverse_func(aug_prediction)

            all_predictions.append(reversed_prediction)

    # Merge predictions
    merged_prediction = merge_tta_predictions(all_predictions)

    return merged_prediction


def flip_boxes_horizontal(boxes, width):
    """Flip bounding boxes horizontally.

    Args:
        boxes: Bounding boxes tensor [x1, y1, x2, y2]
        width: Image width

    Returns:
        torch.Tensor: Horizontally flipped boxes
    """
    if len(boxes) == 0:
        return boxes
    flipped_boxes = boxes.clone()
    flipped_boxes[:, 0] = width - boxes[:, 2]
    flipped_boxes[:, 2] = width - boxes[:, 0]
    return flipped_boxes


def flip_boxes_vertical(boxes, height):
    """Flip bounding boxes vertically.

    Args:
        boxes: Bounding boxes tensor [x1, y1, x2, y2]
        height: Image height

    Returns:
        torch.Tensor: Vertically flipped boxes
    """
    if len(boxes) == 0:
        return boxes
    flipped_boxes = boxes.clone()
    flipped_boxes[:, 1] = height - boxes[:, 3]
    flipped_boxes[:, 3] = height - boxes[:, 1]
    return flipped_boxes


def rotate_boxes_90(boxes, image_shape):
    """Rotate bounding boxes by 90 degrees.

    Args:
        boxes: Bounding boxes tensor [x1, y1, x2, y2]
        image_shape: Image dimensions (height, width)

    Returns:
        torch.Tensor: Rotated boxes
    """
    if len(boxes) == 0:
        return boxes
    height, width = image_shape
    rotated_boxes = boxes.clone()
    # Convert [x1, y1, x2, y2] to [y1, width-x2, y2, width-x1]
    rotated_boxes[:, 0], rotated_boxes[:, 1] = boxes[:, 1], width - boxes[:, 2]
    rotated_boxes[:, 2], rotated_boxes[:, 3] = boxes[:, 3], width - boxes[:, 0]
    return rotated_boxes


def rotate_boxes_180(boxes, image_shape):
    """Rotate bounding boxes by 180 degrees.

    Args:
        boxes: Bounding boxes tensor [x1, y1, x2, y2]
        image_shape: Image dimensions (height, width)

    Returns:
        torch.Tensor: Rotated boxes
    """
    if len(boxes) == 0:
        return boxes
    height, width = image_shape
    rotated_boxes = boxes.clone()
    rotated_boxes[:, 0] = width - boxes[:, 2]
    rotated_boxes[:, 1] = height - boxes[:, 3]
    rotated_boxes[:, 2] = width - boxes[:, 0]
    rotated_boxes[:, 3] = height - boxes[:, 1]
    return rotated_boxes


def rotate_boxes_270(boxes, image_shape):
    """Rotate bounding boxes by 270 degrees.

    Args:
        boxes: Bounding boxes tensor [x1, y1, x2, y2]
        image_shape: Image dimensions (height, width)

    Returns:
        torch.Tensor: Rotated boxes
    """
    if len(boxes) == 0:
        return boxes
    height, width = image_shape
    rotated_boxes = boxes.clone()
    # Convert [x1, y1, x2, y2] to [height-y2, x1, height-y1, x2]
    rotated_boxes[:, 0], rotated_boxes[:, 1] = height - boxes[:, 3], boxes[:, 0]
    rotated_boxes[:, 2], rotated_boxes[:, 3] = height - boxes[:, 1], boxes[:, 2]
    return rotated_boxes


def merge_tta_predictions(predictions):
    """Merge multiple predictions from test-time augmentation.

    Args:
        predictions: List of prediction dictionaries from different augmentations

    Returns:
        dict: Merged prediction dictionary
    """
    # Concatenate all predictions
    all_boxes = torch.cat([p['boxes'] for p in predictions])
    all_scores = torch.cat([p['scores'] for p in predictions])
    all_labels = torch.cat([p['labels'] for p in predictions])
    all_masks = torch.cat([p['masks'] for p in predictions])

    # Initialize containers for merged results
    merged_boxes = []
    merged_scores = []
    merged_labels = []
    merged_masks = []

    # Group by class
    for class_id in torch.unique(all_labels):
        # Get predictions for this class
        indices = (all_labels == class_id).nonzero().squeeze(1)
        class_boxes = all_boxes[indices]
        class_scores = all_scores[indices]
        class_masks = all_masks[indices]

        # Sort by score
        score_sorted_idx = torch.argsort(class_scores, descending=True)
        class_boxes = class_boxes[score_sorted_idx]
        class_scores = class_scores[score_sorted_idx]
        class_masks = class_masks[score_sorted_idx]

        # Apply soft-NMS
        keep = soft_nms(class_boxes, class_scores)

        if len(keep) > 0:
            merged_boxes.append(class_boxes[keep])
            merged_scores.append(class_scores[keep])
            merged_labels.append(torch.full((len(keep),), class_id, dtype=torch.int64))
            merged_masks.append(class_masks[keep])

    # If no boxes were kept, return empty prediction
    if not merged_boxes:
        return {
            'boxes': torch.zeros((0, 4), dtype=torch.float32),
            'scores': torch.zeros(0, dtype=torch.float32),
            'labels': torch.zeros(0, dtype=torch.int64),
            'masks': torch.zeros(
                (0, 1, all_masks.shape[2], all_masks.shape[3]),
                dtype=torch.float32
            )
        }

    # Combine results
    return {
        'boxes': torch.cat(merged_boxes),
        'scores': torch.cat(merged_scores),
        'labels': torch.cat(merged_labels),
        'masks': torch.cat(merged_masks)
    }


def soft_nms(boxes, scores, sigma=0.5, score_threshold=0.001, method='gaussian'):
    """Soft NMS implementation for filtering overlapping detections.

    Args:
        boxes: Bounding boxes
        scores: Confidence scores
        sigma: Gaussian sigma parameter
        score_threshold: Minimum score to keep
        method: NMS method ('gaussian' or 'linear')

    Returns:
        torch.Tensor: Indices of kept boxes
    """
    if len(boxes) == 0:
        return torch.tensor([], dtype=torch.int64)

    boxes = boxes.detach().cpu().numpy()
    scores = scores.detach().cpu().numpy()

    # Indexes of picked boxes
    picked_indices = []

    # Coordinates of boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # Area of boxes
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    # Order by scores
    order = np.argsort(scores)[::-1]

    # Start soft NMS
    while order.size > 0:
        # Pick the box with highest score
        i = order[0]
        picked_indices.append(i)

        # Compute IoU of the picked box with the rest
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h

        # IoU
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        # Compute new scores based on method
        if method == 'gaussian':
            weights = np.exp(-(ovr * ovr) / sigma)
        else:  # linear
            weights = 1 - ovr

        # Apply weights to scores
        scores[order[1:]] = weights * scores[order[1:]]

        # Filter boxes by the new scores
        inds = np.where(scores[order[1:]] > score_threshold)[0]
        order = order[inds + 1]

    return torch.tensor(picked_indices, dtype=torch.int64)


def enhance_mask_boundaries(binary_mask, original_image=None):
    """Enhance mask boundaries using advanced image processing techniques.

    Args:
        binary_mask: Binary mask to enhance
        original_image: Original image for reference (optional)

    Returns:
        np.ndarray: Enhanced binary mask
    """
    # Convert to uint8 for OpenCV
    mask_uint8 = binary_mask.astype(np.uint8) * 255

    # Use distance transform to find the center of the mask
    dist_transform = cv2.distanceTransform(mask_uint8, cv2.DIST_L2, 5)
    normalized_dist = cv2.normalize(dist_transform, None, 0, 1.0, cv2.NORM_MINMAX)

    # Find sure foreground
    _, sure_fg = cv2.threshold(normalized_dist, 0.7, 1, 0)

    # Create gradient in boundary region
    boundary_region = binary_mask.astype(float) - sure_fg

    # Apply boundary refinement using watershed if original image is available
    if original_image is not None and len(original_image.shape) == 3:
        # Convert mask to markers for watershed
        sure_fg_int = sure_fg.astype(np.uint8)
        unknown = cv2.subtract(binary_mask.astype(np.uint8), sure_fg_int)

        # Create markers
        _, markers = cv2.connectedComponents(sure_fg_int)
        markers = markers + 1
        markers[unknown == 1] = 0

        # Apply watershed
        gray_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY).astype(np.uint8)
        markers = cv2.watershed(cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR), markers)
        refined_mask = (markers > 1).astype(np.uint8)

        # Ensure we don't lose the original mask area
        refined_mask = np.logical_or(refined_mask, binary_mask).astype(np.uint8)

        return refined_mask

    # If original image is not available, use morphological operations
    kernel = np.ones((3, 3), np.uint8)
    refined_mask = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)
    refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_OPEN, kernel)

    return (refined_mask > 0).astype(np.uint8)


def adaptive_threshold_mask(mask, base_threshold=0.3, area_based_adjustment=True):
    """Adaptively threshold mask based on its properties.

    Args:
        mask: Probability mask
        base_threshold: Initial threshold value
        area_based_adjustment: Whether to adjust threshold based on mask area

    Returns:
        np.ndarray: Binary mask after thresholding
    """
    # Start with base threshold
    threshold = base_threshold

    # Get mask properties
    if area_based_adjustment and np.max(mask) > 0:
        # Calculate histogram
        hist, bins = np.histogram(mask[mask > 0], bins=10, range=(0, 1))

        # Find dominant probabilities
        if len(hist) > 0:
            peak_idx = np.argmax(hist)
            peak_val = bins[peak_idx]

            # Adjust threshold based on peak value
            if peak_val > 0.7:  # Strong predictions
                threshold = max(0.5, peak_val - 0.2)
            elif peak_val > 0.5:  # Medium predictions
                threshold = max(0.3, peak_val - 0.2)
            else:  # Weak predictions
                threshold = min(0.25, peak_val + 0.1)

    # Apply threshold
    binary_mask = (mask > threshold).astype(np.uint8)
    return binary_mask


def class_specific_processing(binary_mask, label, mask_area):
    """Apply class-specific morphological operations.

    Args:
        binary_mask: Input binary mask
        label: Class label
        mask_area: Area of the mask in pixels

    Returns:
        np.ndarray: Processed binary mask
    """
    # Different processing based on cell type
    if label == 1:  # Class 1 cells
        if mask_area < 100:
            # Small class 1 cells - less aggressive processing
            processed = ndimage.binary_dilation(binary_mask, structure=np.ones((2, 2)))
        else:
            # Larger class 1 cells
            processed = ndimage.binary_opening(binary_mask, structure=np.ones((2, 2)))
            processed = ndimage.binary_closing(processed, structure=np.ones((3, 3)))

    elif label == 2:  # Class 2 cells
        # Class 2 might need more edge preservation
        processed = ndimage.binary_closing(binary_mask, structure=np.ones((3, 3)))
        if mask_area > 200:
            # Remove small holes in larger cells
            processed = ndimage.binary_fill_holes(processed)

    elif label == 3:  # Class 3 cells
        if mask_area < 80:
            # Small class 3 cells might be more sensitive
            processed = ndimage.binary_dilation(binary_mask, structure=np.ones((2, 2)))
        else:
            # Larger class 3 cells
            eroded = ndimage.binary_erosion(binary_mask, structure=np.ones((2, 2)))
            processed = ndimage.binary_dilation(eroded, structure=np.ones((3, 3)))

    elif label == 4:  # Class 4 cells
        # Class 4 might have more irregular shapes
        if mask_area < 150:
            processed = ndimage.binary_closing(binary_mask, structure=np.ones((2, 2)))
        else:
            # More aggressive for larger instances
            processed = ndimage.binary_fill_holes(binary_mask)
            processed = ndimage.binary_opening(processed, structure=np.ones((3, 3)))

    else:  # Default processing
        if mask_area < 50:  # Small instances
            processed = ndimage.binary_dilation(binary_mask, structure=np.ones((2, 2)))
        elif mask_area < 200:  # Medium instances
            eroded = ndimage.binary_erosion(binary_mask, structure=np.ones((2, 2)))
            processed = ndimage.binary_dilation(eroded, structure=np.ones((2, 2)))
        else:  # Large instances
            eroded = ndimage.binary_erosion(binary_mask, structure=np.ones((3, 3)))
            processed = ndimage.binary_dilation(eroded, structure=np.ones((3, 3)))

    return processed


def enhanced_component_analysis(mask, min_size=10):
    """Advanced connected component analysis with small component handling.

    Args:
        mask: Binary mask
        min_size: Minimum size of components to keep

    Returns:
        np.ndarray: Filtered binary mask
    """
    # Label connected components
    labeled_mask = measure.label(mask)
    regions = measure.regionprops(labeled_mask)

    if not regions:
        return mask

    # Sort regions by area
    regions.sort(key=lambda x: x.area, reverse=True)

    # Initialize result mask
    result_mask = np.zeros_like(mask)

    # Process each region based on its properties
    for i, region in enumerate(regions):
        if i == 0:  # Always keep the largest region
            result_mask = np.logical_or(result_mask, labeled_mask == region.label)
        elif region.area < min_size:
            # Skip very small regions
            continue
        elif i < 3 and region.area > regions[0].area * 0.2:
            # Keep up to 3 significant regions (>20% of largest)
            result_mask = np.logical_or(result_mask, labeled_mask == region.label)
        elif regions[0].area / (region.area + 1e-8) < 3:
            # Keep regions that are not much smaller than the largest
            result_mask = np.logical_or(result_mask, labeled_mask == region.label)

    return result_mask.astype(np.uint8)


def predict(model, dataloader, device, id_mapping_path, output_file, args):
    """Run prediction on test images.

    Args:
        model: Trained model
        dataloader: Test data loader
        device: Device to run model on
        id_mapping_path: Path to image ID mapping file
        output_file: Path to save prediction results
        args: Command-line arguments
    """
    model.eval()
    results = []

    # Load image name to ID mapping
    with open(id_mapping_path, 'r') as f:
        id_mapping = json.load(f)

    # Create image name to ID dictionary for easy lookup
    image_to_id = {}
    for item in id_mapping:
        file_name = item['file_name']
        img_id = item['id']
        base_name = os.path.splitext(file_name)[0]
        image_to_id[base_name] = img_id

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting"):
            # Get single image from batch
            image = batch['image'][0].to(device)
            image_name = batch['image_name'][0]
            image_path = batch['image_path'][0]

            # Apply test-time augmentation if enabled
            if args.use_tta:
                prediction = apply_test_time_augmentation(model, image, device)
            else:
                # Model prediction
                predictions = model([image])
                prediction = predictions[0]

            base_name = os.path.splitext(image_name)[0]

            # Check if image is in mapping
            if base_name not in image_to_id:
                print(f"Warning: Image {base_name} not found in ID mapping")
                continue

            img_id = image_to_id[base_name]

            # Get predicted results
            pred_boxes = prediction['boxes'].cpu().numpy()
            pred_masks = prediction['masks'].cpu().numpy()
            pred_scores = prediction['scores'].cpu().numpy()
            pred_labels = prediction['labels'].cpu().numpy()

            # Load original image for boundary enhancement if needed
            original_image = None
            if args.enhance_boundaries:
                original_image = np.array(Image.open(image_path))

            # Adaptive score thresholding
            # Use lower threshold for classes with fewer predictions
            class_counts = {}
            for label in pred_labels:
                if label not in class_counts:
                    class_counts[label] = 0
                class_counts[label] += 1

            # Calculate per-class thresholds
            class_thresholds = {}
            for label, count in class_counts.items():
                if count < 5:  # Very few predictions
                    class_thresholds[label] = max(0.2, args.threshold - 0.2)
                elif count < 10:  # Few predictions
                    class_thresholds[label] = max(0.25, args.threshold - 0.15)
                else:  # Many predictions
                    class_thresholds[label] = args.threshold

            # Apply class and score filtering
            keep_indices = []
            for i, (score, label) in enumerate(zip(pred_scores, pred_labels)):
                if (1 <= label <= 4 and 
                        score > class_thresholds.get(label, args.threshold)):
                    keep_indices.append(i)

            pred_boxes = pred_boxes[keep_indices]
            pred_masks = pred_masks[keep_indices]
            pred_scores = pred_scores[keep_indices]
            pred_labels = pred_labels[keep_indices]

            # Process each detection result
            for box, mask, score, label in zip(
                    pred_boxes, pred_masks, pred_scores, pred_labels):
                # Apply adaptive thresholding
                binary_mask = adaptive_threshold_mask(
                    mask[0], base_threshold=0.3 - 0.05 * (score > 0.7))

                # Calculate mask area and dimensions
                mask_area = np.sum(binary_mask)
                mask_height, mask_width = binary_mask.shape

                if mask_area < 5:  # Skip extremely small masks
                    continue

                # Apply class-specific processing
                processed_mask = class_specific_processing(
                    binary_mask, label, mask_area)

                # Fill small holes
                filled_mask = ndimage.binary_fill_holes(processed_mask)

                # Enhanced connected component analysis
                final_mask = enhanced_component_analysis(
                    filled_mask, min_size=max(5, int(mask_area * 0.05)))

                # Enhance boundaries if enabled
                if args.enhance_boundaries and original_image is not None:
                    # Extract region of interest from original image
                    x1, y1, x2, y2 = map(int, box)
                    x1 = max(0, x1 - 5)
                    y1 = max(0, y1 - 5)
                    x2 = min(original_image.shape[1], x2 + 5)
                    y2 = min(original_image.shape[0], y2 + 5)

                    # Check if we have a valid ROI
                    if x2 > x1 and y2 > y1:
                        roi = original_image[y1:y2, x1:x2]
                        mask_roi = final_mask[y1:y2, x1:x2]

                        if roi.size > 0 and mask_roi.size > 0:
                            enhanced_roi = enhance_mask_boundaries(mask_roi, roi)
                            final_mask[y1:y2, x1:x2] = enhanced_roi

                # Check if mask contains any foreground pixels
                if not np.any(final_mask):
                    continue

                # Get image dimensions (mask height and width)
                height, width = final_mask.shape

                # Encode mask as RLE format
                rle_mask = encode_mask(final_mask)

                # Create detection result dictionary
                detection = {
                    "image_id": img_id,
                    "bbox": box.tolist(),  # [x1, y1, x2, y2] converted to list
                    "score": float(score),
                    "category_id": int(label),
                    "segmentation": {
                        "size": [height, width],
                        "counts": rle_mask["counts"]
                    }
                }

                # Add to results list
                results.append(detection)

    # Save results to json file
    with open(output_file, 'w') as f:
        json.dump(results, f)

    print(f"Prediction results saved to {output_file}")
    print(f"Generated {len(results)} detection results")

    # Count detections per class
    category_counts = {1: 0, 2: 0, 3: 0, 4: 0}
    for result in results:
        category_id = result["category_id"]
        if 1 <= category_id <= 4:
            category_counts[category_id] += 1

    print(f"Class 1: {category_counts[1]} detections")
    print(f"Class 2: {category_counts[2]} detections")
    print(f"Class 3: {category_counts[3]} detections")
    print(f"Class 4: {category_counts[4]} detections")
    print(f"Total detections: {sum(category_counts.values())}")


def main():
    """Main function for running prediction."""
    parser = argparse.ArgumentParser(
        description='Predict cell instance segmentation masks')
    parser.add_argument(
        '--model', type=str, default='best_model.pth', 
        help='Model path')
    parser.add_argument(
        '--test_data', type=str, default='hw3-data-release/test_release', 
        help='Test data path')
    parser.add_argument(
        '--id_mapping', type=str,
        default='hw3-data-release/test_image_name_to_ids.json',
        help='Test image ID mapping file')
    parser.add_argument(
        '--output', type=str, default='test-results.json',
        help='Output filename')
    parser.add_argument(
        '--threshold', type=float, default=0.25,
        help='Detection threshold')
    parser.add_argument(
        '--use_tta', action='store_true',
        help='Use test-time augmentation')
    parser.add_argument(
        '--enhance_boundaries', action='store_true',
        help='Apply boundary enhancement')
    parser.add_argument(
        '--min_mask_area', type=int, default=5,
        help='Minimum mask area to keep')
    args = parser.parse_args()

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create test data loader
    test_dataset = TestDataset(args.test_data)
    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=4)

    # Load model (background + 4 classes)
    num_classes = 5  # background + 4 classes
    model = get_improved_maskrcnn_model(num_classes)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model = model.to(device)
    model.eval()

    # Predict
    predict(model, test_loader, device, args.id_mapping, args.output, args)

    print(f"Prediction complete! Results saved to {args.output}")


if __name__ == '__main__':
    main()
    
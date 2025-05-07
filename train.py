import os
import glob
import random
import json

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchvision import transforms
import skimage.io as sio
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage import measure
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from PIL import Image


def set_seed(seed=42):
    """Set random seeds for reproducibility.
    
    Args:
        seed (int): Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Set seed for reproducibility
set_seed()


class CellInstanceDataset(Dataset):
    """Dataset for cell instance segmentation.
    
    Args:
        data_dir (str): Directory containing the data folders
        transform (callable, optional): Optional transform to be applied to samples
        augment (bool): Whether to apply data augmentation
    """
    def __init__(self, data_dir, transform=None, augment=True):
        self.data_dir = data_dir
        self.transform = transform
        self.augment = augment
        self.folders = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]
        
        # Initialize transformations
        if self.augment:
            self.color_jitter = transforms.ColorJitter(
                brightness=0.1, 
                contrast=0.1, 
                saturation=0.1, 
                hue=0.05
            )

    def __len__(self):
        return len(self.folders)
    
    def __getitem__(self, idx):
        """Get a sample from the dataset.
        
        Args:
            idx (int): Index of the sample
            
        Returns:
            tuple: (image, target) where target contains instance masks and annotations
        """
        folder_name = self.folders[idx]
        folder_path = os.path.join(self.data_dir, folder_name)
        
        # Read image and masks
        image_path = os.path.join(folder_path, 'image.tif')
        class1_path = os.path.join(folder_path, 'class1.tif')
        class2_path = os.path.join(folder_path, 'class2.tif')
        class3_path = os.path.join(folder_path, 'class3.tif')
        class4_path = os.path.join(folder_path, 'class4.tif')
        
        image = sio.imread(image_path)
        image = self.validate_image(image)
        
        # Prepare masks and labels
        masks = []
        labels = []
        
        # Process each class
        for class_id, path in enumerate([class1_path, class2_path, class3_path, class4_path], 1):
            if os.path.exists(path):
                mask = sio.imread(path)
                
                # Find individual instances (connected regions)
                labeled_mask = measure.label(mask > 0)
                props = measure.regionprops(labeled_mask)
                
                for prop in props:
                    instance_mask = (labeled_mask == prop.label).astype(np.uint8)
                    masks.append(instance_mask)
                    labels.append(class_id)
        
        # Data augmentation
        if self.augment:
            # Random horizontal flip
            if random.random() > 0.5:
                image = np.fliplr(image).copy()
                for i in range(len(masks)):
                    masks[i] = np.fliplr(masks[i]).copy()
            
            # Random vertical flip
            if random.random() > 0.5:
                image = np.flipud(image).copy()
                for i in range(len(masks)):
                    masks[i] = np.flipud(masks[i]).copy()
            
            # Random rotation (90-degree multiples)
            k = random.randint(0, 3)
            if k > 0:
                image = np.rot90(image, k).copy()
                for i in range(len(masks)):
                    masks[i] = np.rot90(masks[i], k).copy()
            
            # Image conversion to PIL for torchvision color augmentation
            if len(image.shape) == 3 and image.shape[2] == 3:
                image_pil = Image.fromarray(image.astype(np.uint8))
                image_pil = self.color_jitter(image_pil)
                image = np.array(image_pil)
        
        # Prepare labels and bounding boxes
        boxes = []
        valid_masks = []
        valid_labels = []
        
        for i, mask in enumerate(masks):
            # Find coordinates of non-zero pixels
            pos = np.where(mask)
            
            # Ensure mask contains enough pixels
            if len(pos[0]) == 0:
                continue  # Skip empty masks
            
            # Calculate bounding box
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            
            # Ensure bounding box has positive width and height
            if xmin == xmax:
                xmax = xmin + 1  # Ensure width of at least 1
            if ymin == ymax:
                ymax = ymin + 1  # Ensure height of at least 1
            
            # Double-check if bounding box is valid
            if xmax > xmin and ymax > ymin:
                boxes.append([xmin, ymin, xmax, ymax])
                valid_masks.append(mask)
                valid_labels.append(labels[i])
        
        # Check if there are valid bounding boxes
        if len(boxes) == 0:
            # Create a dummy bounding box and mask (for extreme cases)
            dummy_box = [0, 0, 10, 10]
            dummy_mask = np.zeros((masks[0].shape[0], masks[0].shape[1]), dtype=np.uint8)
            dummy_mask[0:10, 0:10] = 1
            
            boxes = [dummy_box]
            valid_masks = [dummy_mask]
            valid_labels = [1]  # Use class 1 as default
            print(f"Warning: Image {folder_name} has no valid instances, using dummy data")
        
        # Update masks to filtered masks
        masks = valid_masks
        labels = valid_labels
        
        # Convert to Tensor format
        image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        masks = torch.as_tensor(np.array(masks), dtype=torch.uint8)
        
        # Convert to Tensor format 
        boxes = torch.as_tensor(np.array(boxes), dtype=torch.float32)
        labels = torch.as_tensor(np.array(labels), dtype=torch.int64)
        
        # Build target dictionary
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        
        return image, target

    def validate_image(self, image):
        """Ensure image format is correct and convert to required format.
        
        Args:
            image: Input image
            
        Returns:
            np.ndarray: Properly formatted image
            
        Raises:
            ValueError: If image cannot be processed
        """
        # Check if image is None
        if image is None:
            raise ValueError("Unable to read image")
        
        # Check image dimensions
        if len(image.shape) == 2:
            # Convert single-channel to three-channel
            image = np.stack([image, image, image], axis=2)
        elif len(image.shape) == 3:
            if image.shape[2] == 4:
                # Convert RGBA to RGB
                image = image[:, :, :3]
            elif image.shape[2] != 3:
                raise ValueError(f"Unsupported number of image channels: {image.shape[2]}")
        else:
            raise ValueError(f"Unsupported image dimensions: {len(image.shape)}")
        
        return image


def calculate_ap50(predictions, targets, iou_threshold=0.5):
    """Calculate AP50 for a single image.
    
    Args:
        predictions: Model predictions containing 'boxes', 'scores', 'labels', 'masks'
        targets: Ground truth containing 'boxes', 'labels', 'masks'
        iou_threshold: IoU threshold, default 0.5
    
    Returns:
        float: Average precision (AP) value
    """
    # Get predictions and targets
    pred_boxes = predictions['boxes'].cpu().numpy()
    pred_scores = predictions['scores'].cpu().numpy()
    pred_labels = predictions['labels'].cpu().numpy()
    pred_masks = predictions['masks'].cpu().numpy()
    
    target_boxes = targets['boxes'].cpu().numpy()
    target_labels = targets['labels'].cpu().numpy()
    
    # Initialize AP per class
    ap_per_class = {}
    
    # Process each class
    for class_id in np.unique(np.concatenate([pred_labels, target_labels])):
        if class_id == 0:  # Skip background class
            continue
        
        # Get predictions and targets for current class
        pred_indices = np.where(pred_labels == class_id)[0]
        target_indices = np.where(target_labels == class_id)[0]
        
        if len(pred_indices) == 0 or len(target_indices) == 0:
            ap_per_class[class_id] = 0.0
            continue
        
        # Class predictions and targets
        class_pred_boxes = pred_boxes[pred_indices]
        class_pred_scores = pred_scores[pred_indices]
        class_pred_masks = pred_masks[pred_indices]
        
        class_target_boxes = target_boxes[target_indices]
        
        # Sort predictions by confidence score (descending)
        score_sort = np.argsort(-class_pred_scores)
        class_pred_boxes = class_pred_boxes[score_sort]
        class_pred_scores = class_pred_scores[score_sort]
        class_pred_masks = class_pred_masks[score_sort]
        
        # Calculate IoU matrix
        ious = np.zeros((len(class_pred_boxes), len(class_target_boxes)))
        for i, pred_box in enumerate(class_pred_boxes):
            for j, target_box in enumerate(class_target_boxes):
                # Calculate IoU
                # Get intersection rectangle
                xmin = max(pred_box[0], target_box[0])
                ymin = max(pred_box[1], target_box[1])
                xmax = min(pred_box[2], target_box[2])
                ymax = min(pred_box[3], target_box[3])
                
                # Calculate intersection area
                if xmin < xmax and ymin < ymax:
                    intersection = (xmax - xmin) * (ymax - ymin)
                else:
                    intersection = 0
                
                # Calculate union area
                pred_area = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
                target_area = (target_box[2] - target_box[0]) * (target_box[3] - target_box[1])
                union = pred_area + target_area - intersection
                
                # Calculate IoU
                if union > 0:
                    ious[i, j] = intersection / union
                else:
                    ious[i, j] = 0
        
        # Initialize true positives and false positives
        tp = np.zeros(len(class_pred_boxes))
        fp = np.zeros(len(class_pred_boxes))
        
        # Determine whether each prediction is TP or FP
        target_indices_matched = set()
        
        for i in range(len(class_pred_boxes)):
            # Find target with max IoU
            max_iou_idx = np.argmax(ious[i])
            max_iou = ious[i, max_iou_idx]
            
            # If IoU exceeds threshold and target not previously matched
            if max_iou >= iou_threshold and max_iou_idx not in target_indices_matched:
                tp[i] = 1
                target_indices_matched.add(max_iou_idx)
            else:
                fp[i] = 1
        
        # Calculate cumulative TP and FP
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        # Calculate precision and recall
        precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-10)
        recall = tp_cumsum / len(target_indices)
        
        # Add start and end points
        precision = np.concatenate(([1], precision))
        recall = np.concatenate(([0], recall))
        
        # Calculate AP
        ap = np.trapz(precision, recall)
        ap_per_class[class_id] = ap
    
    # If no AP calculated for any class, return 0
    if not ap_per_class:
        return 0.0
    
    # Return average AP across all classes
    return np.mean(list(ap_per_class.values()))


class EnhancedMaskPredictor(nn.Module):
    """Enhanced mask predictor with attention mechanism and efficient architecture.
    
    Args:
        in_channels (int): Number of input channels
        hidden_layer (int): Number of hidden channels
        num_classes (int): Number of classes to predict
    """
    def __init__(self, in_channels, hidden_layer, num_classes):
        super(EnhancedMaskPredictor, self).__init__()
        # Reduce channel count to lower parameter count
        hidden_layer = 256  # Reduced from 512 to 256
        self.conv5_mask = nn.ConvTranspose2d(in_channels, hidden_layer, 2, 2, 0)
        
        # Simplified attention module
        self.attn_module = nn.Sequential(
            nn.Conv2d(hidden_layer, hidden_layer // 2, 3, padding=1),
            nn.BatchNorm2d(hidden_layer // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_layer // 2, hidden_layer, 1),
            nn.Sigmoid()
        )
        
        # Streamlined convolution layers with grouped convolutions
        self.conv_layers = nn.Sequential(
            nn.Conv2d(hidden_layer, hidden_layer, 3, padding=1, groups=4),
            nn.BatchNorm2d(hidden_layer),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_layer, hidden_layer, 3, padding=1, groups=4),
            nn.BatchNorm2d(hidden_layer),
            nn.ReLU(inplace=True)
        )
        
        # Simplified refinement module
        self.refinement = nn.Sequential(
            nn.Conv2d(hidden_layer, hidden_layer//2, 3, padding=1, groups=2),
            nn.BatchNorm2d(hidden_layer//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_layer//2, hidden_layer, 3, padding=1),
            nn.BatchNorm2d(hidden_layer),
            nn.ReLU(inplace=True)
        )
        
        # Mask prediction layer
        self.mask_fcn_logits = nn.Conv2d(hidden_layer, num_classes, 1)
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Forward pass through the mask predictor.
        
        Args:
            x: Input features
            
        Returns:
            torch.Tensor: Predicted masks
        """
        x = F.relu(self.conv5_mask(x))
        
        # Apply attention mechanism
        attn = self.attn_module(x)
        x = x * attn
        
        # Apply deep convolution
        feat = self.conv_layers(x)
        
        # Residual connection
        x = x + feat
        
        # Refinement stage
        x = self.refinement(x)
        
        # Final prediction
        return self.mask_fcn_logits(x)


class EnhancedBoxPredictor(nn.Module):
    """Enhanced box predictor with efficient architecture.
    
    Args:
        in_features (int): Number of input features
        num_classes (int): Number of classes to predict
    """
    def __init__(self, in_features, num_classes):
        super(EnhancedBoxPredictor, self).__init__()
        
        # Reduced intermediate feature dimensions
        hidden_dim = 512  # Reduced from 1024 to 512
        
        # Classification branch
        self.cls_score = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )
        
        # Bounding box regression branch
        self.bbox_pred = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes * 4)
        )
        
        # Initialize weights
        for module in [self.cls_score, self.bbox_pred]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.normal_(layer.weight, mean=0, std=0.01)
                    nn.init.constant_(layer.bias, 0)
    
    def forward(self, x):
        """Forward pass through the box predictor.
        
        Args:
            x: Input features
            
        Returns:
            tuple: (classification scores, bounding box deltas)
        """
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)
        
        return scores, bbox_deltas


def get_improved_maskrcnn_model(num_classes):
    """Create an improved Mask R-CNN model with enhanced heads.
    
    Args:
        num_classes (int): Number of classes to predict
        
    Returns:
        torch.nn.Module: The improved Mask R-CNN model
    """
    # Use pretrained ResNet-50 FPN backbone with reduced feature size
    model = maskrcnn_resnet50_fpn(
        pretrained=True,
        pretrained_backbone=True,
        min_size=512,  # Reduced input size
        max_size=800,  # Reduced max input size
        rpn_pre_nms_top_n_test=1000,
        rpn_post_nms_top_n_test=1000,
        rpn_score_thresh=0.05,
        box_score_thresh=0.05,
        box_nms_thresh=0.5,
        rpn_pre_nms_top_n_train=2000,
        rpn_post_nms_top_n_train=1000,
    )
    
    # Replace box predictor with enhanced version
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    enhanced_box_predictor = EnhancedBoxPredictor(in_features, num_classes)
    
    # Due to torchvision API limitations, we need to adapt to original model structure
    model.roi_heads.box_predictor.cls_score = enhanced_box_predictor.cls_score
    model.roi_heads.box_predictor.bbox_pred = enhanced_box_predictor.bbox_pred
    
    # Replace mask predictor with enhanced version
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256  # Reduced channel count
    model.roi_heads.mask_predictor = EnhancedMaskPredictor(in_features_mask, hidden_layer, num_classes)
    
    return model


def inspect_instances(data_dir):
    """Inspect dataset instances to identify potential issues.
    
    Args:
        data_dir (str): Data directory path
    """
    print("Starting instance inspection in dataset...")
    
    total_instances = 0
    empty_masks = 0
    small_instances = 0
    
    for folder in tqdm(os.listdir(data_dir)):
        folder_path = os.path.join(data_dir, folder)
        if os.path.isdir(folder_path):
            for class_id, class_name in enumerate(['class1', 'class2', 'class3', 'class4'], 1):
                mask_path = os.path.join(folder_path, f'{class_name}.tif')
                if os.path.exists(mask_path):
                    try:
                        mask = sio.imread(mask_path)
                        
                        # Find individual instances
                        labeled_mask = measure.label(mask > 0)
                        props = measure.regionprops(labeled_mask)
                        
                        total_instances += len(props)
                        
                        for prop in props:
                            # Check instance size
                            area = prop.area
                            if area == 0:
                                empty_masks += 1
                            elif area < 5:  # Instances with less than 5 pixels
                                small_instances += 1
                            
                    except Exception as e:
                        print(f"Error processing {mask_path}: {e}")
    
    print(f"Total instances: {total_instances}")
    print(f"Empty masks: {empty_masks}")
    print(f"Small instances (<5 pixels): {small_instances}")


def train_model(model, train_loader, val_loader, optimizer, scheduler, num_epochs, device):
    """Train the model and evaluate on validation set.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        num_epochs: Number of training epochs
        device: Device to train on
        
    Returns:
        nn.Module: Trained model
    """
    best_map = 0.0  # Track best mAP
    best_model_state = None
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for images, targets in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} - Training'):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            # Forward pass
            loss_dict = model(images, targets)
            
            # Apply different weights to different losses
            weighted_losses = {
                'loss_classifier': loss_dict['loss_classifier'] * 1.0,
                'loss_box_reg': loss_dict['loss_box_reg'] * 2.0,     # Increased box regression weight
                'loss_mask': loss_dict['loss_mask'] * 3.0,           # Significantly increased mask loss weight
                'loss_objectness': loss_dict['loss_objectness'] * 1.0,
                'loss_rpn_box_reg': loss_dict['loss_rpn_box_reg'] * 1.5
            }
            
            # Calculate total loss
            losses = sum(loss for loss in weighted_losses.values())
            
            # Backpropagation and optimization
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            
            train_loss += losses.item()
        
        train_loss = train_loss / len(train_loader)
        
        # Validation phase - evaluate with AP50
        model.eval()
        val_maps = []
        
        with torch.no_grad():
            for images, targets in tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} - Validation'):
                images = list(image.to(device) for image in images)
                targets_cpu = [{k: v.cpu() for k, v in t.items()} for t in targets]
                
                # Model predictions
                predictions = model(images)
                
                # Calculate AP50 for each image
                for pred, target in zip(predictions, targets_cpu):
                    ap50 = calculate_ap50(pred, target)
                    val_maps.append(ap50)
        
        # Calculate validation mAP@50
        val_map = np.mean(val_maps)
        
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Validation mAP@50: {val_map:.4f}')
        
        # Save best model based on mAP
        if val_map > best_map:
            best_map = val_map
            best_model_state = model.state_dict().copy()
            torch.save(best_model_state, 'best_model.pth')
            print(f'Saved new best model with mAP@50: {val_map:.4f}')
        
        # Update learning rate
        scheduler.step()
    
    # Load best model weights
    model.load_state_dict(best_model_state)
    return model


def main():
    """Main function to run training pipeline."""
    # Set data path and device
    data_dir = 'hw3-data-release/train'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Inspect dataset instances
    inspect_instances(data_dir)
    
    # Create dataset and DataLoader
    dataset = CellInstanceDataset(data_dir)
    
    # Split into training and validation sets (90% train, 10% validation)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    def collate_fn(batch):
        return tuple(zip(*batch))
    
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, 
                             num_workers=4, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, 
                           num_workers=4, collate_fn=collate_fn)
    
    # Create improved model
    model = get_improved_maskrcnn_model(num_classes=5)  # background + 4 classes
    model = model.to(device)
    
    # Set different learning rates for different parts of the model
    params = [
        {"params": [p for n, p in model.named_parameters() 
                   if "backbone" in n], "lr": 0.0001},  # Lower learning rate for backbone
        {"params": [p for n, p in model.named_parameters() 
                   if "rpn" in n], "lr": 0.0005},  # Medium learning rate for RPN
        {"params": [p for n, p in model.named_parameters() 
                   if "backbone" not in n and "rpn" not in n], "lr": 0.001}  # Higher learning rate for heads
    ]
    
    # Use SGD optimizer with momentum
    optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)
    
    # Learning rate scheduler - cosine annealing
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=40, eta_min=1e-6
    )
    
    # Train model
    print(f"Training on {device}...")
    model = train_model(
        model, 
        train_loader, 
        val_loader, 
        optimizer, 
        lr_scheduler,
        num_epochs=60,  # Increased number of epochs
        device=device
    )
    
    # Save final model
    torch.save(model.state_dict(), 'final_model.pth')
    print('Training completed and model saved!')
    
    # Print model size
    model_size_mb = os.path.getsize('final_model.pth') / (1024 * 1024)
    print(f'Model size: {model_size_mb:.2f} MB')


if __name__ == '__main__':
    main() 
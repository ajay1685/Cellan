import datetime
import os
import cv2
import json
import torch
import math
import torch.optim as optim
from torchvision import transforms
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from torchvision.models.detection.mask_rcnn import MaskRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torch.utils.data import Dataset, DataLoader
from types import SimpleNamespace
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


class Boxes:
    """A minimal Boxes implementation matching Detectron2's API used by Instances"""
    def __init__(self, tensor):
        self.tensor = tensor
    
    def __len__(self):
        return self.tensor.shape[0]
    
    def to(self, device):
        return Boxes(self.tensor.to(device))
    
    def clone(self):
        return Boxes(self.tensor.clone())

class Instances:
    """A minimal Instances implementation matching Detectron2's API"""
    def __init__(self, image_size, **kwargs):
        self._image_size = image_size
        self._fields = {}
        for k, v in kwargs.items():
            self._fields[k] = v
    
    @property
    def image_size(self):
        return self._image_size
    
    def __getattr__(self, name):
        if name == "_fields" or name not in self._fields:
            raise AttributeError(f"{name} not found in Instances")
        return self._fields[name]
    
    def has(self, name):
        return name in self._fields
    
    def to(self, device):
        new_fields = {}
        for k, v in self._fields.items():
            if hasattr(v, "to"):
                new_fields[k] = v.to(device)
            else:
                new_fields[k] = v
        return Instances(self._image_size, **new_fields)

def collate_fn(batch):
    """Custom collate function for handling variable-sized images and targets"""
    return tuple(zip(*batch))

# Create custom learning rate scheduler to match Detectron2 behavior
class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, total_iters, warmup_iters, steps, gamma=0.5, last_epoch=-1):
        self.warmup_iters = warmup_iters
        self.steps = sorted(steps)
        self.gamma = gamma
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_iters:
            # Linear warmup
            alpha = float(self.last_epoch) / self.warmup_iters
            warmup_factor = (1.0 - alpha) * (1.0/3.0) + alpha  # Start from lr/3 and increase linearly
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            # Step decay at specific iterations
            decay_factor = self.gamma ** sum(self.last_epoch >= step for step in self.steps)
            return [base_lr * decay_factor for base_lr in self.base_lrs]

class CellDataset(Dataset):
    """Custom dataset for cell detection using COCO format annotations"""
    
    def __init__(self, annotation_path, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.annotation_data = json.load(open(annotation_path))
        
        # Define conversion from PIL Image to tensor
        self.to_tensor = transforms.ToTensor()
        
        # Process COCO annotations
        self.images = {img['id']: img for img in self.annotation_data['images']}
        self.annotations = {}
        
        # Group annotations by image_id
        for ann in self.annotation_data['annotations']:
            image_id = ann['image_id']
            if image_id not in self.annotations:
                self.annotations[image_id] = []
            self.annotations[image_id].append(ann)
            
        # Get category mapping
        self.categories = {}
        for cat in self.annotation_data['categories']:
            self.categories[cat['id']] = cat['name']
            
        # Create image_ids list for __getitem__
        self.image_ids = list(self.annotations.keys())
        
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_info = self.images[image_id]
        file_name = os.path.join(self.img_dir, image_info['file_name'])
        
        # Load image and convert to tensor directly
        img = Image.open(file_name).convert("RGB")
        img_tensor = self.to_tensor(img)  # Convert PIL Image to tensor
        
        # Process annotations
        annotations = self.annotations[image_id]
        boxes = []
        labels = []
        masks = []
        
        for ann in annotations:
            x, y, w, h = ann['bbox']
            
            # Add validation to ensure non-zero width and height
            # Add a small epsilon (e.g., 1.0) to ensure boxes have positive dimensions
            if w < 1.0:
                w = 1.0
            if h < 1.0:
                h = 1.0
                
            boxes.append([x, y, x + w, y + h])  # Convert to [x1, y1, x2, y2]
            labels.append(ann['category_id'])
            
            # Create binary mask from segmentation
            if 'segmentation' in ann:
                mask = self._create_mask(ann['segmentation'], image_info['height'], image_info['width'])
                masks.append(mask)
        
        # Convert to tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        
        # Assume all instances are not crowd
        iscrowd = torch.zeros((len(annotations),), dtype=torch.int64)
        
        # Create target dictionary
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': image_id,
            'area': area,
            'iscrowd': iscrowd
        }
        
        if len(masks) > 0:
            target['masks'] = torch.as_tensor(np.stack(masks), dtype=torch.uint8)
        
        if self.transform is not None:
            img, target = self.transform(img, target)
        
        # Return tensor instead of PIL Image
        return img_tensor, target
    
    def _create_mask(self, segmentation, height, width):
        """Create binary mask from segmentation data"""
        mask = np.zeros((height, width), dtype=np.uint8)
        for seg in segmentation:
            # For polygon segmentation
            if isinstance(seg, list) and len(seg) > 4:
                poly = np.array(seg).reshape(-1, 2)
                cv2.fillPoly(mask, [poly.astype(np.int32)], 1)
        return mask


class Detector:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.cell_mapping = None
        self.inferencing_framesize = None
        self.black_background = None
        self.current_detector = None

    def _get_model(self, num_classes):
        """Create a Mask R-CNN model with custom number of classes"""
        # Use weights="DEFAULT" instead of pretrained=True
        
        
        model = maskrcnn_resnet50_fpn_v2(weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
        
        # Replace classification head
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        
        # Replace mask head
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        model.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_features_mask, hidden_layer, num_classes)
        
        return model

    def train(self, path_to_annotation, path_to_trainingimages, path_to_detector, 
              iteration_num, inference_size, num_rois, black_background=0):
        """Train a cell detection model"""
        # Create output directory
        os.makedirs(path_to_detector, exist_ok=True)
        
        # Load and parse annotation file
        annotation_data = json.load(open(path_to_annotation))
        cell_names = []
        for i in annotation_data['categories']:
            if i['id'] > 0:
                cell_names.append(i['name'])
        
        print('Cell names in annotation file: ' + str(cell_names))
        
        # Create dataset
        dataset = CellDataset(path_to_annotation, path_to_trainingimages)
        
        # Calculate epochs based on iterations and dataset size
        batch_size = 4  # Similar to Detectron2 IMS_PER_BATCH
        base_lr = 0.001  # cfg.SOLVER.BASE_LR
        
        # Create data loader - replace the lambda with the imported function
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            collate_fn=collate_fn  # Use the function defined outside the class
        )
        
        # Create model
        num_classes = len(cell_names) + 1  # +1 for background class
        model = self._get_model(num_classes)

        # Configure ROI heads batch size
        if hasattr(model, 'roi_heads'):
            model.roi_heads.batch_size_per_image = num_rois  # Similar to cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE
            model.roi_heads.score_thresh = 0.5  # Similar to cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST
    
        model.to(self.device)
        
        # Set up optimizer
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.SGD(params, lr=base_lr, momentum=0.9, weight_decay=0.0001)
        
        # Calculate training iterations and schedule breakpoints
        # Use exact iteration numbers rather than epochs to match Detectron2
        total_iters = iteration_num
        current_iter = 0
        
        # Define warmup phase duration
        warmup_iters = int(total_iters * 0.1)  # 10% of total iterations for warmup

            # Define the step points at 40% and 80% of total iterations
        steps = [
            int(total_iters * 0.4),  # First step at 40% of training
            int(total_iters * 0.8)   # Second step at 80% of training
        ]
        # Create our custom scheduler
        lr_scheduler = WarmupMultiStepLR(
            optimizer, 
            total_iters=total_iters,
            warmup_iters=warmup_iters,
            steps=steps,
            gamma=0.5  # Matches your cfg.SOLVER.GAMMA
        )

        # Calculate epochs based on iterations and dataset size for progress tracking
        # This is just for display purposes, actual scheduling is iteration-based
        batches_per_epoch = len(data_loader)
        epochs = max(1, math.ceil(total_iters / batches_per_epoch))

        # Learning rate scheduler
        # lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(epochs*0.4), int(epochs*0.8)], gamma=0.5)
        # Learning rate scheduler; adjusts the learning rate during training
        #lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001, total_steps=epochs*len(dataset))

        # Training loop
        print("Starting training...")

        model.train()
        running_loss = 0.0
        epoch = 0
        
        while current_iter < total_iters:
            epoch += 1
            
            for i, (images, targets) in enumerate(data_loader):
                current_iter += 1
                
                # Move to device
                images = list(image.to(self.device) for image in images)
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                
                # Forward pass
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                
                # Backward pass
                optimizer.zero_grad()
                losses.backward()
                optimizer.step()
                
                # Update learning rate each iteration (like Detectron2)
                lr_scheduler.step()
                
                # Print progress
                running_loss += losses.item()
                print(f"Iteration: {current_iter}/{total_iters}, Epoch: {epoch}/{epochs}, "
                    f"Batch: {i+1}/{len(data_loader)}, Loss: {losses.item():.4f}, "
                    f"LR: {optimizer.param_groups[0]['lr']:.7f}")
                
                if current_iter >= total_iters:
                    break

        
        # Save model
        torch.save(model.state_dict(), os.path.join(path_to_detector, 'model_final.pth'))
        
        # Save model parameters
        model_parameters_dict = {
            'cell_names': cell_names,
            'inferencing_framesize': int(inference_size),
            'black_background': int(black_background),
            'cell_mapping': {i: name for i, name in enumerate(cell_names)}
        }
        
        with open(os.path.join(path_to_detector, 'model_parameters.txt'), 'w') as f:
            f.write(json.dumps(model_parameters_dict))
        
        # Save config
        config = {
            'num_classes': num_classes,
            'inference_size': inference_size,
        }
        
        with open(os.path.join(path_to_detector, 'config.yaml'), 'w') as f:
            f.write(json.dumps(config))
        
        print('Detector training completed!')

    def test(self, path_to_annotation, path_to_testingimages, path_to_detector, output_path):
        """Test trained detector on test data"""
        os.makedirs(output_path, exist_ok=True)
        
        # Load model parameters
        with open(os.path.join(path_to_detector, 'model_parameters.txt')) as f:
            model_parameters = json.loads(f.read())
        
        cell_names = model_parameters['cell_names']
        inference_size = int(model_parameters['inferencing_framesize'])
        bg = int(model_parameters['black_background'])
        
        print('The total categories of cells in this Detector: ' + str(cell_names))
        print('The inferencing framesize of this Detector: ' + str(inference_size))
        if bg == 0:
            print('The images that can be analyzed by this Detector have black/darker background')
        else:
            print('The images that can be analyzed by this Detector have white/lighter background')
            
        # Load config
        with open(os.path.join(path_to_detector, 'config.yaml')) as f:
            config = json.loads(f.read())
        
        # Create model
        num_classes = len(cell_names) + 1
        model = self._get_model(num_classes)
        model.load_state_dict(torch.load(os.path.join(path_to_detector, 'model_final.pth')))
        model.to(self.device)
        model.eval()
        
        # Create test dataset
        test_dataset = CellDataset(path_to_annotation, path_to_testingimages)
        
        # Use the collate_fn defined at module level instead of lambda
        test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=4,
            collate_fn=collate_fn  # Use the function defined outside the class
        )
        
        # Metrics for evaluation
        results = {
            'pred_boxes': [],
            'pred_classes': [],
            'pred_scores': [],
            'gt_boxes': [],
            'gt_classes': []
        }
        
        # Inference
        with torch.no_grad():
            for images, targets in test_loader:
                images = list(img.to(self.device) for img in images)
                outputs = model(images)
                
                # Process results for each image
                for i, (output, target) in enumerate(zip(outputs, targets)):
                    # Get image path from test_dataset
                    image_id = target['image_id'].item()
                    image_info = test_dataset.images[test_dataset.image_ids[image_id]]
                    img_path = os.path.join(path_to_testingimages, image_info['file_name'])
                    
                    # Load original image for visualization
                    orig_img = cv2.imread(img_path)
                    
                    # Draw predictions
                    pred_img = self._draw_predictions(
                        orig_img.copy(), 
                        output['boxes'].cpu().numpy(),
                        output['labels'].cpu().numpy(),
                        output['scores'].cpu().numpy(),
                        cell_names
                    )
                    
                    # Save output image
                    cv2.imwrite(
                        os.path.join(output_path, os.path.basename(img_path)),
                        pred_img
                    )
                    
                    # Collect results for evaluation
                    results['pred_boxes'].append(output['boxes'].cpu().numpy())
                    results['pred_classes'].append(output['labels'].cpu().numpy())
                    results['pred_scores'].append(output['scores'].cpu().numpy())
                    results['gt_boxes'].append(target['boxes'].cpu().numpy())
                    results['gt_classes'].append(target['labels'].cpu().numpy())
        
        # Calculate mAP
        mAP = self._calculate_map(results)
        
        print(f'The mean average precision (mAP) of the Detector is: {mAP:.4f}%.')
        print('Detector testing completed!')

    def _draw_predictions(self, image, boxes, labels, scores, class_names, threshold=0.5):
        """Draw bounding boxes and labels on image"""
        for box, label, score in zip(boxes, labels, scores):
            if score >= threshold:
                x1, y1, x2, y2 = box.astype(int)
                
                # Get class name (subtract 1 since label 0 is background)
                class_name = class_names[label - 1] if label > 0 and label <= len(class_names) else f"Class {label}"
                
                # Draw box
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw label
                cv2.putText(
                    image, 
                    f"{class_name} {score:.2f}", 
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, 
                    (0, 255, 0), 
                    1
                )
        
        return image

    def _calculate_map(self, results, iou_threshold=0.5):
        """Calculate mean Average Precision"""
        # This is a simplified mAP calculation
        # For a more robust implementation, consider using the pycocotools library
        
        def calculate_iou(boxA, boxB):
            # Calculate intersection over union
            xA = max(boxA[0], boxB[0])
            yA = max(boxA[1], boxB[1])
            xB = min(boxA[2], boxB[2])
            yB = min(boxA[3], boxB[3])
            
            interArea = max(0, xB - xA) * max(0, yB - yA)
            boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
            boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
            
            iou = interArea / float(boxAArea + boxBArea - interArea)
            return iou
        
        # Initialize AP for each class
        all_classes = set()
        for classes in results['gt_classes']:
            all_classes.update(classes)
        
        average_precisions = []
        
        # Calculate AP for each class
        for img_idx in range(len(results['pred_boxes'])):
            pred_boxes = results['pred_boxes'][img_idx]
            pred_classes = results['pred_classes'][img_idx]
            pred_scores = results['pred_scores'][img_idx]
            gt_boxes = results['gt_boxes'][img_idx]
            gt_classes = results['gt_classes'][img_idx]
            
            # Sort predictions by score
            indices = np.argsort(-pred_scores)
            pred_boxes = pred_boxes[indices]
            pred_classes = pred_classes[indices]
            pred_scores = pred_scores[indices]
            
            # Calculate precision and recall
            tp = np.zeros(len(pred_boxes))
            fp = np.zeros(len(pred_boxes))
            
            # Mark which ground-truth boxes have been detected
            detected_gt = [False] * len(gt_boxes)
            
            for pred_idx, (pred_box, pred_class) in enumerate(zip(pred_boxes, pred_classes)):
                # Find best matching ground truth box
                best_iou = 0
                best_gt_idx = -1
                
                for gt_idx, (gt_box, gt_class) in enumerate(zip(gt_boxes, gt_classes)):
                    # Only consider ground truth with same class
                    if gt_class != pred_class:
                        continue
                    
                    # Calculate IoU
                    iou = calculate_iou(pred_box, gt_box)
                    
                    # Select best matching ground truth
                    if iou > best_iou and not detected_gt[gt_idx]:
                        best_iou = iou
                        best_gt_idx = gt_idx
                
                # If IoU exceeds threshold, it's a true positive
                if best_iou >= iou_threshold:
                    tp[pred_idx] = 1
                    detected_gt[best_gt_idx] = True
                else:
                    fp[pred_idx] = 1
            
            # Calculate precision and recall
            tp_cumsum = np.cumsum(tp)
            fp_cumsum = np.cumsum(fp)
            recalls = tp_cumsum / max(1, len(gt_boxes))
            precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
            
            # Add sentinel values for AP calculation
            precisions = np.concatenate(([0], precisions, [0]))
            recalls = np.concatenate(([0], recalls, [1]))
            
            # Ensure precision decreases
            for i in range(len(precisions) - 2, -1, -1):
                precisions[i] = max(precisions[i], precisions[i + 1])
            
            # Calculate AP using precision recall curve
            indices = np.where(recalls[1:] != recalls[:-1])[0]
            ap = np.sum((recalls[indices + 1] - recalls[indices]) * precisions[indices + 1])
            average_precisions.append(ap)
        
        # Return mAP if there are detections, otherwise 0
        return np.mean(average_precisions) * 100 if average_precisions else 0.0

    def load(self, path_to_detector, cell_kinds):
        """Load a trained detector model"""
        # Load model parameters
        with open(os.path.join(path_to_detector, 'model_parameters.txt')) as f:
            model_parameters = json.loads(f.read())
        
        self.cell_mapping = model_parameters['cell_mapping']
        cell_names = model_parameters['cell_names']
        self.inferencing_framesize = int(model_parameters['inferencing_framesize'])
        bg = int(model_parameters['black_background'])
        
        print('The total categories of cells in this Detector: ' + str(cell_names))
        print('The cells of interest in this Detector: ' + str(cell_kinds))
        print('The inferencing framesize of this Detector: ' + str(self.inferencing_framesize))
        
        if bg == 0:
            self.black_background = True
            print('The images that can be analyzed by this Detector have black/darker background')
        else:
            self.black_background = False
            print('The images that can be analyzed by this Detector have white/lighter background')
        
        # Load config
        with open(os.path.join(path_to_detector, 'config.yaml')) as f:
            config = json.loads(f.read())
        
        # Create model
        num_classes = len(cell_names) + 1  # +1 for background
        model = self._get_model(num_classes)
        model.load_state_dict(torch.load(os.path.join(path_to_detector, 'model_final.pth')))
        model.to(self.device)
        model.eval()
        
        self.current_detector = model

  

    def inference(self, inputs):
        """Run inference on input images and return Detectron2-like output"""
        if self.current_detector is None:
            raise ValueError("No detector model loaded. Call load() first.")
        
        outputs = []
        
        for inp in inputs:
            # Handle input format - your analyzer passes {'image': tensor}
            if isinstance(inp, dict) and 'image' in inp:
                image = inp['image']
                if isinstance(image, torch.Tensor):
                    # If already a tensor, use it directly
                    image = image.to(self.device)
                else:
                    # Otherwise convert to tensor
                    image = torch.as_tensor(image, device=self.device)
            elif isinstance(inp, torch.Tensor):
                # If input is directly a tensor
                image = inp.to(self.device)
            elif isinstance(inp, np.ndarray):
                # If input is a numpy array, convert to tensor
                if inp.ndim == 3:  # HWC format
                    image = torch.from_numpy(inp.transpose(2, 0, 1)).float().to(self.device)
                elif inp.ndim == 4:  # NHWC format
                    image = torch.from_numpy(inp.transpose(0, 3, 1, 2)).float().to(self.device)
                else:
                    raise ValueError(f"Invalid input shape: {inp.shape}")
            else:
                raise ValueError(f"Unsupported input type: {type(inp)}")
            
            # Normalize if needed (depends on your model)
            if image.max() > 1.0:
                image = image / 255.0
                
            # Get original image height and width
            if image.dim() == 4:  # Batch of images (B, C, H, W)
                height, width = image.shape[2], image.shape[3]
            else:  # Single image (C, H, W)
                height, width = image.shape[1], image.shape[2]
                # Add batch dimension if needed
                if image.dim() == 3:
                    image = image.unsqueeze(0)
            
            # Run inference
            with torch.no_grad():
                self.current_detector.eval()
                predictions = self.current_detector(image)
            
            # Process predictions from PyTorch model
            if isinstance(predictions, list):
                # Assuming first item in batch
                prediction = predictions[0]
            else:
                prediction = predictions
                
            # Extract relevant outputs
            boxes = prediction.get("boxes", torch.empty(0, 4, device=self.device))
            scores = prediction.get("scores", torch.empty(0, device=self.device))
            labels = prediction.get("labels", torch.empty(0, dtype=torch.int64, device=self.device))
            masks = prediction.get("masks", None)
            
            # IMPORTANT: Adjust class IDs to be 0-indexed if they're 1-indexed
            # Check if we need to shift class IDs based on the cell_mapping
            if len(labels) > 0 and torch.min(labels).item() > 0:
                # If the minimum label is greater than 0, we need to adjust
                # This assumes your model uses 1-indexing but cell_mapping expects 0-indexing
                labels = labels - 1
            
            # Format masks properly
            if masks is not None:
                if masks.dim() == 4 and masks.shape[1] == 1:  # (N, 1, H, W)
                    masks = masks.squeeze(1)  # Convert to (N, H, W)
                elif masks.dim() == 3:
                    # Already in correct format (N, H, W)
                    pass
                else:
                    raise ValueError(f"Unexpected mask shape: {masks.shape}")
                    
                # Ensure masks are binary
                if masks.dtype != torch.bool:
                    masks = masks > 0.5
            
            # Create Instances object with fields
            instances = Instances(
                image_size=(height, width),
                pred_boxes=Boxes(boxes),
                scores=scores,
                pred_classes=labels,
                pred_masks=masks if masks is not None else torch.empty(0, height, width, dtype=torch.bool, device=self.device)
            )
            
            # Package in Detectron2-like format
            outputs.append({"instances": instances})
        
        # Return the outputs according to input format
        if len(outputs) == 1:
            return outputs
        
        return outputs
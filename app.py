#!/usr/bin/env python3
"""
REAL TRAINABLE MYCORRHIZAL AI SYSTEM - FIXED VERSION
Train with real images + masks, achieve 80% accuracy with 25 images per species
Fixed dtype errors and improved training stability
"""

import streamlit as st
import os
import json
import pandas as pd
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score, jaccard_score, precision_recall_fscore_support
import albumentations as A
from albumentations.pytorch import ToTensorV2
import time
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import warnings
import traceback
warnings.filterwarnings('ignore')

# Set device with better error handling
try:
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # Clear GPU memory
        st.success(f"🚀 GPU Available: {torch.cuda.get_device_name()}")
except Exception as e:
    DEVICE = torch.device('cpu')
    st.info(f"🔧 Using CPU: {str(e)}")

# Page config
st.set_page_config(
    page_title="Trainable Mycorrhizal AI",
    page_icon="🧬", 
    layout="wide"
)

def create_directories():
    """Create necessary directories with error handling"""
    directories = [
        "data/raw", "data/masks", "data/augmented", "data/species",
        "models/trained", "models/checkpoints", "results", "temp"
    ]
    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
        except Exception as e:
            st.error(f"Failed to create directory {directory}: {e}")

create_directories()

class MycorrhizalDataset(Dataset):
    """Dataset for mycorrhizal image segmentation with real training"""
    
    def __init__(self, image_paths: List[str], mask_paths: List[str], 
                 transform=None, augment=True):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.augment = augment
        
        # Validate data integrity
        assert len(image_paths) == len(mask_paths), "Mismatch between images and masks"
        
        # Advanced augmentation pipeline
        if augment:
            self.augmentation = A.Compose([
                # Geometric transforms
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.Rotate(limit=45, p=0.3),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=15, p=0.3),
                
                # Optical transforms
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
                A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.3),
                A.RandomGamma(gamma_limit=(1.0, 1.5), p=0.3),
                
                # Noise and blur
                A.GaussNoise(var_limit=(10, 50), p=0.2),
                A.GaussianBlur(blur_limit=3, p=0.2),
                A.MotionBlur(blur_limit=3, p=0.2),
                
                # Elastic transforms for biological variation
                A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.2),
                A.GridDistortion(p=0.2),
                
                # Normalization
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        else:
            self.augmentation = A.Compose([
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        try:
            # Load image and mask
            image = cv2.imread(self.image_paths[idx])
            if image is None:
                raise ValueError(f"Could not load image: {self.image_paths[idx]}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise ValueError(f"Could not load mask: {self.mask_paths[idx]}")
            
            # Resize to standard size
            image = cv2.resize(image, (256, 256))
            mask = cv2.resize(mask, (256, 256))
            
            # Binary threshold for mask - ensure proper binary format
            mask = (mask > 127).astype(np.float32)  # Use float32 instead of uint8
            
            # Validate mask has both classes
            mask_sum = np.sum(mask)
            total_pixels = mask.size
            colonization_pct = (mask_sum / total_pixels) * 100
            
            if mask_sum == 0:
                st.warning(f"⚠️ Mask {os.path.basename(self.mask_paths[idx])} has NO mycorrhizal structures (0% colonization)")
            elif mask_sum == total_pixels:
                st.warning(f"⚠️ Mask {os.path.basename(self.mask_paths[idx])} is 100% mycorrhizal (may be incorrect)")
            else:
                st.info(f"✅ Mask {os.path.basename(self.mask_paths[idx])}: {colonization_pct:.1f}% colonization")
            
            # Apply augmentations
            if self.augmentation:
                augmented = self.augmentation(image=image, mask=mask)
                image = augmented['image']
                mask = augmented['mask']
            
            # Ensure mask is float tensor
            if not isinstance(mask, torch.Tensor):
                mask = torch.from_numpy(mask).float()
            else:
                mask = mask.float()  # Ensure it's float type
            
            # Ensure image is float tensor
            if not isinstance(image, torch.Tensor):
                image = torch.from_numpy(image).float()
            else:
                image = image.float()
            
            return image, mask
            
        except Exception as e:
            st.error(f"Error loading data at index {idx}: {e}")
            # Return a dummy sample to prevent crash
            dummy_image = torch.zeros(3, 256, 256, dtype=torch.float32)
            dummy_mask = torch.zeros(256, 256, dtype=torch.float32)
            return dummy_image, dummy_mask

class UNet(nn.Module):
    """U-Net architecture optimized for mycorrhizal segmentation"""
    
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super(UNet, self).__init__()
        
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature
        
        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv(feature*2, feature))
        
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        
    def forward(self, x):
        skip_connections = []
        
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]
            
            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:])
            
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)
        
        return torch.sigmoid(self.final_conv(x))

class DoubleConv(nn.Module):
    """Double convolution block"""
    
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        return self.conv(x)

class DiceLoss(nn.Module):
    """Dice loss for segmentation"""
    
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, predictions, targets):
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        
        intersection = (predictions * targets).sum()
        dice = (2. * intersection + self.smooth) / (predictions.sum() + targets.sum() + self.smooth)
        
        return 1 - dice

class CombinedLoss(nn.Module):
    """Combined BCE and Dice loss"""
    
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCELoss()
        self.dice = DiceLoss()
    
    def forward(self, predictions, targets):
        bce_loss = self.bce(predictions, targets)
        dice_loss = self.dice(predictions, targets)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss

class MycorrhizalTrainer:
    """Real trainer for mycorrhizal AI with 25-image capability"""
    
    def __init__(self, species_name: str):
        self.species_name = species_name
        self.model = UNet(in_channels=3, out_channels=1).to(DEVICE)
        self.criterion = CombinedLoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=1e-3, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=10, factor=0.5
        )
        
        self.train_losses = []
        self.val_losses = []
        self.val_ious = []
        self.val_dices = []
        
        # Print model info
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        st.info(f"🧠 Model: {total_params:,} total params, {trainable_params:,} trainable")
        
    def prepare_data(self, image_dir: str, mask_dir: str, 
                    train_split: float = 0.8) -> Tuple[DataLoader, DataLoader]:
        """Prepare training data with aggressive augmentation"""
        
        # Get image and mask paths with better matching
        image_files = sorted([f for f in os.listdir(image_dir) 
                            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.tif'))])
        
        image_paths = []
        mask_paths = []
        unmatched_images = []
        
        for img_file in image_files:
            img_path = os.path.join(image_dir, img_file)
            
            # Try multiple mask naming conventions
            base_name = os.path.splitext(img_file)[0]
            possible_mask_names = [
                f"{base_name}_mask.png",
                f"{base_name}_mask.jpg",
                f"{base_name}.png",
                img_file.replace('.jpg', '_mask.png').replace('.jpeg', '_mask.png').replace('.tiff', '_mask.png')
            ]
            
            mask_found = False
            for mask_name in possible_mask_names:
                mask_path = os.path.join(mask_dir, mask_name)
                if os.path.exists(mask_path):
                    image_paths.append(img_path)
                    mask_paths.append(mask_path)
                    mask_found = True
                    break
            
            if not mask_found:
                unmatched_images.append(img_file)
        
        if unmatched_images:
            st.warning(f"⚠️ {len(unmatched_images)} images without matching masks: {unmatched_images[:3]}...")
        
        if len(image_paths) < 5:
            raise ValueError(f"Need at least 5 matched image-mask pairs, found {len(image_paths)}")
        
        # WARNING: Check for insufficient training data
        if len(image_paths) < 25:
            st.error(f"⚠️ **INSUFFICIENT TRAINING DATA**: Found only {len(image_paths)} samples!")
            st.error("🚨 **Expected Issues:**")
            st.error("• Model will severely overfit")
            st.error("• Predictions will be unreliable") 
            st.error("• May predict 100% colonization for everything")
            st.error("**Recommendation**: Upload at least 25 diverse images per species")
        
        st.success(f"✅ Found {len(image_paths)} matched image-mask pairs")
        
        # Validate mask quality BEFORE training
        st.info("🔍 Validating mask quality...")
        mask_stats = []
        for mask_path in mask_paths:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                mask_binary = (mask > 127).astype(np.float32)
                colonization_pct = (np.sum(mask_binary) / mask_binary.size) * 100
                mask_stats.append(colonization_pct)
        
        if mask_stats:
            avg_colonization = np.mean(mask_stats)
            st.info(f"📊 Average colonization in training data: {avg_colonization:.1f}%")
            
            if avg_colonization > 80:
                st.warning("⚠️ Very high average colonization - model may predict 100% for everything")
            elif avg_colonization < 5:
                st.warning("⚠️ Very low average colonization - check if masks are correct")
        
        # Split data
        total_size = len(image_paths)
        train_size = int(train_split * total_size)
        
        indices = list(range(total_size))
        np.random.shuffle(indices)
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:] if len(indices) > train_size else [indices[-1]]  # Ensure at least 1 val sample
        
        train_img_paths = [image_paths[i] for i in train_indices]
        train_mask_paths = [mask_paths[i] for i in train_indices]
        val_img_paths = [image_paths[i] for i in val_indices]
        val_mask_paths = [mask_paths[i] for i in val_indices]
        
        st.info(f"📊 Training: {len(train_img_paths)} samples, Validation: {len(val_img_paths)} samples")
        
        # Create datasets with aggressive augmentation for training
        train_dataset = MycorrhizalDataset(train_img_paths, train_mask_paths, augment=True)
        val_dataset = MycorrhizalDataset(val_img_paths, val_mask_paths, augment=False)
        
        # Adjust batch size based on available memory and data size
        batch_size = min(2 if DEVICE.type == 'cuda' else 1, len(train_img_paths))
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        return train_loader, val_loader
    
    def calculate_metrics(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """Calculate segmentation metrics with error handling"""
        try:
            predictions = (predictions > 0.5).float()
            
            # Flatten for metric calculation
            pred_flat = predictions.view(-1).cpu().numpy()
            target_flat = targets.view(-1).cpu().numpy()
            
            # IoU (Jaccard)
            intersection = np.logical_and(pred_flat, target_flat).sum()
            union = np.logical_or(pred_flat, target_flat).sum()
            iou = intersection / (union + 1e-8)
            
            # Dice
            dice = 2 * intersection / (pred_flat.sum() + target_flat.sum() + 1e-8)
            
            # Pixel accuracy
            pixel_acc = np.mean(pred_flat == target_flat)
            
            return {
                'iou': float(iou),
                'dice': float(dice),
                'pixel_accuracy': float(pixel_acc)
            }
        except Exception as e:
            st.error(f"Error calculating metrics: {e}")
            return {'iou': 0.0, 'dice': 0.0, 'pixel_accuracy': 0.0}
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train one epoch with error handling"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        try:
            for batch_idx, (images, masks) in enumerate(train_loader):
                # Ensure proper data types
                images = images.to(DEVICE).float()  # Ensure float32
                masks = masks.to(DEVICE).float().unsqueeze(1)  # Ensure float32 and add channel dim
                
                # Validate tensor shapes and types
                if images.dtype != torch.float32:
                    st.warning(f"Converting images from {images.dtype} to float32")
                    images = images.float()
                    
                if masks.dtype != torch.float32:
                    st.warning(f"Converting masks from {masks.dtype} to float32")
                    masks = masks.float()
                
                self.optimizer.zero_grad()
                
                outputs = self.model(images)
                
                # Ensure output and target have same shape
                if outputs.shape != masks.shape:
                    st.warning(f"Shape mismatch: outputs {outputs.shape} vs masks {masks.shape}")
                    masks = F.interpolate(masks, size=outputs.shape[2:], mode='nearest')
                
                loss = self.criterion(outputs, masks)
                
                # Check for NaN loss
                if torch.isnan(loss):
                    st.warning(f"NaN loss detected at batch {batch_idx}")
                    continue
                
                loss.backward()
                
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
        except Exception as e:
            st.error(f"Error during training epoch: {e}")
            st.error(f"Error type: {type(e).__name__}")
            return float('inf')
        
        return total_loss / max(num_batches, 1)
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate model with error handling"""
        self.model.eval()
        total_loss = 0
        all_metrics = {'iou': [], 'dice': [], 'pixel_accuracy': []}
        num_batches = 0
        
        try:
            with torch.no_grad():
                for images, masks in val_loader:
                    # Ensure proper data types
                    images = images.to(DEVICE).float()
                    masks = masks.to(DEVICE).float().unsqueeze(1)
                    
                    outputs = self.model(images)
                    
                    # Ensure output and target have same shape
                    if outputs.shape != masks.shape:
                        masks = F.interpolate(masks, size=outputs.shape[2:], mode='nearest')
                    
                    loss = self.criterion(outputs, masks)
                    
                    if not torch.isnan(loss):
                        total_loss += loss.item()
                        num_batches += 1
                        
                        # Calculate metrics
                        metrics = self.calculate_metrics(outputs, masks)
                        for key, value in metrics.items():
                            all_metrics[key].append(value)
        except Exception as e:
            st.error(f"Error during validation: {e}")
        
        # Average metrics
        avg_metrics = {}
        for key, values in all_metrics.items():
            avg_metrics[key] = np.mean(values) if values else 0.0
        
        avg_metrics['loss'] = total_loss / max(num_batches, 1)
        
        return avg_metrics
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
              epochs: int = 100, patience: int = 20) -> Dict:
        """Full training loop with early stopping and error handling"""
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        training_log = {
            'epochs_trained': 0,
            'best_val_loss': float('inf'),
            'best_val_iou': 0,
            'final_metrics': {},
            'training_successful': False
        }
        
        st.info(f"🚀 Starting training for {self.species_name} with {len(train_loader.dataset)} training samples")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        metrics_placeholder = st.empty()
        
        try:
            for epoch in range(epochs):
                # Training
                train_loss = self.train_epoch(train_loader)
                self.train_losses.append(train_loss)
                
                # Validation
                val_metrics = self.validate(val_loader)
                val_loss = val_metrics['loss']
                val_iou = val_metrics['iou']
                val_dice = val_metrics['dice']
                
                self.val_losses.append(val_loss)
                self.val_ious.append(val_iou)
                self.val_dices.append(val_dice)
                
                # Learning rate scheduling
                if not np.isnan(val_loss) and not np.isinf(val_loss):
                    self.scheduler.step(val_loss)
                
                # Early stopping
                if val_loss < best_val_loss and not np.isnan(val_loss):
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best model
                    self.save_checkpoint(epoch, val_metrics, is_best=True)
                    training_log['best_val_loss'] = val_loss
                    training_log['best_val_iou'] = val_iou
                    training_log['training_successful'] = True
                else:
                    patience_counter += 1
                
                # Update UI
                progress = (epoch + 1) / epochs
                progress_bar.progress(progress)
                
                status_text.text(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val IoU: {val_iou:.4f}")
                
                # Display metrics
                col1, col2, col3, col4 = metrics_placeholder.columns(4)
                col1.metric("Train Loss", f"{train_loss:.4f}")
                col2.metric("Val IoU", f"{val_iou:.4f}")
                col3.metric("Val Dice", f"{val_dice:.4f}")
                col4.metric("Pixel Acc", f"{val_metrics['pixel_accuracy']:.4f}")
                
                if patience_counter >= patience:
                    st.success(f"🎯 Early stopping at epoch {epoch+1}")
                    break
                    
                # Emergency stop if loss becomes NaN
                if np.isnan(train_loss) or np.isnan(val_loss):
                    st.error("❌ Training stopped due to NaN loss")
                    break
            
            training_log['epochs_trained'] = epoch + 1
            training_log['final_metrics'] = val_metrics
            
        except Exception as e:
            st.error(f"❌ Training failed with error: {e}")
            st.error(traceback.format_exc())
            training_log['training_successful'] = False
        
        return training_log
    
    def save_checkpoint(self, epoch: int, metrics: Dict, is_best: bool = False):
        """Save model checkpoint with error handling"""
        try:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'metrics': metrics,
                'species_name': self.species_name,
                'train_losses': self.train_losses,
                'val_losses': self.val_losses,
                'val_ious': self.val_ious,
                'val_dices': self.val_dices
            }
            
            checkpoint_dir = os.path.join("models/trained", self.species_name)
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            if is_best:
                torch.save(checkpoint, os.path.join(checkpoint_dir, 'best_model.pth'))
            
            torch.save(checkpoint, os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth'))
            
        except Exception as e:
            st.error(f"Error saving checkpoint: {e}")
    
    def load_model(self, checkpoint_path: str):
        """Load trained model with error handling"""
        try:
            checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.train_losses = checkpoint.get('train_losses', [])
            self.val_losses = checkpoint.get('val_losses', [])
            self.val_ious = checkpoint.get('val_ious', [])
            return checkpoint['metrics']
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return {}

class MycorrhizalPredictor:
    """Real-time prediction with trained models"""
    
    def __init__(self, model_path: str):
        try:
            self.model = UNet(in_channels=3, out_channels=1).to(DEVICE)
            checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            self.transform = A.Compose([
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
            
            st.success("✅ Model loaded successfully")
            
        except Exception as e:
            st.error(f"Error loading predictor: {e}")
            raise
    
    def predict(self, image_path: str) -> Dict:
        """Predict mycorrhizal colonization with error handling"""
        try:
            # Load and preprocess image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
                
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            original_shape = image.shape[:2]
            
            # Resize for model
            image_resized = cv2.resize(image, (256, 256))
            
            # Transform
            transformed = self.transform(image=image_resized)
            image_tensor = transformed['image'].unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                prediction = self.model(image_tensor)
                prediction = prediction.squeeze().cpu().numpy()
            
            # Resize back to original shape
            prediction_resized = cv2.resize(prediction, (original_shape[1], original_shape[0]))
            
            # Calculate metrics
            binary_mask = (prediction_resized > 0.5).astype(np.uint8)
            colonization_percentage = (np.sum(binary_mask) / binary_mask.size) * 100
            
            # Confidence based on prediction clarity
            confidence = np.std(prediction_resized)  # Higher std = more confident boundaries
            
            return {
                'colonization_percentage': float(colonization_percentage),
                'confidence': float(min(confidence * 10, 1.0)),  # Scale to 0-1
                'binary_mask': binary_mask,
                'probability_mask': prediction_resized,
                'original_image_shape': original_shape
            }
            
        except Exception as e:
            st.error(f"Error during prediction: {e}")
            return {
                'colonization_percentage': 0.0,
                'confidence': 0.0,
                'binary_mask': np.zeros((100, 100)),
                'probability_mask': np.zeros((100, 100)),
                'original_image_shape': (100, 100)
            }

def debug_file_matching():
    st.header("🔍 Debug File Matching")
    st.markdown("Let's see exactly what files exist and what the system is looking for")
    
    # Species selection
    species_dirs = []
    if os.path.exists("data/species"):
        species_dirs = [d for d in os.listdir("data/species") if os.path.isdir(os.path.join("data/species", d))]
    
    if not species_dirs:
        st.error("❌ No species directories found")
        return
    
    selected_species = st.selectbox("Select Species to Debug:", species_dirs)
    
    if selected_species:
        images_dir = os.path.join("data/species", selected_species, "images")
        masks_dir = os.path.join("data/species", selected_species, "masks")
        
        st.subheader(f"🧬 Debugging: {selected_species}")
        
        # Check if directories exist
        col1, col2 = st.columns(2)
        with col1:
            if os.path.exists(images_dir):
                st.success(f"✅ Images directory exists: `{images_dir}`")
            else:
                st.error(f"❌ Images directory missing: `{images_dir}`")
                return
        
        with col2:
            if os.path.exists(masks_dir):
                st.success(f"✅ Masks directory exists: `{masks_dir}`")
            else:
                st.error(f"❌ Masks directory missing: `{masks_dir}`")
                return
        
        # Get ALL files (not just valid extensions)
        try:
            all_image_files = os.listdir(images_dir)
            all_mask_files = os.listdir(masks_dir)
        except Exception as e:
            st.error(f"❌ Error reading directories: {e}")
            return
        
        st.markdown("---")
        
        # Show ALL files found
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📂 ALL Files in Images Directory")
            st.code(f"Path: {images_dir}")
            if all_image_files:
                for i, file in enumerate(all_image_files, 1):
                    file_path = os.path.join(images_dir, file)
                    size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
                    st.write(f"{i}. `{file}` ({size:,} bytes)")
                st.info(f"Total files: {len(all_image_files)}")
            else:
                st.warning("📭 No files found in images directory")
        
        with col2:
            st.subheader("📂 ALL Files in Masks Directory")
            st.code(f"Path: {masks_dir}")
            if all_mask_files:
                for i, file in enumerate(all_mask_files, 1):
                    file_path = os.path.join(masks_dir, file)
                    size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
                    st.write(f"{i}. `{file}` ({size:,} bytes)")
                st.info(f"Total files: {len(all_mask_files)}")
            else:
                st.warning("📭 No files found in masks directory")
        
        st.markdown("---")
        
        # Filter valid image files
        valid_extensions = ('.png', '.jpg', '.jpeg', '.tiff', '.tif')
        valid_image_files = [f for f in all_image_files if f.lower().endswith(valid_extensions)]
        valid_mask_files = [f for f in all_mask_files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        st.subheader("🎯 Valid Files (by extension)")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Valid Images:**")
            for img in valid_image_files:
                st.write(f"• `{img}`")
            st.info(f"Valid images: {len(valid_image_files)}")
        
        with col2:
            st.write("**Valid Masks:**")
            for mask in valid_mask_files:
                st.write(f"• `{mask}`")
            st.info(f"Valid masks: {len(valid_mask_files)}")
        
        st.markdown("---")
        
        # Test matching logic
        st.subheader("🔗 Testing Matching Logic")
        
        if valid_image_files:
            st.write("**For each image, looking for these mask names:**")
            
            found_matches = []
            
            for img_file in valid_image_files:
                base_name = os.path.splitext(img_file)[0]
                
                # All possible mask names we're looking for
                possible_mask_names = [
                    f"{base_name}_mask.png",
                    f"{base_name}_mask.jpg",
                    f"{base_name}-mask.png",
                    f"{base_name}mask.png",
                    f"{base_name}.png",
                ]
                
                with st.expander(f"🔍 For image: `{img_file}`"):
                    st.write("**Looking for any of these mask files:**")
                    
                    match_found = False
                    for mask_name in possible_mask_names:
                        mask_path = os.path.join(masks_dir, mask_name)
                        exists = os.path.exists(mask_path)
                        
                        if exists:
                            st.success(f"✅ `{mask_name}` - **FOUND**")
                            found_matches.append((img_file, mask_name))
                            match_found = True
                        else:
                            st.write(f"❌ `{mask_name}` - not found")
                    
                    if not match_found:
                        st.error("❌ **No matching mask found for this image**")
                        
                        # Show what mask files actually exist
                        st.write("**Available mask files:**")
                        for mask in valid_mask_files:
                            st.write(f"• `{mask}`")
            
            # Summary
            st.markdown("---")
            st.subheader("📊 Matching Summary")
            
            if found_matches:
                st.success(f"✅ Found {len(found_matches)} matches:")
                for img, mask in found_matches:
                    st.write(f"• `{img}` ↔ `{mask}`")
            else:
                st.error("❌ **NO MATCHES FOUND!**")
                
                st.markdown("""
                ### 🛠️ **Fix Required:**
                
                **The Problem:** Your mask file names don't match the expected patterns.
                
                **Solutions:**
                """)
                
                # Show specific renaming needed
                if valid_image_files and valid_mask_files:
                    st.markdown("**Option 1: Rename masks to match images**")
                    
                    if len(valid_image_files) == len(valid_mask_files):
                        st.info("✅ Same number of images and masks - can pair them")
                        
                        rename_suggestions = []
                        for i, (img, current_mask) in enumerate(zip(valid_image_files, valid_mask_files)):
                            base_name = os.path.splitext(img)[0]
                            suggested_name = f"{base_name}_mask.png"
                            rename_suggestions.append((current_mask, suggested_name))
                        
                        st.write("**Rename these files:**")
                        for old_name, new_name in rename_suggestions[:10]:
                            st.code(f"mv '{old_name}' '{new_name}'")
                        
                        if len(rename_suggestions) > 10:
                            st.write(f"... and {len(rename_suggestions)-10} more")
                    
                    else:
                        st.warning(f"⚠️ Number mismatch: {len(valid_image_files)} images vs {len(valid_mask_files)} masks")
        
        # Manual file existence check
        st.markdown("---")
        st.subheader("🧪 Manual File Check")
        st.markdown("Enter exact filenames to check if they exist:")
        
        col1, col2 = st.columns(2)
        with col1:
            test_image = st.text_input("Image filename:", placeholder="e.g., IMG10_original.png")
            if test_image:
                img_path = os.path.join(images_dir, test_image)
                if os.path.exists(img_path):
                    st.success(f"✅ Found: `{test_image}`")
                else:
                    st.error(f"❌ Not found: `{test_image}`")
        
        with col2:
            test_mask = st.text_input("Mask filename:", placeholder="e.g., IMG10_original_mask.png")
            if test_mask:
                mask_path = os.path.join(masks_dir, test_mask)
                if os.path.exists(mask_path):
                    st.success(f"✅ Found: `{test_mask}`")
                else:
                    st.error(f"❌ Not found: `{test_mask}`")

def training_data_tab():
    st.header("📚 Training Data Setup")
    
    st.markdown("""
    ### 📋 Data Requirements:
    - **Images**: 25+ microscopy images (.jpg, .png, .tiff)
    - **Masks**: Corresponding binary masks (_mask.png suffix)
    - **Structure**: `data/species/{species_name}/images/` and `data/species/{species_name}/masks/`
    """)
    
    # Species setup
    st.subheader("🧬 Species Configuration")
    species_name = st.text_input("Species Name:", placeholder="e.g., Glomus_intraradices")
    
    if species_name:
        species_dir = os.path.join("data/species", species_name)
        images_dir = os.path.join(species_dir, "images")
        masks_dir = os.path.join(species_dir, "masks")
        
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(masks_dir, exist_ok=True)
        
        st.success(f"✅ Directories created for {species_name}")
        
        # File upload
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📸 Upload Images")
            uploaded_images = st.file_uploader(
                "Upload microscopy images",
                type=['png', 'jpg', 'jpeg', 'tiff', 'tif'],
                accept_multiple_files=True,
                key="images"
            )
            
            if uploaded_images:
                for img_file in uploaded_images:
                    img_path = os.path.join(images_dir, img_file.name)
                    with open(img_path, "wb") as f:
                        f.write(img_file.getbuffer())
                st.success(f"✅ {len(uploaded_images)} images saved")
        
        with col2:
            st.subheader("🎭 Upload Masks")
            uploaded_masks = st.file_uploader(
                "Upload binary masks (white=mycorrhizal, black=background)",
                type=['png'],
                accept_multiple_files=True,
                key="masks"
            )
            
            if uploaded_masks:
                for mask_file in uploaded_masks:
                    mask_path = os.path.join(masks_dir, mask_file.name)
                    with open(mask_path, "wb") as f:
                        f.write(mask_file.getbuffer())
                st.success(f"✅ {len(uploaded_masks)} masks saved")
        
        # Data summary
        if os.path.exists(images_dir) and os.path.exists(masks_dir):
            images = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.tif'))]
            masks = [f for f in os.listdir(masks_dir) if f.endswith('.png')]
            
            st.subheader("📊 Data Summary")
            col1, col2, col3 = st.columns(3)
            col1.metric("Images", len(images))
            col2.metric("Masks", len(masks))
            col3.metric("Ready for Training", "✅" if len(images) >= 5 and len(masks) >= 5 else "❌")

def train_model_tab():
    st.header("🚀 Train New AI Model")
    
    # Species selection
    species_dirs = []
    if os.path.exists("data/species"):
        species_dirs = [d for d in os.listdir("data/species") if os.path.isdir(os.path.join("data/species", d))]
    
    if not species_dirs:
        st.warning("⚠️ No species data found. Please setup training data first.")
        return
    
    selected_species = st.selectbox("Select Species to Train:", species_dirs)
    
    if selected_species:
        images_dir = os.path.join("data/species", selected_species, "images")
        masks_dir = os.path.join("data/species", selected_species, "masks")
        
        # Check data availability
        if os.path.exists(images_dir) and os.path.exists(masks_dir):
            images = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.tif'))]
            masks = [f for f in os.listdir(masks_dir) if f.endswith('.png')]
            
            st.info(f"📊 Found {len(images)} images and {len(masks)} masks")
            
            # CRITICAL WARNING for insufficient data
            if len(images) < 25:
                st.error("🚨 **CRITICAL WARNING: INSUFFICIENT TRAINING DATA**")
                st.error(f"• You have only **{len(images)} images** (need 25+ for reliable results)")
                st.error("• **Expected problems with {len(images)} images:**")
                st.error("  - Model will severely overfit")
                st.error("  - Will likely predict 100% colonization for everything")
                st.error("  - Predictions will be unreliable")
                st.error("• **Recommendation**: Add more diverse images before training")
                
                if len(images) < 10:
                    st.error("⛔ **Training with <10 images is not recommended**")
                    if st.checkbox("⚠️ I understand the risks and want to proceed anyway"):
                        proceed_anyway = True
                    else:
                        proceed_anyway = False
                else:
                    proceed_anyway = True
            else:
                proceed_anyway = True
            
            if len(images) >= 5 and proceed_anyway:
                # Training parameters
                col1, col2, col3 = st.columns(3)
                with col1:
                    epochs = st.slider("Training Epochs:", 50, 300, 100)
                with col2:
                    train_split = st.slider("Training Split:", 0.6, 0.9, 0.8)
                with col3:
                    patience = st.slider("Early Stopping Patience:", 10, 50, 20)
                
                if st.button(f"🚀 Start Training {selected_species}", type="primary"):
                    try:
                        # Initialize trainer
                        trainer = MycorrhizalTrainer(selected_species)
                        
                        # Prepare data
                        st.info("📊 Preparing training data...")
                        train_loader, val_loader = trainer.prepare_data(images_dir, masks_dir, train_split)
                        
                        # Start training
                        training_log = trainer.train(train_loader, val_loader, epochs, patience)
                        
                        # Save training log
                        if training_log['training_successful']:
                            log_path = os.path.join("models/trained", selected_species, "training_log.json")
                            os.makedirs(os.path.dirname(log_path), exist_ok=True)
                            with open(log_path, 'w') as f:
                                json.dump(training_log, f, indent=2)
                            
                            # Display results
                            st.success("🎉 Training completed!")
                            
                            col1, col2, col3 = st.columns(3)
                            col1.metric("Epochs Trained", training_log['epochs_trained'])
                            col2.metric("Best Val IoU", f"{training_log['best_val_iou']:.3f}")
                            col3.metric("Final Accuracy", f"{training_log['final_metrics']['pixel_accuracy']:.1%}")
                            
                            if training_log['best_val_iou'] > 0.8:
                                st.balloons()
                                st.success("🎯 Achieved >80% accuracy target!")
                        else:
                            st.error("❌ Training was not successful")
                        
                    except Exception as e:
                        st.error(f"❌ Training failed: {str(e)}")
                        st.error(traceback.format_exc())
            else:
                st.warning(f"⚠️ Need at least 5 images, found {len(images)}")

def test_model_tab():
    st.header("🔍 Test Trained Models")
    
    # Find trained models
    trained_models = []
    if os.path.exists("models/trained"):
        for species in os.listdir("models/trained"):
            model_path = os.path.join("models/trained", species, "best_model.pth")
            if os.path.exists(model_path):
                trained_models.append((species, model_path))
    
    if not trained_models:
        st.warning("⚠️ No trained models found. Please train a model first.")
        return
    
    # Model selection
    model_names = [species for species, _ in trained_models]
    selected_model = st.selectbox("Select Trained Model:", model_names)
    
    if selected_model:
        model_path = next(path for species, path in trained_models if species == selected_model)
        
        st.success(f"✅ Loaded model: {selected_model}")
        
        # Image upload for testing
        uploaded_test_images = st.file_uploader(
            "Upload test images",
            type=['png', 'jpg', 'jpeg', 'tiff', 'tif'],
            accept_multiple_files=True
        )
        
        if uploaded_test_images and st.button("🔍 Run Prediction", type="primary"):
            try:
                # Initialize predictor
                predictor = MycorrhizalPredictor(model_path)
                
                results = []
                
                for i, img_file in enumerate(uploaded_test_images):
                    st.text(f"🔍 Analyzing {img_file.name}...")
                    
                    # Save temp file
                    temp_path = os.path.join("temp", img_file.name)
                    with open(temp_path, "wb") as f:
                        f.write(img_file.getbuffer())
                    
                    # Predict
                    prediction = predictor.predict(temp_path)
                    
                    results.append({
                        'Image': img_file.name,
                        'Colonization %': f"{prediction['colonization_percentage']:.1f}%",
                        'Confidence': f"{prediction['confidence']:.2f}",
                        'Model': selected_model
                    })
                    
                    # Display prediction
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.image(temp_path, caption=img_file.name, width=300)
                    
                    with col2:
                        st.subheader("🎯 Prediction Results")
                        st.metric("Colonization", f"{prediction['colonization_percentage']:.1f}%")
                        st.metric("Confidence", f"{prediction['confidence']:.2f}")
                        
                        # Show mask overlay
                        original_img = cv2.imread(temp_path)
                        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
                        
                        # Create overlay
                        mask_colored = np.zeros_like(original_img)
                        mask_colored[:, :, 0] = prediction['binary_mask'] * 255  # Red channel
                        
                        overlay = cv2.addWeighted(original_img, 0.7, mask_colored, 0.3, 0)
                        st.image(overlay, caption="Detected Mycorrhizal Structures", width=300)
                    
                    # Cleanup
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                
                # Results summary
                if results:
                    st.subheader("📊 Results Summary")
                    df = pd.DataFrame(results)
                    st.dataframe(df, use_container_width=True)
                    
                    # Download results
                    csv = df.to_csv(index=False)
                    st.download_button(
                        "📥 Download Results",
                        csv,
                        f"predictions_{selected_model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        "text/csv"
                    )
                
            except Exception as e:
                st.error(f"❌ Prediction failed: {str(e)}")
                st.error(traceback.format_exc())

def model_management_tab():
    st.header("📊 Model Management")
    
    # List trained models
    if os.path.exists("models/trained"):
        species_dirs = [d for d in os.listdir("models/trained") 
                       if os.path.isdir(os.path.join("models/trained", d))]
        
        if species_dirs:
            st.subheader("🎯 Trained Models")
            
            for species in species_dirs:
                with st.expander(f"🧬 {species}", expanded=True):
                    model_dir = os.path.join("models/trained", species)
                    
                    # Check for best model
                    best_model_path = os.path.join(model_dir, "best_model.pth")
                    log_path = os.path.join(model_dir, "training_log.json")
                    
                    if os.path.exists(best_model_path):
                        st.success("✅ Trained model available")
                        
                        # Load training log if available
                        if os.path.exists(log_path):
                            try:
                                with open(log_path, 'r') as f:
                                    log = json.load(f)
                                
                                col1, col2, col3, col4 = st.columns(4)
                                col1.metric("Epochs", log['epochs_trained'])
                                col2.metric("Best IoU", f"{log['best_val_iou']:.3f}")
                                col3.metric("Val Loss", f"{log['best_val_loss']:.4f}")
                                if 'final_metrics' in log and 'pixel_accuracy' in log['final_metrics']:
                                    col4.metric("Accuracy", f"{log['final_metrics']['pixel_accuracy']:.1%}")
                            except Exception as e:
                                st.error(f"Error loading log: {e}")
                        
                        # Model actions
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button(f"📊 Load for Testing", key=f"load_{species}"):
                                st.session_state.selected_test_model = species
                                st.success(f"✅ {species} model loaded for testing")
                        
                        with col2:
                            if st.button(f"🗑️ Delete Model", key=f"delete_{species}"):
                                try:
                                    import shutil
                                    shutil.rmtree(model_dir)
                                    st.success(f"✅ {species} model deleted")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Error deleting model: {e}")
                    else:
                        st.warning("⚠️ Model training in progress or failed")
        else:
            st.info("📭 No trained models found")
    else:
        st.info("📭 No models directory found")

def main():
    st.title("🧬 Real Trainable Mycorrhizal AI System")
    st.markdown("### Train with 25 images + masks → Achieve 80% accuracy")
    
    # System status
    col1, col2, col3 = st.columns(3)
    with col1:
        st.success("✅ Real Training Ready")
    with col2:
        st.info(f"🔧 Device: {DEVICE}")
    with col3:
        st.info("🎯 U-Net Architecture")
    
    # Main tabs - ADDED DEBUG TAB
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📚 Training Data Setup",
        "🚀 Train New Model", 
        "🔍 Test Trained Model",
        "📊 Model Management",
        "🔧 Debug Files"  # NEW DEBUG TAB
    ])
    
    with tab1:
        training_data_tab()
    
    with tab2:
        train_model_tab()
    
    with tab3:
        test_model_tab()
    
    with tab4:
        model_management_tab()
    
    with tab5:  # NEW DEBUG TAB
        debug_file_matching()

# This should be at the VERY END
if __name__ == "__main__":
    main()

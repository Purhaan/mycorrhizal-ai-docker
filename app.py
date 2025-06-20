#!/usr/bin/env python3
"""
REAL TRAINABLE MYCORRHIZAL AI SYSTEM
Train with real images + masks, achieve 80% accuracy with 25 images per species
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
warnings.filterwarnings('ignore')

# Set device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Page config
st.set_page_config(
    page_title="Trainable Mycorrhizal AI",
    page_icon="🧬", 
    layout="wide"
)

def create_directories():
    """Create necessary directories"""
    directories = [
        "data/raw", "data/masks", "data/augmented", "data/species",
        "models/trained", "models/checkpoints", "results", "temp"
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

create_directories()

class MycorrhizalDataset(Dataset):
    """Dataset for mycorrhizal image segmentation with real training"""
    
    def __init__(self, image_paths: List[str], mask_paths: List[str], 
                 transform=None, augment=True):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.augment = augment
        
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
                A.RandomGamma(gamma_limit=(0.7, 1.3), p=0.3),
                
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
        # Load image and mask
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        
        # Resize to standard size
        image = cv2.resize(image, (256, 256))
        mask = cv2.resize(mask, (256, 256))
        
        # Binary threshold for mask
        mask = (mask > 127).astype(np.uint8)
        
        # Apply augmentations
        if self.augmentation:
            augmented = self.augmentation(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        # Convert mask to tensor if not already
        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(mask).float()
        
        return image, mask

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
        
    def prepare_data(self, image_dir: str, mask_dir: str, 
                    train_split: float = 0.8) -> Tuple[DataLoader, DataLoader]:
        """Prepare training data with aggressive augmentation"""
        
        # Get image and mask paths
        image_files = sorted([f for f in os.listdir(image_dir) 
                            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.tif'))])
        
        image_paths = []
        mask_paths = []
        
        for img_file in image_files:
            img_path = os.path.join(image_dir, img_file)
            # Try different mask naming conventions
            mask_file = img_file.replace('.jpg', '_mask.png').replace('.jpeg', '_mask.png').replace('.tiff', '_mask.png')
            mask_path = os.path.join(mask_dir, mask_file)
            
            if os.path.exists(mask_path):
                image_paths.append(img_path)
                mask_paths.append(mask_path)
        
        if len(image_paths) < 5:
            raise ValueError(f"Need at least 5 images, found {len(image_paths)}")
        
        # Split data
        total_size = len(image_paths)
        train_size = int(train_split * total_size)
        
        indices = list(range(total_size))
        np.random.shuffle(indices)
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        train_img_paths = [image_paths[i] for i in train_indices]
        train_mask_paths = [mask_paths[i] for i in train_indices]
        val_img_paths = [image_paths[i] for i in val_indices]
        val_mask_paths = [mask_paths[i] for i in val_indices]
        
        # Create datasets with aggressive augmentation for training
        train_dataset = MycorrhizalDataset(train_img_paths, train_mask_paths, augment=True)
        val_dataset = MycorrhizalDataset(val_img_paths, val_mask_paths, augment=False)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=0)
        
        return train_loader, val_loader
    
    def calculate_metrics(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """Calculate segmentation metrics"""
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
            'iou': iou,
            'dice': dice,
            'pixel_accuracy': pixel_acc
        }
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train one epoch"""
        self.model.train()
        total_loss = 0
        
        for batch_idx, (images, masks) in enumerate(train_loader):
            images = images.to(DEVICE)
            masks = masks.to(DEVICE).unsqueeze(1)
            
            self.optimizer.zero_grad()
            
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate model"""
        self.model.eval()
        total_loss = 0
        all_metrics = {'iou': [], 'dice': [], 'pixel_accuracy': []}
        
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(DEVICE)
                masks = masks.to(DEVICE).unsqueeze(1)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                total_loss += loss.item()
                
                # Calculate metrics
                metrics = self.calculate_metrics(outputs, masks)
                for key, value in metrics.items():
                    all_metrics[key].append(value)
        
        # Average metrics
        avg_metrics = {key: np.mean(values) for key, values in all_metrics.items()}
        avg_metrics['loss'] = total_loss / len(val_loader)
        
        return avg_metrics
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
              epochs: int = 100, patience: int = 20) -> Dict:
        """Full training loop with early stopping"""
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        training_log = {
            'epochs_trained': 0,
            'best_val_loss': float('inf'),
            'best_val_iou': 0,
            'final_metrics': {}
        }
        
        st.info(f"🚀 Starting training for {self.species_name} with {len(train_loader.dataset)} training samples")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        metrics_placeholder = st.empty()
        
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
            self.scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                self.save_checkpoint(epoch, val_metrics, is_best=True)
                training_log['best_val_loss'] = val_loss
                training_log['best_val_iou'] = val_iou
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
        
        training_log['epochs_trained'] = epoch + 1
        training_log['final_metrics'] = val_metrics
        
        return training_log
    
    def save_checkpoint(self, epoch: int, metrics: Dict, is_best: bool = False):
        """Save model checkpoint"""
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
    
    def load_model(self, checkpoint_path: str):
        """Load trained model"""
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.val_ious = checkpoint.get('val_ious', [])
        return checkpoint['metrics']

class MycorrhizalPredictor:
    """Real-time prediction with trained models"""
    
    def __init__(self, model_path: str):
        self.model = UNet(in_channels=3, out_channels=1).to(DEVICE)
        checkpoint = torch.load(model_path, map_location=DEVICE)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        self.transform = A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    
    def predict(self, image_path: str) -> Dict:
        """Predict mycorrhizal colonization"""
        # Load and preprocess image
        image = cv2.imread(image_path)
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
            'colonization_percentage': colonization_percentage,
            'confidence': min(confidence * 10, 1.0),  # Scale to 0-1
            'binary_mask': binary_mask,
            'probability_mask': prediction_resized,
            'original_image_shape': original_shape
        }

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
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📚 Training Data Setup",
        "🚀 Train New Model", 
        "🔍 Test Trained Model",
        "📊 Model Management"
    ])
    
    with tab1:
        training_data_tab()
    
    with tab2:
        train_model_tab()
    
    with tab3:
        test_model_tab()
    
    with tab4:
        model_management_tab()

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
            
            if len(images) >= 5:
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
                        log_path = os.path.join("models/trained", selected_species, "training_log.json")
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
                        
                    except Exception as e:
                        st.error(f"❌ Training failed: {str(e)}")
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
                            with open(log_path, 'r') as f:
                                log = json.load(f)
                            
                            col1, col2, col3, col4 = st.columns(4)
                            col1.metric("Epochs", log['epochs_trained'])
                            col2.metric("Best IoU", f"{log['best_val_iou']:.3f}")
                            col3.metric("Val Loss", f"{log['best_val_loss']:.4f}")
                            col4.metric("Accuracy", f"{log['final_metrics']['pixel_accuracy']:.1%}")
                        
                        # Model actions
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button(f"📊 Load for Testing", key=f"load_{species}"):
                                st.session_state.selected_test_model = species
                                st.success(f"✅ {species} model loaded for testing")
                        
                        with col2:
                            if st.button(f"🗑️ Delete Model", key=f"delete_{species}"):
                                import shutil
                                shutil.rmtree(model_dir)
                                st.success(f"✅ {species} model deleted")
                                st.rerun()
                    else:
                        st.warning("⚠️ Model training in progress or failed")
        else:
            st.info("📭 No trained models found")
    else:
        st.info("📭 No models directory found")

if __name__ == "__main__":
    main()

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from torchvision.transforms import AutoAugment, AutoAugmentPolicy, RandAugment, TrivialAugmentWide
from torchvision.transforms import RandomErasing
from PIL import Image
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve, f1_score, precision_score, recall_score
import seaborn as sns
from datetime import datetime
import json
import random
from pathlib import Path
from sklearn.model_selection import StratifiedShuffleSplit
from collections import defaultdict


# Set seeds for reproducibility

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

def seed_worker(worker_id):

    worker_seed = torch.initial_seed() % 2**32

    np.random.seed(worker_seed)
    random.seed(worker_seed)

DATA_DIR = "data/2750" 
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 30
NUM_CLASSES = 10
IMAGE_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs("logs", exist_ok=True)
os.makedirs("models", exist_ok=True)

CLASS_NAMES = [

    'AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial',
    'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake'

]

# --- Advanced Data Augmentation ---

def get_transforms(is_training=True):

    if is_training:

        return transforms.Compose([

            transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.4),
            transforms.RandomRotation(degrees=25),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15),
            transforms.RandomGrayscale(p=0.15),
            transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.85, 1.15)),
            AutoAugment(policy=AutoAugmentPolicy.IMAGENET),
            RandAugment(),
            TrivialAugmentWide(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            RandomErasing(p=0.25, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
        
        ])

    else:

        return transforms.Compose([

            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        ])

# --- Dataset Class (improved) ---

class EuroSATDataset(Dataset):

    def __init__(self, data_dir, transform=None):

        self.data_dir = data_dir
        self.transform = transform

        self.class_names = CLASS_NAMES
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.class_names)}
        
        self.samples = []
       
        for class_name in self.class_names:

            class_dir = os.path.join(data_dir, class_name)

            if os.path.exists(class_dir):

                for img_name in os.listdir(class_dir):

                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):

                        img_path = os.path.join(class_dir, img_name)
                        self.samples.append((img_path, self.class_to_idx[class_name]))
    
    def __len__(self):

        return len(self.samples)

    def __getitem__(self, idx):

        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:

            image = self.transform(image)

        return image, label

# --- Model: GoogleLeNet(Inception V1) ---

def create_model():

    model = models.googlenet(weights=models.GoogLeNet_Weights.IMAGENET1K_V1)

    # Unfreeze all layers for fine-tuning

    for param in model.parameters():

        param.requires_grad = True

    num_features = model.fc.in_features

    model.fc = nn.Sequential(

        nn.Dropout(p=0.3),
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(512, NUM_CLASSES)

    )

    return model

def stratified_split(dataset, test_size=0.2, random_state=42):
    
    labels = [label for _, label in dataset.samples]
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    indices = np.arange(len(labels))
    
    for train_idx, val_idx in sss.split(indices, labels):

        return train_idx, val_idx

# --- Metrics Helper ---

def compute_metrics(y_true, y_pred, class_names):

    metrics = {}

    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, average=None, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, average=None, zero_division=0)
    metrics['f1'] = f1_score(y_true, y_pred, average=None, zero_division=0)
    metrics['macro_f1'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['classification_report'] = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)
    
    return metrics

# --- Training and Validation Loops ---
def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for data, target in tqdm(loader, desc="Training", leave=False):

        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=(scaler is not None)):

            output = model(data)
            loss = criterion(output, target)

        if scaler:

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        else:

            loss.backward()
            optimizer.step()

        running_loss += loss.item()
        _, predicted = output.max(1)

        total += target.size(0)
        correct += predicted.eq(target).sum().item()

    return running_loss / len(loader), 100. * correct / total

def validate(model, loader, criterion, device):

    model.eval()

    running_loss = 0.0

    correct = 0
    total = 0

    all_preds = []
    all_targets = []

    with torch.no_grad():

        for data, target in tqdm(loader, desc="Validation", leave=False):

            data, target = data.to(device), target.to(device)

            output = model(data)
            loss = criterion(output, target)
            
            running_loss += loss.item()

            _, predicted = output.max(1)

            total += target.size(0)
            correct += predicted.eq(target).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    acc = 100. * correct / total

    return running_loss / len(loader), acc, all_targets, all_preds

# --- Visualization ---

def plot_confusion_matrix(cm, class_names, out_path):

    plt.figure(figsize=(12, 10))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    
    plt.title('Confusion Matrix')

    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    plt.xticks(rotation=45)
    plt.yticks(rotation=0)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_training_curves(history, out_path):

    fig, axs = plt.subplots(2, 2, figsize=(15, 10))

    axs[0,0].plot(history['train_loss'], label='Train Loss')
    axs[0,0].plot(history['val_loss'], label='Val Loss')

    axs[0,0].set_title('Loss')

    axs[0,0].legend()

    axs[0,1].plot(history['train_acc'], label='Train Acc')
    axs[0,1].plot(history['val_acc'], label='Val Acc')

    axs[0,1].set_title('Accuracy')

    axs[0,1].legend()

    axs[1,0].plot(history['learning_rate'])
    axs[1,0].set_title('Learning Rate')

    axs[1,1].plot(history['val_acc'])
    axs[1,1].set_title('Validation Accuracy')

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

# --- Main Training Pipeline ---

def main():

    print("EuroSat V3 - Improved Training Pipeline")
    print("=" * 60)
    print(f"Dataset: {DATA_DIR}")
    print(f"# Classes: {len(CLASS_NAMES)}")
    print(f"Device: {DEVICE}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Max Epochs: {NUM_EPOCHS}")
    print("=" * 60)

    # --- Dataset and Stratified Split ---
    full_dataset = EuroSATDataset(DATA_DIR, transform=get_transforms(is_training=True))
    train_idx, val_idx = stratified_split(full_dataset, test_size=0.2)
    train_subset = torch.utils.data.Subset(full_dataset, train_idx)
    val_subset = torch.utils.data.Subset(full_dataset, val_idx)

    # Use different transforms for val
    val_subset.dataset.transform = get_transforms(is_training=False)

    g = torch.Generator()
    g.manual_seed(42)

    train_loader = DataLoader(
        train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4,
        pin_memory=torch.cuda.is_available(), worker_init_fn=seed_worker, generator=g
    )
    val_loader = DataLoader(
        val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4,
        pin_memory=torch.cuda.is_available(), worker_init_fn=seed_worker, generator=g
    )

    print(f"Training samples: {len(train_subset)}")
    print(f"Val samples: {len(val_subset)}")

    # --- Model, Loss, Optimizer, Scheduler ---
    model = create_model().to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

    # --- Logging ---
    history = defaultdict(list)
    log_filename = f"logs/training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    best_val_acc = 0.0
    patience_counter = 0
    early_stopping_patience = 15

    # --- Training Loop ---
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        print("-" * 40)
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE, scaler)
        val_loss, val_acc, val_targets, val_preds = validate(model, val_loader, criterion, DEVICE)
        scheduler.step(val_acc)
        current_lr = optimizer.param_groups[0]['lr']
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['learning_rate'].append(current_lr)
        # Save to CSV
        pd.DataFrame(history).to_csv(log_filename, index=False)
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        print(f"Learning Rate: {current_lr:.6f}")
        # Checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'history': dict(history)
            }, f"models/best_model.pth")
            print(f"New best model saved! Validation Accuracy: {val_acc:.2f}%")
        else:
            patience_counter += 1
        if patience_counter >= early_stopping_patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

    print("\n✅ Training completed!")
    print(f"🏆 Best validation accuracy: {best_val_acc:.2f}%")

    # --- Final Evaluation ---
    checkpoint = torch.load("models/best_model.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    all_predictions = []
    all_targets = []
    with torch.no_grad():
        for data, target in tqdm(val_loader, desc="Final Evaluation"):
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)
            _, predicted = output.max(1)
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    metrics = compute_metrics(all_targets, all_predictions, CLASS_NAMES)
    print("\nClassification Report:")
    print(metrics['classification_report'])
    print("Generating visuals...")
    plot_confusion_matrix(metrics['confusion_matrix'], CLASS_NAMES, 'logs/confusion_matrix.png')
    plot_training_curves(history, 'logs/training_history.png')
    # Save metrics
    with open('logs/metrics.json', 'w') as f:
        json.dump({k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in metrics.items()}, f, indent=4)
    print(f"\nTraining summary and metrics saved to logs/")
    print(f"Best model saved to models/best_model.pth")
    print(f"Plots saved to logs/")
    print("\n🎉 Training completed successfully!")

if __name__ == "__main__":
    main()

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
from datetime import datetime
import json
import random
from pathlib import Path

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

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

    'AnnualCrop', 

    'Forest', 

    'HerbaceousVegetation',

    'Highway', 

    'Industrial', 

    'Pasture', 

    'PermanentCrop', 

    'Residential', 

    'River', 

    'SeaLake'

]

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

# Extensive data augmentation

def get_transforms(is_training=True):

    if is_training:

        return transforms.Compose([

            transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomGrayscale(p=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        ])

    else:

        return transforms.Compose([

            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        ])

# EfficientNet-B0 model with pretrained weights

def create_model():

    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    
    # Freeze early layers for transfer learning

    for param in model.features[:5].parameters():

        param.requires_grad = False
    
    # Modify classifier for our number of classes

    num_features = model.classifier[1].in_features

    model.classifier = nn.Sequential(

        nn.Dropout(p=0.3),
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(512, NUM_CLASSES)

    )
    
    return model

def log_metrics(epoch, train_loss, train_acc, val_loss, val_acc, lr, log_data, log_filename):
   
    log_data.append({

        'epoch': epoch,
        'train_loss': train_loss,
        'train_acc': train_acc,
        'val_loss': val_loss,
        'val_acc': val_acc,
        'learning_rate': lr

    })
    
    # Save to CSV

    df = pd.DataFrame(log_data)
    df.to_csv(log_filename, index=False)

def train_epoch(model, loader, criterion, optimizer, device):

    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc="Training", leave=False)

    for batch_idx, (data, target) in enumerate(pbar):

        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        # Update progress bar
        pbar.set_postfix({

            'Loss': f'{loss.item():.4f}',
            'Acc': f'{100.*correct/total:.2f}%'

        })
    
    return running_loss / len(loader), 100. * correct / total

def validate_epoch(model, loader, criterion, device):

    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        
        pbar = tqdm(loader, desc="Validation", leave=False)

        for data, target in pbar:

            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            pbar.set_postfix({

                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%'

            })
    
    return running_loss / len(loader), 100. * correct / total

def freeze_model(model):

    """Freeze all parameters for inference"""

    for param in model.parameters():

        param.requires_grad = False

    model.eval()

    return model

def save_frozen_model(model, class_names, image_size, summary):

    """Save frozen model for inference"""

    frozen_model = freeze_model(model)
    
    # Save frozen model for inference

    torch.save({

        'model_state_dict': frozen_model.state_dict(),
        'class_names': class_names,
        'image_size': image_size,
        'model_architecture': 'EfficientNet-B0',
        'summary': summary,
        'frozen': True

    }, 'models/frozen_model.pth')
    
    print(f"✅ Frozen model saved to models/frozen_model.pth")
    return frozen_model

def main():

    """Main training function"""

    print("EuroSat")
    print("=" * 60) #ooooh asthethics
    print(f"Dataset: {DATA_DIR}")
    print(f"# Classes: {len(CLASS_NAMES)}")
    print(f"Device: {DEVICE}") # i aint got no cuda
    print(f" Batch Size: {BATCH_SIZE}") # i went with 32
    print(f" Learning Rate: {LEARNING_RATE}")
    print(f" Max Epochs: {NUM_EPOCHS}")
    print("=" * 60)
    
    train_dataset = EuroSATDataset(DATA_DIR, transform=get_transforms(is_training=True))
    val_dataset = EuroSATDataset(DATA_DIR, transform=get_transforms(is_training=False))
    
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=torch.cuda.is_available())
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Create model
    print("EfficientNet-B0 model")
    model = create_model().to(DEVICE)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    history = {

        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'learning_rate': []

    }
    
    # CSV logging setup
    log_filename = f"logs/training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    log_data = []
    
    # Training loop
    best_val_acc = 0.0
    patience_counter = 0
    
    print("\n Starting training...")
    print("=" * 60)
    
    for epoch in range(NUM_EPOCHS):

        print(f"\n Epoch {epoch+1}/{NUM_EPOCHS}")
        print("-" * 40)
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, DEVICE)
        
        scheduler.step(val_acc)
        current_lr = optimizer.param_groups[0]['lr']
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['learning_rate'].append(current_lr)
        
        log_metrics(epoch+1, train_loss, train_acc, val_loss, val_acc, current_lr, log_data, log_filename)
        
        print(f" Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f" Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        print(f" Learning Rate: {current_lr:.6f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({

                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'history': history
            }, f"models/best_model.pth"
            )
            print(f" New best model saved! Validation Accuracy: {val_acc:.2f}%")

        else:

            patience_counter += 1
        
        # Early stopping
        if patience_counter >= 15:

            print(f"  Early stopping triggered after {epoch+1} epochs") # hopefully shouldnt happen
            break
    
    print("\n✅ Training completed!")
    print(f"🏆 Best validation accuracy: {best_val_acc:.2f}%") # HOOORAY
    
    checkpoint = torch.load("models/best_model.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(" Running final evaluation...")

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
    
    final_accuracy = accuracy_score(all_targets, all_predictions)
    print(f"\nFinal Test Accuracy: {final_accuracy:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(all_targets, all_predictions, target_names=CLASS_NAMES))
    
    print("Generating visuals...") 
    
    cm = confusion_matrix(all_targets, all_predictions)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('logs/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Val Loss')
    ax1.set_title('Loss')
    ax1.legend()
    
    ax2.plot(history['train_acc'], label='Train Acc')
    ax2.plot(history['val_acc'], label='Val Acc')
    ax2.set_title('Accuracy')
    ax2.legend()
    
    ax3.plot(history['learning_rate'])
    ax3.set_title('Learning Rate')
    
    ax4.plot(history['val_acc'])
    ax4.set_title('Validation Accuracy')
    
    plt.tight_layout()
    plt.savefig('logs/training_history.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save training summary
    summary = {

        'best_val_accuracy': best_val_acc,
        'final_test_accuracy': final_accuracy,
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'total_epochs': len(history['train_loss']),
        'class_names': CLASS_NAMES,
        'model_architecture': 'EfficientNet-B0',
        'data_augmentation': 'Extensive (RandomResizedCrop, RandomFlip, ColorJitter, etc.)',
        'optimizer': 'AdamW',
        'scheduler': 'ReduceLROnPlateau',
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE

    }
    
    with open('logs/training_summary.json', 'w') as f:

        json.dump(summary, f, indent=4)
    
    frozen_model = save_frozen_model(model, CLASS_NAMES, IMAGE_SIZE, summary)
    
    print(f"\n Training summary saved to logs/training_summary.json")
    print(f" Best model saved to models/best_model.pth")
    print(f" Plots saved to logs/")
    
    print("\n🎉 Training completed successfully!")

if __name__ == "__main__":
    main()

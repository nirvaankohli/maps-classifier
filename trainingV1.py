import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import json
from datetime import datetime
import random

# seeeeds

random.seed(42)
torch.manual_seed(42)
np.random.seed(42)

DATA_PATH = "data/2750"
BATCH = 32
LR = 0.001
EPOCHS = 30
N_CLASSES = 10
IMG_DIM = 224

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs("logs", exist_ok=True)
os.makedirs("models", exist_ok=True)

LABELS = [

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

class FolderImageDataset(Dataset):

    def __init__(self, root, transform=None):

        self.transform = transform
        self.samples = []
        self.targets = []
        self.label_map = {name: idx for idx, name in enumerate(LABELS)}

        for label in LABELS:

            folder = os.path.join(root, label)
            files = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(('jpg', 'jpeg', 'png'))]
            
            self.samples.extend(files)
            self.targets.extend([self.label_map[label]] * len(files))

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, idx):

        img = Image.open(self.samples[idx]).convert('RGB')

        if self.transform:

            img = self.transform(img)

        return img, self.targets[idx]

def build_transforms(train=True):
    
    if train:

        return transforms.Compose([

            transforms.RandomResizedCrop(IMG_DIM, scale=(0.85, 1.0)),

            transforms.RandomHorizontalFlip(p=0.4),

            transforms.RandomVerticalFlip(p=0.2),

            transforms.RandomRotation(12),
            
            transforms.ColorJitter(0.15, 0.15, 0.15, 0.07),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((IMG_DIM, IMG_DIM)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

def mobilenet(num_classes):
    net = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    for p in net.features[:3].parameters():
        p.requires_grad = False
    in_feat = net.classifier[1].in_features
    net.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(in_feat, num_classes)
    )
    return net

def run_epoch(model, loader, loss_fn, opt, device, train=True):
    if train:
        model.train()
    else:
        model.eval()
    total_loss = 0
    correct = 0
    total = 0
    loop = tqdm(loader, leave=False)
    for x, y in loop:
        x, y = x.to(device), y.to(device)
        if train:
            opt.zero_grad()
        with torch.set_grad_enabled(train):
            out = model(x)
            loss = loss_fn(out, y)
            if train:
                loss.backward()
                opt.step()
        total_loss += loss.item()
        preds = out.argmax(1)
        correct += (preds == y).sum().item()
        total += y.size(0)
        loop.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{100*correct/total:.2f}%"})
    return total_loss / len(loader), 100 * correct / total

def main():
    train_set = FolderImageDataset(DATA_PATH, build_transforms(True))
    val_set = FolderImageDataset(DATA_PATH, build_transforms(False))
    n_train = int(0.8 * len(train_set))
    n_val = len(train_set) - n_train
    train_set, val_set = torch.utils.data.random_split(train_set, [n_train, n_val])
    train_loader = DataLoader(train_set, batch_size=BATCH, shuffle=True, num_workers=4, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_set, batch_size=BATCH, shuffle=False, num_workers=4, pin_memory=torch.cuda.is_available())
    model = mobilenet(N_CLASSES).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    best_acc = 0
    patience = 0
    log = []
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}/{EPOCHS}")
        tr_loss, tr_acc = run_epoch(model, train_loader, loss_fn, optimizer, DEVICE, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, loss_fn, optimizer, DEVICE, train=False)
        scheduler.step(val_acc)
        log.append({
            "epoch": epoch+1,
            "train_loss": tr_loss,
            "train_acc": tr_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": optimizer.param_groups[0]['lr']
        })
        print(f"Train: {tr_loss:.4f} {tr_acc:.2f}% | Val: {val_loss:.4f} {val_acc:.2f}% | LR: {optimizer.param_groups[0]['lr']:.6f}")
        if val_acc > best_acc:
            best_acc = val_acc
            patience = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc
            }, "models/best_model.pth")
        else:
            patience += 1
        if patience >= 10:
            print("Early stopping")
            break
    print(f"Best validation accuracy: {best_acc:.2f}%")
    with open("logs/training_log.json", "w") as f:
        json.dump(log, f, indent=2)
    checkpoint = torch.load("models/best_model.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for x, y in tqdm(val_loader, desc="Eval", leave=False):
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            preds.extend(out.argmax(1).cpu().numpy())
            targets.extend(y.cpu().numpy())
    acc = accuracy_score(targets, preds)
    print(f"Final accuracy: {acc*100:.2f}%")
    print(classification_report(targets, preds, target_names=LABELS))
    cm = confusion_matrix(targets, preds)
    print("Confusion matrix:")
    print(cm)

if __name__ == "__main__":
    main()

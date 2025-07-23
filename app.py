import streamlit as st
import torch
from torchvision import transforms, models
from PIL import Image
import numpy as np
import requests
import io
import cv2
import os
from typing import Tuple
from streamlit_folium import st_folium
import folium
import pandas as pd


st.set_page_config(page_title="Satellite Map Tile Classifier", layout="wide")

st.markdown("""
<style>
    section[data-testid="stSidebar"] .css-ng1t4o {
        background: linear-gradient(180deg, #e0e7ff 0%, #f1f5f9 100%);
    }
    section[data-testid="stSidebar"] .css-1v0mbdj {
        background: none;
    }
    .sidebar-title {
        font-size:1.6rem;
        font-weight:700;
        color:#3d7bff;
        margin-bottom:0.5rem;
    }
    .sidebar-sub {
        color:#e0e0e0;
        font-size:1.1rem;
        margin-bottom:1.2rem;
    }
    .sidebar-link {
        color:#2563eb;
        text-decoration:none;
        font-weight:500;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2) MODEL & TILE UTILS
# -----------------------------------------------------------------------------
CLASS_NAMES = [
    'AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial',
    'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake'
]
IMAGE_SIZE = 224
MODEL_PATH = 'V2_best_model.pth'
TILE_SIZE_METERS = 640
TILE_ZOOM = 15

def get_transforms():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

def create_model():
    model = models.efficientnet_b0(weights=None)
    num_features = model.classifier[1].in_features
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.3),
        torch.nn.Linear(num_features, 512),
        torch.nn.ReLU(),
        torch.nn.Dropout(p=0.2),
        torch.nn.Linear(512, len(CLASS_NAMES))
    )
    return model

def load_model():
    model = create_model()
    checkpoint = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()
        self.hook_handles.append(
            self.target_layer.register_forward_hook(forward_hook))
        self.hook_handles.append(
            self.target_layer.register_backward_hook(backward_hook))

    def __call__(self, input_tensor, class_idx=None):
        self.model.zero_grad()
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        loss = output[0, class_idx]
        loss.backward()
        gradients = self.gradients[0]
        activations = self.activations[0]
        weights = gradients.mean(dim=(1, 2), keepdim=True)
        cam = (weights * activations).sum(0)
        cam = torch.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        cam = cam.cpu().numpy()
        cam = cv2.resize(cam, (IMAGE_SIZE, IMAGE_SIZE))
        return cam

    def remove_hooks(self):
        for handle in self.hook_handles:
            handle.remove()

def latlng_to_tilexy(lat, lng, zoom):
    lat_rad = np.radians(lat)
    n = 2 ** zoom
    x = int((lng + 180.0) / 360.0 * n)
    y = int((1.0 - np.log(np.tan(lat_rad) + 1/np.cos(lat_rad)) / np.pi) / 2.0 * n)
    return x, y

def get_tile_url(lat, lng, zoom=TILE_ZOOM):
    x, y = latlng_to_tilexy(lat, lng, zoom)
    return f"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{zoom}/{y}/{x}"

def fetch_tile_image(lat, lng):
    url = get_tile_url(lat, lng)
    response = requests.get(url)
    img = Image.open(io.BytesIO(response.content)).convert('RGB')
    return img

def get_rectangle_bounds(lat, lng):
    latDelta = TILE_SIZE_METERS / 111000
    lngDelta = TILE_SIZE_METERS / (111000 * np.cos(np.radians(lat)))
    return [
        [lat - latDelta/2, lng - lngDelta/2],
        [lat + latDelta/2, lng + lngDelta/2]
    ]


if 'page' not in st.session_state:
    st.session_state.page = 'home'

# Callbacks
def show_home():
    st.session_state.page = 'home'
def show_paper():
    st.session_state.page = 'paper'

colA, colB = st.columns([1, 1])
with colA:
    st.button("🏠 Home", on_click=show_home)
with colB:
    st.button("📖 Behind the Scenes (READ PLS)", on_click=show_paper)
st.markdown("---")


if st.session_state.page == 'home':
    # Sidebar
    st.sidebar.markdown(
        "<div class='sidebar-title'>Satellite Map Tile Classifier</div>",
        unsafe_allow_html=True)
    st.sidebar.markdown("""
    <div class='sidebar-sub'>
    <b>EuroSAT Model</b><br>
    10 land use classes<br>
    640m × 640m tiles<br>
    w/ Grad‑CAM & probability visualizations
    </div>
    """, unsafe_allow_html=True)
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "**How it works:**\n1. Select a tile on the map OR upload an image.\n"
        "2. The model predicts the land use class.\n"
        "3. Visualizations help you understand the prediction."
    )

    # Header banner
    st.markdown("""
    <div style='background: linear-gradient(90deg,#e0e7ff,#f1f5f9);
                border-radius: 1rem; padding: 1.5rem 2rem;
                margin-bottom:1.5rem; box-shadow: 0 2px 8px rgba(59,130,246,0.07);'>
      <h2 style='margin-bottom:0.5rem;'>🌍 <span style='color:black;'>
          Satellite Map Tile Classifier</span></h2>
      <div style='font-size:1.1rem; color:#334155;'>
        <b>Choose <span style='color:#2563eb;'>either</span>:</b>
        <ul style='margin-top:0.5rem;'>
          <li><b>Click on the map</b> to select a 640m × 640m satellite tile</li>
          <li><b>Upload a satellite image</b> (EuroSAT format)</li>
        </ul>
        <span style='color:#64748b;'>The model will predict the land use
        class and show you <b>why</b> it made that decision, with interactive
        visualizations.</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    empire_lat, empire_lng = 40.748817, -73.985428
    if 'tile_lat' not in st.session_state:
        st.session_state['tile_lat'] = empire_lat
    if 'tile_lng' not in st.session_state:
        st.session_state['tile_lng'] = empire_lng

    # Two columns: map selector & upload
    col1, col2 = st.columns([5, 3], gap="large")
    with col1:
        st.subheader("Select a Tile on the Map")
        lat = st.session_state['tile_lat']
        lng = st.session_state['tile_lng']
        m = folium.Map(location=[lat, lng], zoom_start=TILE_ZOOM, tiles=None)
        folium.TileLayer(
            tiles='https://server.arcgisonline.com/ArcGIS/rest/services/'
                  'World_Imagery/MapServer/tile/{z}/{y}/{x}',
            attr='Tiles © Esri — Source: Esri, USGS, USDA',
            overlay=True
        ).add_to(m)
        bounds = get_rectangle_bounds(lat, lng)
        folium.Rectangle(
            bounds=bounds,
            color='#2563eb',
            weight=4,
            fill=True,
            fill_opacity=0.35
        ).add_to(m)
        map_data = st_folium(
            m, height=600, width=None, returned_objects=["last_clicked"])
        if map_data and map_data.get("last_clicked"):
            lat = map_data["last_clicked"]["lat"]
            lng = map_data["last_clicked"]["lng"]
            st.session_state['tile_lat'] = lat
            st.session_state['tile_lng'] = lng
            st.success(f"Selected coordinates: {lat:.6f}, {lng:.6f}")

    with col2:
        st.subheader("Upload Image (optional)")
        uploaded_file = st.file_uploader(
            "Upload a satellite tile image", type=["png", "jpg", "jpeg"])
        use_upload = uploaded_file is not None
        if use_upload:
            uploaded_img = Image.open(uploaded_file).convert('RGB')
            st.image(uploaded_img, caption="Uploaded Image",
                     use_container_width=True)

        st.markdown(
            "<div style='margin:1.5rem 0 0.5rem 0;"
            "border-top:1px solid #e2e8f0;'></div>",
            unsafe_allow_html=True
        )
        st.markdown("Tile Preview (640m × 640m)")
        with st.spinner("Fetching tile image..."):
            preview_img = fetch_tile_image(
                st.session_state['tile_lat'],
                st.session_state['tile_lng']
            )
        st.image(
            preview_img,
            caption=f"Selected Tile: ({st.session_state['tile_lat']:.6f}, "
                    f"{st.session_state['tile_lng']:.6f})",
            use_container_width=True
        )

    img = uploaded_img if use_upload else preview_img
    if img is not None:
        st.markdown("""
        <div style='margin-top:2rem; margin-bottom:1rem;
                    font-size:1.3rem; font-weight:600; color:#2563eb;'>
          🔎 Model Prediction & Visual Explanations
        </div>
        """, unsafe_allow_html=True)
        with st.spinner("Running model and Grad‑CAM..."):
            model = load_model()
            transform = get_transforms()
            input_tensor = transform(img).unsqueeze(0)
            with torch.no_grad():
                logits = model(input_tensor)
                probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
            top5_idx = probs.argsort()[-5:][::-1]
            top1_idx = top5_idx[0]
            top2_idx = top5_idx[1]
            gradcam = GradCAM(model, model.features[-1])
            cam = gradcam(input_tensor, class_idx=top1_idx)
            gradcam.remove_hooks()

            img_np = np.array(img.resize((IMAGE_SIZE, IMAGE_SIZE))) / 255.0
            heatmap = cv2.applyColorMap(
                np.uint8(255 * cam), cv2.COLORMAP_JET
            )
            heatmap = np.float32(heatmap) / 255
            overlay = heatmap[..., ::-1] * 0.5 + img_np * 0.5
            overlay = np.clip(overlay, 0, 1)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric(
                "Top‑1 Prediction",
                CLASS_NAMES[top1_idx],
                f"{probs[top1_idx]*100:.2f}%"
            )
        with c2:
            st.metric(
                "2nd Guess",
                CLASS_NAMES[top2_idx],
                f"{probs[top2_idx]*100:.2f}%"
            )
        with c3:
            st.metric(
                "Top‑1 Confidence",
                f"{probs[top1_idx]*100:.2f}%"
            )

        st.divider()
        st.markdown("#### Top‑5 Predictions")
        df = pd.DataFrame({
            "Class": [CLASS_NAMES[i] for i in top5_idx],
            "Probability (%)": [f"{probs[i]*100:.2f}" for i in top5_idx]
        })
        st.dataframe(df, hide_index=True, use_container_width=True)

        st.divider()
        st.markdown("#### Class Probabilities (All)")
        chart_df = pd.DataFrame({
            "Class": CLASS_NAMES,
            "Probability": probs
        }).sort_values("Probability", ascending=False)
        st.bar_chart(chart_df.set_index("Class"))

        st.divider()
        st.markdown("#### Grad‑CAM Visualization")
        st.image(overlay, caption="Grad‑CAM Overlay",
                 use_container_width=True)


elif st.session_state.page == 'paper':

    st.header("📖 Behind the Scenes (READ PLS)")

    st.markdown("""
    
    # EuroSAT Training Pipeline Comparison: V1 vs V2 vs V3

This document provides a deep-dive comparison of the three major versions of the EuroSAT land cover classification training pipeline in this project. It explains the code logic, behind-the-scenes improvements, and visualizations, referencing upgrade notes, logs, and images. The goal is to help you understand not just what changed, but why, and how each version works under the hood.

---

## Table of Contents
- [EuroSAT Training Pipeline Comparison: V1 vs V2 vs V3](#eurosat-training-pipeline-comparison-v1-vs-v2-vs-v3)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [V1: MobileNetV2 Baseline](#v1-mobilenetv2-baseline)
    - [Code Highlights](#code-highlights)
    - [Result](#result)
  - [V2: EfficientNet-B0 Upgrade](#v2-efficientnet-b0-upgrade)
    - [Code Highlights](#code-highlights-1)
    - [Result](#result-1)
  - [V3: GoogleLeNet + Advanced Augmentation (Planned)](#v3-googlelenet--advanced-augmentation-planned)
    - [Code Highlights](#code-highlights-2)
    - [Status](#status)
  - [Visualizations \& Metrics](#visualizations--metrics)
    - [V2 Training History](#v2-training-history)
    - [V2 Confusion Matrix](#v2-confusion-matrix)
    - [V1 and V2 Logs](#v1-and-v2-logs)
  - [Summary Table](#summary-table)
  - [References](#references)

---

## Overview

The EuroSAT project aims to classify satellite images into 10 land cover classes. Over three major versions, the pipeline has evolved in model architecture, data augmentation, evaluation, and code structure. Each version builds on the lessons and results of the previous one.

---

## V1: MobileNetV2 Baseline
- **Model:** MobileNetV2 (pretrained, classifier head replaced)
- **Data Augmentation:** Moderate (random crops, flips, rotation, color jitter)
- **Dataset Handling:** Custom `FolderImageDataset` with manual label mapping
- **Training Logic:**
  - 80/20 random split for train/val
  - AdamW optimizer, ReduceLROnPlateau scheduler
  - Early stopping (patience=10)
  - Logging to JSON (logging was buggy)
  - Best model checkpointing
- **EuroSAT Adaptation:** 10 classes, resized to 224x224, ImageNet normalization

### Code Highlights
- **Simple, readable structure**: All logic in one file, with clear dataset and model classes.
- **Augmentation**: Uses `RandomResizedCrop`, horizontal/vertical flips, rotation, and color jitter.
- **Model**: Loads MobileNetV2, freezes early layers, replaces classifier for 10 classes.
- **Training**: Loops over epochs, tracks best validation accuracy, saves best model.

### Result
- **Best Validation Accuracy:** ~96.78% ([see V1_log.csv](#references))
- **Limitations:**
  - Moderate augmentation, may overfit
  - Logging not robust
  - No visualizations

---

## V2: EfficientNet-B0 Upgrade

- **Model:** EfficientNet-B0 (pretrained, early layers frozen, custom classifier head)
- **Data Augmentation:** Extensive (random resized crop, flips, rotation, color jitter, grayscale, affine)
- **Dataset Handling:** Custom `EuroSATDataset` with class-to-index mapping
- **Training Logic:**
  - 80/20 random split for train/val
  - AdamW optimizer, ReduceLROnPlateau scheduler
  - Early stopping (patience=15)
  - Logging to CSV (epoch-by-epoch)
  - Best model checkpointing
- **EuroSAT Adaptation:** 10 classes, resized to 224x224, ImageNet normalization
- **Visualization:**
  - Confusion matrix and training history plots
  - Saves a detailed training summary as JSON

### Code Highlights
- **Model**: EfficientNet-B0, with a larger, more regularized classifier head (Dropout, ReLU, Linear).
- **Augmentation**: Much more aggressive, including `RandomAffine`, `RandomGrayscale`, and stronger jitter.
- **Logging**: Metrics logged to CSV for easy analysis; summary and plots saved for reproducibility.
- **Visualization**: Generates and saves confusion matrix and training curves.

### Result
- **Best Validation Accuracy:** 97.39% ([see V2_log.csv](#references))
- **Visualizations:**
  - ![V2 Training History](V2_training_history.png)
  - ![V2 Confusion Matrix](V2_confusion_matrix.png)
- **Improvements:**
  - Stronger augmentation = better generalization
  - More robust logging and visualization
  - Modular code, easier to maintain

---

## V3: GoogleLeNet + Advanced Augmentation (Planned)

- **Model:** GoogleLeNet (Inception V1, pretrained, all layers unfrozen, custom classifier head)
- **Data Augmentation:** Very extensive (adds AutoAugment, RandAugment, TrivialAugmentWide, RandomErasing)
- **Dataset Handling:** Stratified split (ensures class balance in train/val)
- **Training Logic:**
  - Mixed-precision (AMP) support for faster training
  - Deterministic seeding for full reproducibility
  - Expanded metrics: accuracy, precision, recall, F1, macro F1, confusion matrix, classification report
  - Logging to CSV and JSON (all metrics)
  - Improved visualizations
- **EuroSAT Adaptation:** 10 classes, resized to 224x224, ImageNet normalization

### Code Highlights
- **Model**: GoogleLeNet, all layers trainable, custom classifier head for EuroSAT.
- **Augmentation**: Adds state-of-the-art policies (AutoAugment, RandAugment, etc.) for maximum diversity.
- **Split**: Uses `StratifiedShuffleSplit` for balanced validation.
- **Metrics**: Computes and logs per-class precision, recall, F1, macro F1, and full classification report.
- **Visualization**: Improved confusion matrix and training curves.

### Status
- **Training not yet completed** (see [V2toV3.md](#references) for planned upgrades and expected results)
- **Expected Accuracy:** >98%

---

## Visualizations & Metrics

### V2 Training History
![V2 Training History](V2_training_history.png)

- **Top Left:** Loss curves (train vs val)
- **Top Right:** Accuracy curves (train vs val)
- **Bottom Left:** Learning rate schedule
- **Bottom Right:** Validation accuracy

### V2 Confusion Matrix
![V2 Confusion Matrix](V2_confusion_matrix.png)

- **Interpretation:**
  - Most predictions are correct (diagonal)
  - Minor confusion between similar classes (e.g., Pasture vs AnnualCrop)
  - No class is severely underperforming

### V1 and V2 Logs
- **V1:** Logging did not work, but best val accuracy was 96.78%
- **V2:** See [V2_log.csv](#references) for full epoch-by-epoch metrics

---

## Summary Table

| Aspect                | V1 (MobileNetV2)         | V2 (EfficientNet-B0)        | V3 (GoogleLeNet, planned)   |
|-----------------------|--------------------------|-----------------------------|-----------------------------|
| Model                 | MobileNetV2              | EfficientNet-B0             | GoogleLeNet (Inception V1)  |
| Augmentation          | Moderate                 | Extensive                   | Very extensive + AutoAug    |
| Dataset Split         | Random 80/20             | Random 80/20                | Stratified 80/20            |
| Metrics               | Accuracy                 | Accuracy, confusion matrix   | Accuracy, precision, recall, F1, macro F1, confusion matrix, classification report |
| Logging               | JSON (buggy)             | CSV, plots, JSON summary     | CSV, plots, JSON (all metrics) |
| Visualization         | None                     | Confusion matrix, history    | Improved confusion matrix, history |
| Model Saving          | Best only                | Best + frozen + metadata     | Best + all metrics + ready for deployment |
| Accuracy (Val)        | 96.78%                   | 97.39%                      | >98% (expected)             |

---

## References

- **Upgrade Notes:**

  - [V1 to V2](upgrade‑notes/V1toV2.md)

  - [V2 to V3](upgrade‑notes/V2toV3.md)

- **Logs:**

  - [V1_log.csv](V1_log.csv)

  - [V2_log.csv](V2_log.csv)

- **Visualizations:**

  - [V2_training_history.png](V2_training_history.png)

  - [V2_confusion_matrix.png](V2_confusion_matrix.png)

- **Code:**

  - [trainingV1.py](trainingV1.py)

  - [trainingV2.py](trainingV2.py)
  
  - [trainingV3.py](trainingV3.py)

    
    
    """)

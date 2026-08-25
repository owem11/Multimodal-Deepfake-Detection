# Bifurcated Multimodal Deepfake Detection Framework

An end-to-end deep learning architecture designed to detect cross-modal "half-fakes" (authentic video paired with AI-cloned audio) by combining spatial visual inspection, temporal sequence tracking, and frequency-domain acoustic verification.

Achieved **99.62% Accuracy** and an **AUC-ROC of 0.999** on the FakeAVCeleb dataset.

---

## 📌 Architecture Overview

The system processes video and audio streams independently before merging their representations through a late-fusion multi-layer perceptron (MLP) decision engine:

1. **Visual Pipeline (Spatial & Temporal):**
   * **Spatial Extraction:** 10 contiguous video frames are processed through **XceptionNet** utilizing Depthwise Separable Convolutions to spot facial boundary artifacts and pixel-blending irregularities.
   * **Temporal Tracking:** Extracted spatial feature vectors are passed into an **LSTM** network to detect temporal frame-to-frame jitter and unnatural facial dynamics over time.

2. **Acoustic Pipeline (Frequency-Domain Vision):**
   * **Signal Processing:** 16kHz raw audio tracks are processed via Short-Time Fourier Transform (STFT) into $64 \times 94$ 2D **Mel-Spectrograms**.
   * **Feature Extraction:** **ResNet-18** scans the spectrogram images using residual skip connections to identify high-frequency neural vocoder phase artifacts left by voice clones.

3. **Multimodal Late-Fusion Engine:**
   * Concatenates the 1024-dimensional visual and acoustic feature vectors.
   * Passes data through Dense layers with **Batch Normalization**, **ReLU** activations, and **Dropout (0.5)** to prevent overfitting.
   * Uses a **Sigmoid Classifier** to output a final binary probability score $P(\text{Fake})$.

---

## 📊 Performance & Evaluation

* **Accuracy:** 99.62%
* **AUC-ROC:** 0.999
* **Evaluation Dataset:** FakeAVCeleb v1.2 (4,216 forged vs. 96 authentic validation split)

---

## 📂 Repository Structure

```text
Multimodal-Deepfake-Detection/
├── notebooks/
│   ├── 01_spatial_visual_xception.ipynb     # Frame extraction & XceptionNet features
│   ├── 02_temporal_visual_lstm.ipynb        # LSTM temporal sequence modeling
│   ├── 03_acoustic_resnet18.ipynb           # Mel-spectrogram & ResNet-18 pipeline
│   ├── 04_multimodal_late_fusion.ipynb      # 1024-D concatenation & MLP engine
│   └── 05_evaluation_metrics.ipynb          # ROC-AUC curves & confusion matrix
├── models/
│   └── phase1-xception-weights.h5           # Pre-trained XceptionNet model weights
├── sample_data/                             # Sample media files for inference testing
├── .gitignore
├── requirements.txt
└── README.md
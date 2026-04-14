# Messy Mashup: Robust Music Genre Classification

## Overview

This project addresses the **Messy Mashup Kaggle competition**, which focuses on music genre classification under realistic and noisy conditions.

Unlike conventional datasets, the test data consists of **synthetic mashups** created by:

* Mixing instrument stems from different songs of the same genre
* Applying tempo synchronization
* Adding environmental noise

This introduces a **significant distribution shift** between training (clean stems) and testing (noisy mashups), making generalization the primary challenge.

---

## Problem Statement

Given an audio input, the task is to classify it into one of **10 music genres**, while being invariant to:

* Cross-song stem recombination
* Tempo variations
* Instrument balance differences
* Additive environmental noise

---

## Dataset

### Training Data

* 1000 songs across 10 genres
* Each song split into 4 stems:

  * `drums.wav`
  * `vocals.wav`
  * `bass.wav`
  * `other.wav`
* Total: 4000 audio files

### Noise Dataset

* ESC-50 dataset (2000 environmental audio clips)

### Test Data

* 3020 mashups with:

  * Mixed stems (same genre)
  * Tempo alignment
  * Noise injection

---

## Key Challenge: Distribution Shift

* **Train:** Clean, isolated stems
* **Test:** Mixed, noisy audio

The project emphasizes **closing this gap via data augmentation and preprocessing** rather than relying solely on model complexity.

---

## Methodology

### 1. Preprocessing

* Resampling to 22,050 Hz
* Fixed-length audio segments (30 seconds)
* Silence-aware cropping

---

### 2. Data Augmentation Pipeline

A custom augmentation pipeline was designed to simulate test conditions:

* **Stem Mixing:** Combine drums, vocals, bass, and other stems with random weights
* **Tempo Synchronization:** BPM estimation and time-stretching
* **Noise Injection:** ESC-50 noise with random SNR (0–15 dB)
* **Normalization:**

  * RMS normalization (~ -17 dBFS)
  * Peak normalization to prevent clipping

This pipeline was the **most critical component** in improving performance.

---

### 3. Feature Extraction

#### Log-Mel Spectrogram

* 128 Mel bins
* FFT size: 2048
* Hop length: 512
* Output shape: (128, 1292)

This representation captures both spectral and temporal information and is well-suited for CNN-based models.

---

## Models

### Classical Baselines

* XGBoost
* Random Forest

Features:

* MFCCs
* Chroma
* Spectral contrast
* RMS, ZCR

---

### GenreCNN

* 4 convolutional blocks (1 → 256 channels)
* Batch normalization + ReLU
* Adaptive average pooling
* Fully connected classifier

**Performance:**

* Validation F1: ~0.896
* Public leaderboard: ~0.801

---

### CNN + LSTM

* CNN front-end for feature extraction
* Bidirectional LSTM for temporal modeling

**Observation:**

* Improved temporal modeling but limited by GPU memory constraints

**Performance:**

* Validation F1: ~0.876
* Public leaderboard: ~0.707

---

### EfficientNet (Transfer Learning)

* EfficientNet-B0 and B3
* Modified to accept single-channel spectrogram input

Training techniques:

* SpecAugment
* Mixup
* Two-phase training (head warmup + fine-tuning)

**Performance:**

* Validation F1: ~0.86–0.88
* Public leaderboard: ~0.75

---

### Ensemble Model

* Combination of GenreCNN and EfficientNet-B0
* Test Time Augmentation (TTA)

**Best Public Score:** 0.85

---

## Evaluation Metric

**Macro F1 Score**

* Equal weight for all classes
* Sensitive to performance on difficult genres

---

## Results Summary

| Model           | Val F1 | Public Score |
| --------------- | ------ | ------------ |
| XGBoost         | ~0.65  | —            |
| GenreCNN        | 0.896  | 0.801        |
| CNN + LSTM      | 0.876  | 0.707        |
| EfficientNet-B0 | 0.866  | 0.750        |
| Ensemble        | —      | 0.85        |

---

## Key Insights

* **Data augmentation dominates performance gains**
  Most improvements came from matching the test distribution rather than changing models.

* **Distribution alignment is critical**
  Fixes such as including `other.wav`, peak normalization, and proper cropping significantly improved results.

* **CNNs are sufficient for this task**
  Spectrograms already encode temporal structure, reducing the need for explicit sequence models.

* **Pretrained models require careful adaptation**
  Input modification, learning rate tuning, and staged training are essential.

---

## Tools and Libraries

* PyTorch
* Librosa
* NumPy / Pandas
* Scikit-learn
* Weights & Biases

---

## Reproducibility

### Run Notebook

Open and execute:

```
kaggle_notebook.ipynb
```

Ensure dataset paths are correctly configured before running.

---

## Future Work

* Improved augmentation strategies
* Transformer-based audio models
* Larger pretrained audio representations
* Multi-modal feature integration

---

## References

* EfficientNet (Tan & Le, 2019)
* SpecAugment (2019)
* Mixup (2018)
* ESC-50 Dataset
* Librosa
* PyTorch

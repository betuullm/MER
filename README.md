# Multimodal Emotion Recognition using EEG and Face Modalities 😃🧠

 This repository contains code and resources for recognizing human emotions by combining EEG (Electroencephalography) signals and facial image features. The project leverages advanced signal processing, computer vision, and machine learning techniques to achieve robust emotion classification.

## 🚀 Project Overview

This project aims to classify emotions (e.g., valence) using two complementary data sources:
- **EEG Signals**: Brainwave data processed to extract statistical, spectral, and connectivity features.
- **Face Images**: Facial expressions analyzed using deep learning-based feature extraction (VGG16).

By fusing these modalities, the system achieves higher accuracy and robustness in emotion recognition tasks.

## 🗂️ Project Structure

```
├── src/
│   ├── eeg_feature.py        # EEG feature extraction
│   ├── eeg_train.py          # EEG model training
│   ├── face_feature.py       # Face feature extraction
│   ├── face_train.py         # Face model training
│   ├── fusion.py             # Multimodal fusion and evaluation
│   └── ...
├── data/                    # Processed features, labels, and results
├── DEAP_Signals/            # Raw EEG signals and labels (DEAP dataset)
├── DEAP_Face_Images/        # Extracted face images per subject
└── README.md
```

## 🧑‍💻 Main Features

- **EEG Feature Extraction**: Band power, Hjorth parameters, statistical features, and channel correlations.
- **Face Feature Extraction**: Deep features from VGG16 pretrained on ImageNet.
- **Machine Learning Pipelines**: Random Forest, Gradient Boosting, PCA, feature selection, and calibration.
- **Late Fusion**: Combines EEG and face predictions for improved accuracy.
- **Reproducible Experiments**: All steps are modular and easy to run.

## 📊 Results

- Achieves high accuracy on valence classification by leveraging both EEG and face data.
- Supports cross-validation, calibration, and detailed reporting.


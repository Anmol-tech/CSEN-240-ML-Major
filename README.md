# Visual Firewall: Detecting Network Intrusions Using Image-Based Deep Learning

A novel approach to network intrusion detection that converts network traffic flows into Gramian Angular Field (GAF) images and applies deep convolutional neural networks (CNNs) for multi-class classification.

## Overview

This project transforms network flow features into visual representations using Gramian Angular Field transformations and applies various CNN architectures to classify different types of network attacks. Using the CIC-IDS-2017 dataset, we achieve up to 99.8% accuracy in detecting network intrusions.

## Features

- **GAF Image Generation**: Converts 32-dimensional network flow features to 32×32 GAF images
- **Multiple CNN Architectures**: Implements SimpleCNN, ResNet18, and MobileNetV2 models
- **Comprehensive Evaluation**: Includes accuracy, precision, recall, F1-score, and ROC-AUC metrics
- **Class Imbalance Handling**: Implements undersampling techniques for better model performance
- **LMDB Storage**: Efficient storage and retrieval of generated GAF images

## Dataset

The project uses the **CIC-IDS-2017 dataset**, which includes:

- Normal traffic (BENIGN)
- Various attack types:
  - DDoS attacks
  - DoS attacks (GoldenEye, Hulk, Slowhttptest, slowloris)
  - Port Scan attacks
  - Brute Force attacks (FTP-Patator, SSH-Patator)
  - Web attacks (Brute Force, SQL Injection, XSS)
  - Bot attacks
  - Infiltration attacks
  - Heartbleed attacks

## Project Structure

```
CSEN-240-ML-Major/
├── data/                           # Raw CSV dataset files
│   ├── Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
│   ├── Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
│   ├── Friday-WorkingHours-Morning.pcap_ISCX.csv
│   ├── Monday-WorkingHours.pcap_ISCX.csv
│   ├── Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
│   ├── Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
│   ├── Tuesday-WorkingHours.pcap_ISCX.csv
│   └── Wednesday-workingHours.pcap_ISCX.csv
├── output/                         # Generated outputs and results
│   ├── Precision-recall-curve-simple-CNN.png
│   ├── Resnet18-confusion-matrix.png
│   ├── resnet18-precision-curve.png
│   ├── roc-curve-resnet18.png
│   ├── ROC-curve-simple-CNN.png
│   └── Simple-CNN-confusion-matrix.png
├── main.ipynb                      # Main Jupyter notebook
├── last_report.tex                 # LaTeX report
└── README.md                       # This file
```

## Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (optional but recommended)
- 32+ GB RAM (for processing large datasets)

### Dependencies

```bash
pip install torch torchvision
pip install scikit-learn
pip install pandas numpy
pip install matplotlib seaborn
pip install lmdb
pip install pyts
pip install Pillow
```

## Usage

### 1. Data Preprocessing and GAF Generation

The main notebook (`main.ipynb`) handles the complete pipeline:

1. **Data Loading**: Loads CSV files from the CIC-IDS-2017 dataset
2. **Feature Engineering**:
   - Removes non-numeric columns
   - Handles infinite values and NaN
   - Applies StandardScaler normalization
   - Reduces dimensions using PCA (32 components)
3. **GAF Transformation**: Converts 32D vectors to 32×32 GAF images
4. **LMDB Storage**: Stores images efficiently for fast access during training

### 2. Model Training

The project implements three CNN architectures:

#### SimpleCNN

- 3 convolutional layers with max pooling
- 2 fully connected layers
- Achieves 99.81% accuracy

#### ResNet18

- Modified for grayscale input (1 channel)
- Pre-trained weights with custom classification head
- Achieves 99.77% accuracy

#### MobileNetV2

- Lightweight architecture with regularization
- Custom classification head with dropout
- Achieves 99.73% accuracy

### 3. Running the Project

```bash
# Open and run the Jupyter notebook
jupyter notebook main.ipynb

# Or run cells sequentially for:
# 1. Data preprocessing
# 2. GAF image generation
# 3. Model training
# 4. Evaluation and visualization
```

## Results

### Model Performance

| Model       | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| ----------- | -------- | --------- | ------ | -------- | ------- |
| SimpleCNN   | 99.81%   | 99.81%    | 99.81% | 99.81%   | 0.997   |
| ResNet18    | 99.77%   | 99.77%    | 99.77% | 99.76%   | 0.997   |
| MobileNetV2 | 99.73%   | 99.73%    | 99.73% | 99.72%   | 0.996   |

### Key Findings

- **High Performance**: All models achieve >99% accuracy on the test set
- **Class Imbalance**: BENIGN traffic dominates the dataset; undersampling improves performance
- **Model Complexity**: SimpleCNN performs comparably to deeper models, suggesting the GAF representation is highly effective
- **Visual Learning**: Converting network flows to images enables effective pattern recognition

## Technical Details

### GAF Transformation

```python
def gaf_transform_torch(x, device='cpu'):
    min_x = x.min(dim=1, keepdim=True).values
    max_x = x.max(dim=1, keepdim=True).values
    scaled_x = (2 * (x - min_x) / (max_x - min_x + 1e-8)) - 1
    scaled_x = torch.clamp(scaled_x, -1, 1)
    phi = torch.arccos(scaled_x)
    gaf = torch.cos(phi.unsqueeze(2) + phi.unsqueeze(1))
    return gaf
```

### Data Pipeline

1. **Feature Selection**: Removes non-predictive columns (IPs, timestamps, etc.)
2. **Normalization**: StandardScaler for feature scaling
3. **Dimensionality Reduction**: PCA to 32 components
4. **Image Generation**: GAF transformation to 32×32 images
5. **Storage**: LMDB for efficient I/O during training

### Class Mapping

The project maps specific attack types to broader categories:

- DoS attacks → 'DoS'
- Brute force attacks → 'BruteForce'
- Web attacks → 'WebAttack'
- Individual categories: DDoS, PortScan, Bot, Infiltration, Heartbleed

## Evaluation Metrics

The project provides comprehensive evaluation including:

- **Confusion Matrices**: Visual representation of classification performance
- **ROC Curves**: True positive rate vs false positive rate
- **Precision-Recall Curves**: Precision vs recall trade-offs
- **Multi-class Metrics**: Weighted averages across all classes

## Hardware Requirements

- **RAM**: 32+ GB (for large dataset processing)
- **Storage**: 50+ GB (for LMDB database and outputs)
- **GPU**: CUDA-compatible (optional, significantly speeds up training)

## Future Work

- **Real-time Detection**: Implement streaming GAF generation
- **Rare Class Handling**: Better techniques for minority class detection
- **Feature Engineering**: Explore different PCA components and normalization methods
- **Ensemble Methods**: Combine multiple models for improved performance

## Contributors

- **Siddharth Bhat** - Santa Clara University
- **Anmol Sharma** - Santa Clara University

## License

This project is for academic purposes as part of CSEN-240 coursework at Santa Clara University.

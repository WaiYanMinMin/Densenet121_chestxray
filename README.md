# Advanced Machine Learning and Neural Networks - Assignment 1

Evaluation of a pre-trained Convolutional Neural Network (CNN) for multi-label chest X-ray classification using the CheXpert dataset.

## Project Overview

This project demonstrates an end-to-end deep learning workflow for medical image analysis:

- loading a pre-trained CNN (`DenseNet-121`)
- preparing and analyzing a labeled dataset (`CheXpert`)
- training and validating a multi-label classifier
- evaluating model performance with standard metrics
- visualizing label-wise confusion matrices
- running inference on unseen unlabeled images

The work is implemented in:

- `notebook/WAIYANMINMIN_AMLNN_AS1.ipynb`

## Assignment Scope Covered

This notebook is structured to satisfy the assignment requirements:

1. **Investigate architecture** of a pre-trained CNN and discuss improvements.
2. **Evaluate performance** using known metrics on training/validation data.
3. **Compare predictions vs ground truth** with confusion matrices.
4. **Summarize findings** with limitations and recommendations.

## Dataset and Labels

- **Dataset**: CheXpert (small), accessed via `kagglehub`
- **Task type**: Multi-label image classification
- **Selected target labels (5)**:
  - Atelectasis
  - Cardiomegaly
  - Consolidation
  - Edema
  - Pleural Effusion

## Model and Training Setup

- **Backbone**: `torchvision.models.densenet121` (ImageNet pre-trained)
- **Classifier head**: Replaced final layer for 5-label output
- **Loss**: `BCEWithLogitsLoss`
- **Optimizer**: Adam (`lr=1e-4`)
- **Image size**: `224 x 224`
- **Epochs**: 3
- **Threshold for binary decisions**: 0.5

## Key Validation AUC (from notebook run)

- Atelectasis: **0.7659**
- Cardiomegaly: **0.8171**
- Consolidation: **0.9313**
- Edema: **0.8919**
- Pleural Effusion: **0.9155**

> Note: metrics can vary slightly by runtime environment and random seed.

## How to Run

### Option A: Kaggle/Colab notebook workflow (recommended)

1. Open `notebook/WAIYANMINMIN_AMLNN_AS1.ipynb`
2. Run cells from top to bottom
3. The notebook installs required packages and downloads CheXpert via `kagglehub`

### Option B: Local environment

Install dependencies:

```bash
pip install kagglehub torch torchvision pandas pillow scikit-learn tqdm seaborn matplotlib
```

Then run the notebook in Jupyter:

```bash
jupyter notebook
```

## Repository Structure

```text
.
├── notebook/
│   └── WAIYANMINMIN_AMLNN_AS1.ipynb
└── README.md
```

## Results and Discussion Highlights

- Transfer learning with DenseNet-121 provides a strong baseline for chest X-ray labels.
- Performance differs by label due to class imbalance and label prevalence.
- Confusion matrices help identify false positives/false negatives for each pathology.
- Improvements such as threshold tuning, weighted losses, and LR scheduling are discussed in the notebook.

## Author

**Wai Yan Min Min**  
MSc/Advanced ML coursework portfolio project

# AMLNN Assignment 1 — Chest X-ray classifier

Pre-trained DenseNet-121 on CheXpert for 5 labels: Atelectasis, Cardiomegaly, Consolidation, Edema, Pleural Effusion.

## Run the notebook

1. Open `notebook/WAIYANMINMIN_AMLNN_AS1.ipynb`
2. Run all cells (Colab/Kaggle or local Jupyter)

```bash
pip install kagglehub torch torchvision pandas pillow scikit-learn tqdm seaborn matplotlib
```

# Deployment Guide

## 1) Save and place model

1. In the notebook, run the **Save model for deployment** cell.
2. If using Colab, run the **copy model to Google Drive** cell.
3. Download `chest_xray_densenet.pth` and place it at:

```bash
models/chest_xray_densenet.pth
```

## 2) Create and activate virtual environment

From the project root:

```bash
python -m venv .venv
```

Activate it:

- **Windows (PowerShell):**

```powershell
.\.venv\Scripts\Activate.ps1
```

- **Windows (Command Prompt):**

```bat
.\.venv\Scripts\activate.bat
```

- **macOS/Linux:**

```bash
source .venv/bin/activate
```

## 3) Install dependencies and run app

```bash
pip install -r requirements-deploy.txt
python app.py
```

Open the printed URL (for example, `http://127.0.0.1:7860`) and upload an X-ray image.

## Live Demo

Hugging Face Space: https://huggingface.co/spaces/Wymm2003/chest_X_ray_detection

# AMLNN Assignment 1 — Chest X-ray classifier

Pre-trained DenseNet-121 on CheXpert for 5 labels: Atelectasis, Cardiomegaly, Consolidation, Edema, Pleural Effusion.

## Run the notebook

1. Open `notebook/WAIYANMINMIN_AMLNN_AS1.ipynb`
2. Run all cells (Colab/Kaggle or local Jupyter)

```bash
pip install kagglehub torch torchvision pandas pillow scikit-learn tqdm seaborn matplotlib
```

## Deploy (upload X-ray → predictions)

1. **Save model** — In the notebook, run the “Save model for deployment” cell.
2. **Get model to your PC (Colab kernel)** — Run the “copy model to Google Drive” cell, then open [drive.google.com](https://drive.google.com) → **My Drive → ColabOutput** → download `chest_xray_densenet.pth` → put it in `models/chest_xray_densenet.pth` in this project.
3. **Run app** — From project root:

```bash
pip install -r requirements-deploy.txt
python app.py
```

Open the URL (e.g. http://127.0.0.1:7860), upload an X-ray.


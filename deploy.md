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

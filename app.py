"""
Simple Gradio app: upload a chest X-ray image and get model predictions.
Run: pip install gradio torch torchvision pillow
     python app.py
Then open the URL (e.g. http://127.0.0.1:7860) in your browser.
"""
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import gradio as gr

# Must match notebook
LABELS = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Pleural Effusion",
]

IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def get_model(num_classes: int):
    model = models.densenet121(weights=None)
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    return model


def load_model(path: str = "models/chest_xray_densenet.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(len(LABELS)).to(device)
    state = torch.load(path, map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, device


def predict(image: Image.Image, model, device) -> str:
    if image is None:
        return "Please upload a chest X-ray image."
    img_tensor = IMAGE_TRANSFORM(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(img_tensor)
        probs = torch.sigmoid(logits).cpu().numpy()[0]
    preds = probs >= 0.5
    lines = ["**Predictions:**"]
    for i, label in enumerate(LABELS):
        p = float(probs[i])
        status = "✓" if preds[i] else "—"
        lines.append(f"- {status} **{label}**: {p:.1%}")
    active = [LABELS[i] for i in range(len(LABELS)) if preds[i]]
    if active:
        lines.append(f"\n**Detected:** {', '.join(active)}")
    else:
        lines.append("\n**Detected:** No positive findings (above threshold).")
    return "\n".join(lines)


def main():
    import os
    model_path = os.path.join(os.path.dirname(__file__), "models", "chest_xray_densenet.pth")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"Model not found at {model_path}. "
            "Train the notebook and run the 'Save model for deployment' cell first."
        )
    model, device = load_model(model_path)

    demo = gr.Interface(
        fn=lambda img: predict(img, model, device),
        inputs=gr.Image(type="pil", label="Upload chest X-ray"),
        outputs=gr.Markdown(label="Prediction"),
        title="Chest X-ray classifier",
        description="Upload a chest X-ray image. The model predicts: Atelectasis, Cardiomegaly, Consolidation, Edema, Pleural Effusion.",
    )
    demo.launch()


if __name__ == "__main__":
    main()

"""
Final Gradio app for Hugging Face Space — loads checkpoint from local files.
Requirements (requirements.txt):
- gradio
- torch
- torchvision
- pillow
- numpy
- gradio_client
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torchvision import transforms
import numpy as np
from PIL import Image
import gradio as gr
import cv2
from gradio_client import Client, handle_file
import tempfile, os

# -------------------- Config --------------------
CKPT_PATH = "best_checkpoint6.pth"
LABELS = [
    'Aquarius','Aries','Cancer','Capricornus','Gemini','Leo',
    'Libra','Pisces','Sagittarius','Scorpius','Taurus','Virgo'
]
NUM_CLASSES = len(LABELS)
ALLOWED_KEYWORDS = ["stars", "starry", "constellation", "white dots", "night sky", "milky way"]

# CLIP Interrogator client
clip_client = Client("fffiloni/CLIP-Interrogator-2")

# -------------------- Image Verification --------------------
def check_image_description(pil_img: Image.Image):
    """Verify that the image looks like a constellation or deep sky object. Returns (is_valid, description)."""
    temp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
    pil_img.save(temp_path)

    result = clip_client.predict(
        image=handle_file(temp_path),
        mode="best",
        best_max_flavors=4,
        api_name="/clipi2"
    )
    os.remove(temp_path)

    description = str(result).lower()
    is_valid = any(keyword in description for keyword in ALLOWED_KEYWORDS)
    return is_valid, description

# -------------------- Preprocessing --------------------
def contrast_stretch_and_binarize(image: Image.Image, threshold: int = 128) -> Image.Image:
    img_arr = np.array(image).astype(np.float32)
    min_val, max_val = img_arr.min(), img_arr.max()
    if max_val - min_val < 1e-6:
        stretched = np.zeros_like(img_arr)
    else:
        stretched = (img_arr - min_val) * (255.0 / (max_val - min_val))

    binary = np.where(stretched >= threshold, 255, 0).astype(np.uint8)
    if binary.ndim == 3 and binary.shape[2] == 4:
        binary = binary[:, :, :3]

    # Dilation
    if binary.ndim == 3:
        gray = cv2.cvtColor(binary, cv2.COLOR_RGB2GRAY)
    else:
        gray = binary
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(gray, kernel, iterations=1)
    dilated_rgb = cv2.cvtColor(dilated, cv2.COLOR_GRAY2RGB)

    return Image.fromarray(dilated_rgb)

def get_prediction_json(logits: torch.Tensor, labels: list, temp: float = 1.0):
    probs = F.softmax(logits / temp, dim=0)
    top_idx = torch.argmax(probs).item()
    return {
        "predicted_class": labels[top_idx],
        "confidence": float(round(probs[top_idx].item(), 4))
    }

inference_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# -------------------- Model --------------------
class EfficientNetB0(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, num_classes)
    def forward(self, x):
        return self.model(x)

def load_model_once():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = EfficientNetB0(num_classes=NUM_CLASSES)
    checkpoint = torch.load(CKPT_PATH, map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, device

MODEL, DEVICE = load_model_once()

# -------------------- Prediction --------------------
def predict_image_from_pil(pil_img: Image.Image, temperature: float = 1.0):
    # Step 1: Verify with CLIP Interrogator
    is_valid, description = check_image_description(pil_img)
    if not is_valid:
        return {"error": "Only constellation or deep sky images are allowed."}, None, description

    # Step 2: Process image once for both preview and model
    img_rgb = pil_img.convert('RGB')
    processed_img = contrast_stretch_and_binarize(img_rgb)

    # Step 3: Inference
    tensor = inference_transform(processed_img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = MODEL(tensor).squeeze(0)
        result = get_prediction_json(logits, LABELS, temperature)

    return result, processed_img, description

# -------------------- Gradio UI --------------------
with gr.Blocks(title="EfficientNet-B0 Constellation Classifier") as demo:
    gr.Markdown("""## Classifies only the 12 zodiac constellations""")
    with gr.Row():
        with gr.Column(scale=2):
            img_in = gr.Image(type="pil", label="Input image")
            temp_slider = gr.Slider(0.1, 6.0, value=1.0, step=0.1, label="Temperature")
            predict_btn = gr.Button("Predict")
        with gr.Column(scale=1):
            out_json = gr.JSON(label="Prediction")
            preproc_image = gr.Image(type="pil", label="Binarized + Dilated Preview")
            description_box = gr.Textbox(label="Detected Description", interactive=False)

    def _on_predict(image, temperature):
        if image is None:
            return {"error": "Please upload an image."}, None, ""
        return predict_image_from_pil(image, temperature)

    predict_btn.click(
        _on_predict,
        inputs=[img_in, temp_slider],
        outputs=[out_json, preproc_image, description_box]
    )

if __name__ == "__main__":
    demo.launch()

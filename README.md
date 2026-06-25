# 🌌 Zenoverse - Zodiac Constellation Classifier

Zenoverse is an AI-powered zodiac constellation classifier built with **PyTorch**, **EfficientNet-B0**, and **Gradio**. It identifies the **12 zodiac constellations** from uploaded night-sky images while ensuring that only valid astronomical images are accepted using the **CLIP Interrogator**.

## ✨ Features

* 🔭 Classifies the **12 Zodiac Constellations**
* 🧠 EfficientNet-B0 fine-tuned for constellation recognition
* 🛡️ CLIP-based image verification to reject non-space images
* 🖼️ Automatic image preprocessing

  * Contrast Stretching
  * Binary Thresholding
  * Morphological Dilation
* 📊 Confidence score for every prediction
* 🎛️ Adjustable Softmax Temperature
* ⚡ Fast interactive Gradio interface
* 🤗 Ready to deploy on Hugging Face Spaces

---

## Supported Constellations

* Aquarius
* Aries
* Cancer
* Capricornus
* Gemini
* Leo
* Libra
* Pisces
* Sagittarius
* Scorpius
* Taurus
* Virgo

---

## Model Architecture

* Backbone: **EfficientNet-B0**
* Framework: **PyTorch**
* Input Size: **224 × 224**
* Output Classes: **12**

The final classification layer is replaced with a custom linear layer matching the zodiac constellation classes.

---

## Image Verification

Before classification, every uploaded image is verified using the **CLIP Interrogator**.

Only images that resemble:

* Stars
* Starry sky
* Constellations
* White dots
* Night sky
* Milky Way

are accepted.

Images unrelated to astronomy are automatically rejected.

---

## Image Preprocessing Pipeline

Each accepted image undergoes the following preprocessing steps:

1. Convert to RGB
2. Contrast Stretching
3. Binary Thresholding
4. Morphological Dilation
5. Resize to 224×224
6. ImageNet Normalization

The processed image is also displayed for transparency.

---

## Project Structure

```
.
├── app.py
├── best_checkpoint6.pth
├── requirements.txt
└── README.md
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/Zenoverse.git
cd Zenoverse
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running Locally

```bash
python app.py
```

Gradio will launch locally and provide a browser link.

---

## Requirements

```
gradio
torch
torchvision
numpy
opencv-python
Pillow
gradio_client
```

---

## Model Checkpoint

Place the trained checkpoint in the project directory:

```
best_checkpoint6.pth
```

The application automatically loads the checkpoint during startup.

---

## Output

For every valid image, Zenoverse returns:

* Predicted Zodiac Constellation
* Confidence Score
* Preprocessed Image Preview
* CLIP Image Description

Example:

```json
{
    "predicted_class": "Scorpius",
    "confidence": 0.9824
}
```

---

## Technologies Used

* PyTorch
* TorchVision
* EfficientNet-B0
* Gradio
* OpenCV
* Pillow
* NumPy
* CLIP Interrogator

---

## Future Improvements

* Support for all 88 IAU constellations
* Top-k prediction visualization
* Grad-CAM explainability
* Mobile deployment
* Real-time sky recognition
* Star map overlay
* Multi-language interface

---

## License

This project is intended for educational and research purposes.

---

## Acknowledgements

* PyTorch
* TorchVision
* Gradio
* OpenCV
* Hugging Face
* CLIP Interrogator

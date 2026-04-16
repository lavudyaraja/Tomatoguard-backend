<<<<<<< HEAD
# 🧠 TomatoGuard AI Backend

This is the high-performance **Inference Engine** for the TomatoGuard AI ecosystem. It uses Deep Learning to diagnose tomato diseases from images with high accuracy.

## 🚀 Technical Highlights
- **Framework**: FastAPI for high-speed, asynchronous request handling.
- **Model Architecture**: **MaxViT (Multi-Axis Vision Transformer)** for superior image feature extraction.
- **Computer Vision**: Powered by **PyTorch** and **Torchvision**.
- **Data Persistence**: Integrated with **Neon PostgreSQL** and **Cloudinary** for persistent history and CDNs.

## 🧬 Diagnostic Capability
The engine is trained/configured to identify following conditions:
- Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot
- Spider Mite, Target Spot, Yellow Leaf Curl Virus, Mosaic Virus, Powdery Mildew
- Healthy Plant Identification

## 🛠️ API Endpoints
- `POST /predict`: Receives a leaf image, performs inference, and stores results.
- `GET /health`: Checks system and model status.

## 👨‍💻 Author
Developed by **Lavudya Raja**.
Explore more of my work at **[lavudyaraja.in](https://lavudyaraja.in)**.

---

## 🚦 Installation
```bash
# Clone the repository
git clone https://github.com/lavudyaraja/Tomatoguard-backend.git

# Install dependencies
pip install -r requirements.txt

# Run the server
uvicorn main:app --reload --port 8000
```

*Protecting your crops with the power of Machine Intelligence.*
=======
---
title: Tomato Disease Classifier
emoji: 📉
colorFrom: purple
colorTo: blue
sdk: docker
pinned: false
short_description: AI-powered Tomato Leaf Disease Detection using MaxViT
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
>>>>>>> 0e47c30e6bb6f04d6a7dd0eb59ffe4a4e41848ac

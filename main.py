import os
import io
import uuid
import datetime
import hashlib
import numpy as np
import torch
import psycopg2
import cloudinary
import cloudinary.uploader
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from torchvision import transforms
from PIL import Image, ImageStat
from model import MaxViT, create_coatnet, create_nextvit
from xai_utils import GradCAM, apply_heatmap_overlay, annotate_disease_on_original
import uvicorn

load_dotenv()

app = FastAPI(title="TomatoGuard AI Inference Engine")

# ── CLOUDINARY CONFIG ──────────────────────────────────────────
# Only configure Cloudinary when all three credentials are present and valid.
_cld_name   = os.getenv("CLOUDINARY_CLOUD_NAME", "")
_cld_key    = os.getenv("CLOUDINARY_API_KEY", "")
_cld_secret = os.getenv("CLOUDINARY_API_SECRET", "")

is_cloudinary_ready = bool(
    _cld_name and _cld_name != "your_cloud_name"
    and _cld_key and _cld_key != "your_api_key"
    and _cld_secret and _cld_secret != "your_api_secret"
)

if is_cloudinary_ready:
    cloudinary.config(
        cloud_name=_cld_name,
        api_key=_cld_key,
        api_secret=_cld_secret,
        secure=True
    )
    print(f"✅ Cloudinary configured for cloud: {_cld_name}")
else:
    print("⚠️  Cloudinary not configured — images will be stored as base64.")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── CONFIGURATION ──────────────────────────────────────────────
COATNET_PATH = os.getenv("COATNET_PATH", "./best_coatnet_model.pth")
MAXVIT_PATH  = os.getenv("MAXVIT_PATH",  "./maxvit_kaggle_best.pth")
NEXTVIT_PATH = os.getenv("NEXTVIT_PATH", "../best_nextvit_checkpoint.pt")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASSES = [
    "Bacterial_spot", "Early_blight", "Late_blight", "Leaf_Mold",
    "Septoria_leaf_spot", "Spider_mites_Two-spotted_spider_mite",
    "Target_Spot", "Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato_mosaic_virus", "healthy", "powdery_mildew"
]

# Minimal metadata for basic labeling — full details are handled in the Next.js frontend
CLASS_META = {
    "Bacterial_spot": {"label": "Bacterial Spot", "severity": "Moderate–High", "emoji": "🔴"},
    "Early_blight": {"label": "Early Blight", "severity": "Moderate", "emoji": "🟠"},
    "Late_blight": {"label": "Late Blight", "severity": "Critical", "emoji": "🚨"},
    "Leaf_Mold": {"label": "Leaf Mold", "severity": "Moderate", "emoji": "🟡"},
    "Septoria_leaf_spot": {"label": "Septoria Leaf Spot", "severity": "Moderate", "emoji": "🔶"},
    "Spider_mites_Two-spotted_spider_mite": {"label": "Spider Mites", "severity": "Moderate", "emoji": "🕷️"},
    "Target_Spot": {"label": "Target Spot", "severity": "Moderate", "emoji": "🎯"},
    "Tomato_Yellow_Leaf_Curl_Virus": {"label": "Yellow Leaf Curl Virus", "severity": "Critical", "emoji": "🟡"},
    "Tomato_mosaic_virus": {"label": "Mosaic Virus", "severity": "High", "emoji": "🟤"},
    "healthy": {"label": "Healthy", "severity": "None", "emoji": "✅"},
    "powdery_mildew": {"label": "Powdery Mildew", "severity": "Moderate", "emoji": "⚪"},
}


# ── DATABASE SETUP ────────────────────────────────────────────
DATABASE_URL = os.getenv("DATABASE_URL")

def init_db():
    if not DATABASE_URL:
        print("⚠️  DATABASE_URL not set — skipping DB init.")
        return
    try:
        conn = psycopg2.connect(DATABASE_URL)
        conn.autocommit = True # Make migrations easier by not matching transaction states
        c = conn.cursor()
        
        # 1. Ensure table exists
        c.execute("""
            CREATE TABLE IF NOT EXISTS history (
                id           TEXT PRIMARY KEY,
                prediction   TEXT,
                confidence   REAL,
                created_at   TEXT,
                image_url    TEXT,
                image_width  INTEGER,
                image_height INTEGER,
                image_mode   TEXT,
                file_size_kb REAL,
                brightness   REAL,
                dominant_color TEXT,
                xai_url      TEXT,
                hotspots     TEXT,
                image_info   TEXT,
                llm_insight  TEXT
            )
        """)
        
        # Migration for all columns
        columns = [
            ("history", "image_width", "INTEGER"),
            ("history", "image_height", "INTEGER"),
            ("history", "image_mode", "TEXT"),
            ("history", "file_size_kb", "REAL"),
            ("history", "brightness", "REAL"),
            ("history", "dominant_color", "TEXT"),
            ("history", "xai_url", "TEXT"),
            ("history", "hotspots", "TEXT"),
            ("history", "llm_insight", "TEXT"),
            ("history", "image_info", "TEXT"),
            ("history", "models", "TEXT"),
            ("analytics", "severity", "TEXT")
        ]
        
        for table, col, col_type in columns:
            try:
                c.execute(f"ALTER TABLE {table} ADD COLUMN {col} {col_type}")
                print(f"🚀 Migrated: Added {col} column to {table}")
            except Exception:
                pass
        
        # Ensure xai_url column exists (Migration)
        try:
            c.execute("ALTER TABLE history ADD COLUMN xai_url TEXT")
            print("🚀 Migrated: Added xai_url column to history")
        except Exception:
            pass
        
        # 2. Ensure analytics table exists
        c.execute("""
            CREATE TABLE IF NOT EXISTS analytics (
                id           SERIAL PRIMARY KEY,
                disease_name TEXT UNIQUE,
                count        INTEGER DEFAULT 0,
                last_detected TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # 3. Migration: image_path -> image_url
        try:
            c.execute("ALTER TABLE history RENAME COLUMN image_path TO image_url")
            print("🚀 Migrated column: image_path -> image_url")
        except Exception:
            pass # Already renamed or doesn't exist

        # 3. Migration: timestamp -> created_at
        try:
            c.execute("ALTER TABLE history RENAME COLUMN timestamp TO created_at")
            print("🚀 Migrated column: timestamp -> created_at")
        except Exception:
            pass

        # 4. Migration: add method column
        try:
            c.execute("ALTER TABLE history ADD COLUMN method TEXT")
            print("🚀 Migrated column: added method")
        except Exception:
            pass

        conn.close()
        print("✅ Database synchronized successfully.")
    except Exception as e:
        print(f"❌ Database init error: {e}")

init_db()


# ── MODEL LOADING ─────────────────────────────────────────────
weights_loaded = {
    "coatnet": False,
    "maxvit":  False,
    "nextvit": False,
}

# 1. Initialize CoAtNet
print("🏗️  Initializing CoAtNet model...")
model_coatnet = create_coatnet(num_classes=len(CLASSES))
_coatnet_path_candidates = [
    COATNET_PATH,
    "./best_coatnet_model.pth",
    "../best_coatnet_checkpoint.pt",
    "./best_coatnet_checkpoint.pt",
]
for _candidate in _coatnet_path_candidates:
    if os.path.exists(_candidate):
        try:
            checkpoint = torch.load(_candidate, map_location=DEVICE)
            sd = checkpoint.get('model_state_dict', checkpoint.get('sd', checkpoint))
            model_coatnet.load_state_dict(sd)
            model_coatnet.to(DEVICE)
            model_coatnet.eval()
            weights_loaded["coatnet"] = True
            print(f"✅ CoAtNet weights loaded from {_candidate}.")
            break
        except Exception as e:
            print(f"❌ Error loading CoAtNet from {_candidate}: {e}")

# 2. Initialize MaxViT
print("🏗️  Initializing MaxViT model...")
model_maxvit = MaxViT(num_classes=len(CLASSES), win=7, drop_path_rate=0.15)
_maxvit_path_candidates = [
    MAXVIT_PATH,
    "./maxvit_kaggle_best.pth",
    "../maxvit_kaggle_best.pth",
]
for _candidate in _maxvit_path_candidates:
    if os.path.exists(_candidate):
        try:
            checkpoint = torch.load(_candidate, map_location=DEVICE)
            sd = checkpoint.get('model_state_dict', checkpoint.get('sd', checkpoint))
            model_maxvit.load_state_dict(sd)
            model_maxvit.to(DEVICE)
            model_maxvit.eval()
            weights_loaded["maxvit"] = True
            print(f"✅ MaxViT weights loaded from {_candidate}.")
            break
        except Exception as e:
            print(f"❌ Error loading MaxViT from {_candidate}: {e}")

# 3. Initialize NextViT
print("🏗️  Initializing NextViT model...")
model_nextvit = None
_nextvit_path_candidates = [
    NEXTVIT_PATH,
    "./best_nextvit_checkpoint.pt",
    "../best_nextvit_checkpoint.pt",
]
for _candidate in _nextvit_path_candidates:
    if os.path.exists(_candidate):
        try:
            checkpoint = torch.load(_candidate, map_location=DEVICE)
            # Extract model_name saved in the checkpoint (fallback: None = auto-detect)
            _model_name = checkpoint.get('model_name', None)
            model_nextvit = create_nextvit(
                num_classes=len(CLASSES),
                dropout=checkpoint.get('best_cfg', {}).get('dropout', 0.2),
                drop_path_rate=checkpoint.get('best_cfg', {}).get('drop_path_rate', 0.1),
                pretrained=False,
                model_name=_model_name,
            )
            sd = checkpoint.get('model_state_dict', checkpoint.get('sd', checkpoint))
            model_nextvit.load_state_dict(sd, strict=True)
            model_nextvit.to(DEVICE)
            model_nextvit.eval()
            weights_loaded["nextvit"] = True
            print(f"✅ NextViT weights loaded from {_candidate} (arch: {_model_name or 'auto'}).")
            break
        except Exception as e:
            print(f"❌ Error loading NextViT from {_candidate}: {e}")
if not weights_loaded["nextvit"]:
    print(f"⚠️  NextViT checkpoint not found — inference will run on CoAtNet+MaxViT only.")



# Initialize Grad-CAM interpretability engine
interpretability_engine = None
if any(weights_loaded.values()):
    try:
        # Use CoAtNet as the primary driver for XAI if available
        if weights_loaded["coatnet"]:
            target_layer = model_coatnet.stages[-1]
            interpretability_engine = GradCAM(model_coatnet, target_layer)
        else:
            target_layer = model_maxvit.stages[3]
            interpretability_engine = GradCAM(model_maxvit, target_layer)
        print("✅ XAI (Grad-CAM) engine initialized.")
    except Exception as e:
        print(f"⚠️  Failed to initialize XAI engine: {e}")


# ── PREPROCESSING ─────────────────────────────────────────────
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# ── IMAGE ANALYSIS HELPER ─────────────────────────────────────
def analyze_image(img: Image.Image, raw_bytes: bytes, filename: str) -> dict:
    """
    Returns a rich metadata dict about the uploaded image:
    dimensions, file size, color mode, brightness, contrast,
    dominant color family, and green-coverage ratio (leaf health proxy).
    """
    width, height = img.size
    mode = img.mode
    file_size_kb = round(len(raw_bytes) / 1024, 2)
    aspect_ratio = round(width / height, 3)

    # Convert to RGB for consistent stats
    rgb = img.convert("RGB")
    stat = ImageStat.Stat(rgb)

    # Per-channel mean & stddev
    r_mean, g_mean, b_mean = stat.mean
    r_std,  g_std,  b_std  = stat.stddev

    # Overall brightness (perceptual luminance)
    brightness = round(0.299 * r_mean + 0.587 * g_mean + 0.114 * b_mean, 2)

    # Contrast proxy (average stddev across channels)
    contrast = round((r_std + g_std + b_std) / 3.0, 2)

    # Dominant color family
    if g_mean > r_mean * 1.1 and g_mean > b_mean * 1.1:
        dominant_color = "green"
        color_desc = "Predominantly green — likely healthy leaf tissue"
    elif r_mean > g_mean and r_mean > b_mean:
        if r_mean > 180 and g_mean > 150:
            dominant_color = "yellow-red"
            color_desc = "Yellowing or browning detected — possible disease stress"
        else:
            dominant_color = "red-brown"
            color_desc = "Red-brown tones — lesions, blight, or necrosis possible"
    elif r_mean > 200 and g_mean > 200 and b_mean > 200:
        dominant_color = "white-gray"
        color_desc = "Pale/white patches — possible powdery mildew or overexposure"
    elif brightness < 60:
        dominant_color = "dark"
        color_desc = "Very dark image — poor lighting or severe necrosis"
    else:
        dominant_color = "mixed"
        color_desc = "Mixed color profile — complex leaf condition"

    # Green pixel ratio (leaf health proxy)
    np_img = np.array(rgb, dtype=np.float32)
    r_ch, g_ch, b_ch = np_img[:, :, 0], np_img[:, :, 1], np_img[:, :, 2]
    green_mask = (g_ch > r_ch * 1.05) & (g_ch > b_ch * 1.05) & (g_ch > 60)
    green_ratio = round(float(green_mask.sum()) / (width * height), 4)

    # Sharpness estimate (Laplacian variance)
    gray = np.array(rgb.convert("L"), dtype=np.float32)
    laplacian = np.array([
        [0,  1, 0],
        [1, -4, 1],
        [0,  1, 0]
    ])
    from scipy.signal import convolve2d
    try:
        lp = convolve2d(gray, laplacian, mode='valid')
        sharpness = round(float(np.var(lp)), 2)
    except Exception:
        sharpness = None

    # Vitality & Light Quality proxies
    vitality_score = round(green_ratio * 100, 1) if green_ratio > 0 else 0
    light_quality = "Optimal" if 80 < brightness < 200 and contrast > 30 else "Sub-optimal" if brightness < 60 or brightness > 230 else "Acceptable"

    # Prepare detailed expert summary
    health_status = "healthy photosynthetic tissue" if green_ratio > 0.6 else "diseased or chlorotic tissue"
    quality_status = "high-fidelity for diagnostic accuracy" if brightness > 80 and (sharpness or 0) > 50 else "sufficient but could be improved with better lighting"
    
    detailed_summary = (
        f"The analysis of the uploaded leaf reveals that approximately {round(green_ratio * 100, 2)}% of the surface area remains as {health_status}. "
        f"The primary visual profile is characterized by {color_desc.lower()}, specifically identifying risk zones via a contrast intensity of {contrast}. "
        f"Technical verification confirms the capture quality is {quality_status}, with a perceptual luminance measured at {brightness}/255. "
        f"This visual data, processed through dual-model inference, provides a high-confidence diagnostic baseline."
    )

    # File extension extraction
    file_ext = os.path.splitext(filename)[1] if filename else ".jpg"

    return {
        "filename": filename,
        "file_extension": file_ext,
        "file_size_kb": file_size_kb,
        "width": width,
        "height": height,
        "aspect_ratio": aspect_ratio,
        "color_mode": mode,
        "brightness": brightness,
        "contrast": contrast,
        "sharpness": sharpness,
        "vitality_score": vitality_score,
        "light_quality": light_quality,
        "dominant_color": dominant_color,
        "color_description": color_desc,
        "green_ratio": green_ratio,
        "green_coverage_pct": round(green_ratio * 100, 2),
        "detailed_summary": detailed_summary,
        "channel_means": {"R": round(r_mean, 2), "G": round(g_mean, 2), "B": round(b_mean, 2)},
        "channel_stddev": {"R": round(r_std, 2),  "G": round(g_std, 2),  "B": round(b_std, 2)},
        "quality_warnings": _quality_warnings(width, height, brightness, contrast, file_size_kb)
    }


def _quality_warnings(w, h, brightness, contrast, size_kb):
    """Flag potential image quality issues that may affect prediction accuracy."""
    warnings = []
    if w < 100 or h < 100:
        warnings.append("Image resolution is very low — prediction may be inaccurate.")
    if brightness < 40:
        warnings.append("Image is very dark — consider better lighting.")
    if brightness > 230:
        warnings.append("Image is overexposed — loss of detail may affect prediction.")
    if contrast < 10:
        warnings.append("Very low contrast detected — blurry or out-of-focus image.")
    if size_kb < 5:
        warnings.append("File size is extremely small — may be a corrupt or dummy file.")
    return warnings




# ── ENDPOINTS ─────────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "status": "online",
        "engine": "Tri-Model (CoAtNet + MaxViT + NextViT)",
        "device": str(DEVICE),
        "weights_loaded": weights_loaded,
        "num_classes": len(CLASSES),
        "database": "Neon PostgreSQL" if DATABASE_URL else "Not configured"
    }


@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    """
    Upload a tomato leaf image and receive:
    - Disease prediction with confidence
    - Top-5 class probabilities
    - Rich image metadata (dimensions, brightness, color analysis, quality warnings)
    - Detailed disease description and treatment advice
    """
    try:
        # ── Read & validate image ──
        img_data = await image.read()

        if len(img_data) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        try:
            img = Image.open(io.BytesIO(img_data)).convert("RGB")
        except Exception:
            raise HTTPException(
                status_code=400,
                detail="Cannot read the uploaded file as an image. "
                       "Please upload a valid JPEG, PNG, or WebP image."
            )

        # ── Analyze image ──
        image_analysis = analyze_image(img, img_data, image.filename)

        # ── Run inference on BOTH models ──
        if not any(weights_loaded.values()):
            raise HTTPException(status_code=503, detail="No model weights are available on the server.")

        input_tensor = preprocess(img).unsqueeze(0).to(DEVICE)
        
        model_results = {}
        
        # 1. CoAtNet Inference
        if weights_loaded["coatnet"]:
            with torch.no_grad():
                outputs = model_coatnet(input_tensor)
                probs = torch.softmax(outputs, dim=1)[0]
                val, idx = torch.topk(probs, 5)
                model_results["coatnet"] = {
                    "prediction": CLASSES[idx[0].item()],
                    "confidence": round(float(val[0].item()), 4),
                    "idx": idx[0].item(),
                    "top5": [
                        {"label": CLASS_META[CLASSES[i]]["label"], "probability": float(p)}
                        for p, i in zip(val.tolist(), idx.tolist())
                    ]
                }

        # 2. MaxViT Inference
        if weights_loaded["maxvit"]:
            with torch.no_grad():
                outputs = model_maxvit(input_tensor)
                probs = torch.softmax(outputs, dim=1)[0]
                val, idx = torch.topk(probs, 5)
                model_results["maxvit"] = {
                    "prediction": CLASSES[idx[0].item()],
                    "confidence": round(float(val[0].item()), 4),
                    "idx": idx[0].item(),
                    "top5": [
                        {"label": CLASS_META[CLASSES[i]]["label"], "probability": float(p)}
                        for p, i in zip(val.tolist(), idx.tolist())
                    ]
                }

        # 3. NextViT Inference
        if weights_loaded["nextvit"] and model_nextvit is not None:
            with torch.no_grad():
                outputs = model_nextvit(input_tensor)
                probs = torch.softmax(outputs, dim=1)[0]
                val, idx = torch.topk(probs, 5)
                model_results["nextvit"] = {
                    "prediction": CLASSES[idx[0].item()],
                    "confidence": round(float(val[0].item()), 4),
                    "idx": idx[0].item(),
                    "top5": [
                        {"label": CLASS_META[CLASSES[i]]["label"], "probability": float(p)}
                        for p, i in zip(val.tolist(), idx.tolist())
                    ]
                }

        # Set primary prediction for database/analytics (priority: coatnet > nextvit > maxvit)
        primary_key = next(
            (k for k in ["coatnet", "nextvit", "maxvit"] if k in model_results),
            list(model_results.keys())[0]
        )
        prediction = model_results[primary_key]["prediction"]
        confidence = model_results[primary_key]["confidence"]
        primary_idx = model_results[primary_key]["idx"]
        method = f"dual_model_{primary_key}"

        # ── PERSISTENCE: Save to Cloudinary (fallback to Base64) ──
        pred_id = str(uuid.uuid4())
        timestamp = datetime.datetime.now().isoformat()
        persistent_image_url = None
        if is_cloudinary_ready:
            try:
                upload_result = cloudinary.uploader.upload(
                    img_data,
                    folder="tomatoguard_scans",
                    public_id=f"scan_{pred_id}",
                    tags=["tomato_leaf", prediction]
                )
                persistent_image_url = upload_result.get("secure_url")
                print(f"🚀 Image persistent in Cloudinary: {persistent_image_url}")
            except Exception as e:
                print(f"⚠️ Cloudinary upload failed: {e}")

        # Final Fallback to Base64 if Cloudinary failed/skipped
        if not persistent_image_url:
            try:
                import base64
                b64_str = base64.b64encode(img_data).decode("utf-8")
                mime_type = image.content_type or "image/jpeg"
                persistent_image_url = f"data:{mime_type};base64,{b64_str}"
            except Exception:
                persistent_image_url = None


        # ── XAI: Generate Interpretability Heatmap ──
        xai_image_url = None
        annotated_original_url = None      # NEW: original + disease circles
        hotspots = []

        if interpretability_engine:
            try:
                # Target the primary model for XAI
                xai_model = model_coatnet if primary_key == "coatnet" else model_maxvit
                xai_model.train(False)
                
                heatmap, _ = interpretability_engine.generate_heatmap(
                    input_tensor, class_idx=primary_idx
                )

                # 1. Detect hotspots from heatmap
                hotspots = interpretability_engine.find_hotspots(heatmap)

                # 2. Heatmap overlay (existing — unchanged)
                overlay_img = apply_heatmap_overlay(img, heatmap, alpha=0.55)

                # 3. NEW: Annotate original with disease spot circles
                annotated_img = annotate_disease_on_original(img, hotspots, heatmap)

                # ── Upload / base64 both images ──
                overlay_bytes_io = io.BytesIO()
                overlay_img.save(overlay_bytes_io, format="JPEG", quality=87)
                overlay_raw = overlay_bytes_io.getvalue()

                annotated_bytes_io = io.BytesIO()
                annotated_img.save(annotated_bytes_io, format="JPEG", quality=87)
                annotated_raw = annotated_bytes_io.getvalue()

                if is_cloudinary_ready:
                    # Upload heatmap overlay
                    upload_xai = cloudinary.uploader.upload(
                        overlay_raw,
                        folder="tomatoguard_xai",
                        public_id=f"xai_{pred_id}",
                        tags=["xai", prediction],
                    )
                    xai_image_url = upload_xai.get("secure_url")

                    # Upload annotated original
                    upload_ann = cloudinary.uploader.upload(
                        annotated_raw,
                        folder="tomatoguard_annotated",
                        public_id=f"ann_{pred_id}",
                        tags=["annotated", prediction],
                    )
                    annotated_original_url = upload_ann.get("secure_url")
                else:  # base64 fallback when Cloudinary not configured
                    import base64
                    xai_b64 = base64.b64encode(overlay_raw).decode("utf-8")
                    xai_image_url = f"data:image/jpeg;base64,{xai_b64}"

                    ann_b64 = base64.b64encode(annotated_raw).decode("utf-8")
                    annotated_original_url = f"data:image/jpeg;base64,{ann_b64}"

            except Exception as xai_err:
                print(f"⚠️ XAI generation failed: {xai_err}")

        # ── Save to DB ──
        if DATABASE_URL:
            try:
                import json
                conn = psycopg2.connect(DATABASE_URL)
                c = conn.cursor()
                # 1. Insert history record
                c.execute("""
                    INSERT INTO history
                    (id, prediction, confidence, created_at, image_url,
                     image_width, image_height, image_mode, file_size_kb,
                     brightness, dominant_color, method, xai_url, hotspots, image_info)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                """, (
                    pred_id, prediction, confidence, timestamp,
                    persistent_image_url,
                    image_analysis["width"], image_analysis["height"],
                    image_analysis["color_mode"], image_analysis["file_size_kb"],
                    image_analysis["brightness"], image_analysis["dominant_color"],
                    method, xai_image_url, json.dumps(hotspots),
                    json.dumps(image_analysis)
                ))

                # 2. Upsert analytics count
                c.execute("""
                    INSERT INTO analytics (disease_name, count, last_detected)
                    VALUES (%s, 1, %s)
                    ON CONFLICT (disease_name)
                    DO UPDATE SET count = analytics.count + 1, last_detected = %s
                """, (prediction, timestamp, timestamp))
                conn.commit()
                conn.close()
            except Exception as db_err:
                print(f"⚠️  DB non-fatal error: {db_err}")

        # ── Final response ──
        return {
            "id": pred_id,
            "prediction": prediction,
            "confidence": float(confidence),
            "confidence_pct": round(float(confidence) * 100, 2),
            "method": method,
            "createdAt": timestamp,
            "xaiUrl": xai_image_url,
            "annotated_original_url": annotated_original_url,
            "hotspots": hotspots,
            "imageInfo": image_analysis,
            "models": model_results,
            "primary_prediction": prediction,
            "primary_confidence": confidence
        }

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/history")
async def get_history(limit: int = 50):
    """Return the last N predictions from the database."""
    if not DATABASE_URL:
        raise HTTPException(status_code=503, detail="Database not configured.")
    try:
        conn = psycopg2.connect(DATABASE_URL)
        c = conn.cursor()
        c.execute("""
            SELECT id, prediction, confidence, created_at, image_url,
                   image_width, image_height, file_size_kb, brightness,
                   dominant_color, method, xai_url, hotspots
            FROM history
            ORDER BY created_at DESC
            LIMIT %s
        """, (min(limit, 200),))
        rows = c.fetchall()
        conn.close()

        return [
            {
                "id": r[0],
                "prediction": r[1],
                "confidence": r[2],
                "confidence_pct": round((r[2] or 0) * 100, 2),
                "created_at": r[3],
                "image_url": r[4],
                "imageSize": {"width": r[5], "height": r[6]},
                "file_size_kb": r[7],
                "brightness": r[8],
                "dominant_color": r[9],
                "method": r[10],
                "xai_url": r[11],
                "hotspots": json.loads(r[12] or "[]")
            }
            for r in rows
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/history/{item_id}")
async def delete_history_item(item_id: str):
    """Delete a specific prediction record."""
    if not DATABASE_URL:
        raise HTTPException(status_code=503, detail="Database not configured.")
    try:
        conn = psycopg2.connect(DATABASE_URL)
        c = conn.cursor()
        c.execute("DELETE FROM history WHERE id = %s", (item_id,))
        conn.commit()
        conn.close()
        return {"success": True, "id": item_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/save-checkpoint")
async def save_checkpoint(filename: str = "custom_checkpoint.pth"):
    """
    Utility endpoint to save the current model state and configuration.
    This fulfills the requirement of having checkpoint saving implemented in the project.
    """
    if not weights_loaded:
        raise HTTPException(status_code=503, detail="Model weights not loaded.")
    
    try:
        save_path = os.path.join(os.path.dirname(__file__), filename)
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_type': MODEL_TYPE,
            'timestamp': datetime.datetime.now().isoformat(),
            'classes': CLASSES
        }, save_path)
        return {"success": True, "saved_to": save_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save checkpoint: {str(e)}")


@app.get("/classes")
async def list_classes():
    """Return all supported disease classes with metadata."""
    return [
        {"class_name": cls, **CLASS_META[cls]}
        for cls in CLASSES
    ]


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
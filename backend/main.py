from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import torch
from torchvision import models, transforms
from PIL import Image
import io
import uvicorn
import torch.nn as nn
import os
import numpy as np
import base64
import sys
import gc
from contextlib import asynccontextmanager # ADDED: For Lifespan
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# --- IMPORT MEDICAL DATA ---
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from medical_data import MEDICAL_INFO

# --- GLOBAL MODEL STORAGE ---
# Data structures to store models loaded by the lifespan function
ROUTER_MODEL = None
SPECIALIST_MODELS = {} 

# --- Lifespan Function ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Initializes models and resources when the server starts up.
    """
    global ROUTER_MODEL, SPECIALIST_MODELS
    print(f"🚀 Loading AI System onto {DEVICE}...")

    # 1. Load Router
    router = build_router()
    router_path = os.path.join(MODELS_DIR, 'router.pth')
    
    if os.path.exists(router_path):
        try:
            router.load_state_dict(torch.load(router_path, map_location=DEVICE), strict=False) 
            router.to(DEVICE).eval()
            ROUTER_MODEL = router
            print("✅ Router (MobileNetV3) Loaded.")
        except Exception as e:
            print(f"❌ Router Load Failure: {e}")
            
    # 2. Load Specialists
    for idx, info in HIERARCHY.items():
        classes = info['classes']
        model_path = os.path.join(MODELS_DIR, info['model_file'])
        
        # Single Class Case (Adnexal/Eyelid) - Direct Pass Logic
        if len(classes) <= 1:
            SPECIALIST_MODELS[idx] = {'type': 'direct', 'class': classes[0], 'group_name': info['name']}
            print(f"✅ Specialist (Adnexal) Setup: Direct Pass.")
            continue

        # Multi-Class Specialists
        model = build_specialist(len(classes))
        
        if os.path.exists(model_path):
            try:
                model.load_state_dict(torch.load(model_path, map_location=DEVICE))
                model.to(DEVICE).eval()
                SPECIALIST_MODELS[idx] = {'type': 'model', 'model': model, 'classes': classes, 'group_name': info['name']}
                print(f"✅ Specialist ({info['name']}) Loaded.")
            except Exception as e:
                print(f"❌ Specialist Load Failure for {info['name']}: {e}")
        else:
            print(f"❌ Specialist Model Missing: {info['model_file']}")

    yield  # Application is ready to handle requests

    # --- SHUTDOWN EVENT (Unload large models if needed, though not strictly necessary here) ---
    print("👋 Shutting down API.")


app = FastAPI(lifespan=lifespan) # MODIFIED: Initialize app with lifespan

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- CONFIGURATION ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
MODELS_DIR = os.path.join(project_root, "models")

# Use CUDA if available (for PC performance), otherwise CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

# --- HIERARCHY SETUP ---
HIERARCHY = {
    0: {'name': 'Adnexal Oculoplastic', 'model_file': 'specialist_eyelid.pth', 'classes': ['Eyelid']},
    1: {'name': 'Anterior Segment Pathology', 'model_file': 'specialist_anterior.pth', 'classes': ['Cataract', 'Uveitis']},
    2: {'name': 'Ocular Surface Disorders', 'model_file': 'specialist_surface.pth', 'classes': ['Conjunctivitis', 'Jaundice', 'Normal', 'Pterygium']}
}

# --- MODEL ARCHITECTURE HELPERS ---
def build_router():
    # Router uses MobileNetV3 (output is 3 groups)
    model = models.mobilenet_v3_large(weights=None)
    model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, len(HIERARCHY))
    return model

def build_specialist(num_classes):
    # Specialists use EfficientNetB4
    model = models.efficientnet_b4(weights=None)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
    return model

# --- PREPROCESSING ---
preprocess = transforms.Compose([
    transforms.Resize((380, 380)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- ADDED: Root Health Check Endpoint (Fixes 'Not Found' error) ---
@app.get("/")
def read_root():
    if ROUTER_MODEL:
        return {"status": "AI System Ready (Local)", "device": str(DEVICE)}
    else:
        return {"status": "AI System Loading Failed", "error": "Router model is missing or corrupt."}
# -------------------------------------------------------------------

def analyze_symptoms(diagnosis, pain_level, vision_loss, itchiness):
    warnings = []
    if diagnosis == "Conjunctivitis" and pain_level == "Severe":
        warnings.append("⚠️ Pain Mismatch: Severe pain is unusual for Pink Eye. Rule out Glaucoma.")
    if vision_loss == "Yes" and diagnosis in ["Conjunctivitis", "Eyelid"]:
         warnings.append("⚠️ Vision Loss Warning: Eyelid/Surface conditions usually don't affect vision. Check for Keratitis or Uveitis.")
    if itchiness == "Yes" and diagnosis == "Conjunctivitis":
        warnings.append("✅ Symptom Match: Itchiness supports Allergic Conjunctivitis.")
    return warnings

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    pain: str = Form(...),
    vision: str = Form(...),
    itch: str = Form(...)
):
    if ROUTER_MODEL is None: return {"error": "AI System offline. Router failed to load."}

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        input_tensor = preprocess(image).to(DEVICE).unsqueeze(0)

        # 1. STAGE 1: ROUTER INFERENCE
        with torch.no_grad():
            router_out = ROUTER_MODEL(input_tensor)
            router_probs = torch.nn.functional.softmax(router_out[0], dim=0)
            group_idx = torch.argmax(router_probs).item()
            group_conf = router_probs[group_idx].item()
        
        spec_data = SPECIALIST_MODELS.get(group_idx)
        if not spec_data:
             return {"error": "Specialist model for predicted group failed to load."}

        heatmap_base64 = None
        
        # 2. STAGE 2: DIAGNOSIS
        if spec_data['type'] == 'direct':
            diagnosis = spec_data['classes'][0]
            confidence = group_conf * 100
            probs_dict = {diagnosis: 1.0}
            model_for_cam = None
            class_idx_for_cam = 0
            
        else:
            model_for_cam = spec_data['model']
            with torch.no_grad():
                out = model_for_cam(input_tensor)
                probs = torch.nn.functional.softmax(out[0], dim=0)
                class_idx = torch.argmax(probs).item()
            
            diagnosis = spec_data['classes'][class_idx]
            confidence = probs[class_idx].item() * 100
            probs_dict = {spec_data['classes'][i]: float(probs[i].item()) for i in range(len(spec_data['classes']))}
            class_idx_for_cam = class_idx

            # Generate Heatmap
            target_layer = [model_for_cam.features[-1]]
            cam = GradCAM(model=model_for_cam, target_layers=target_layer)
            grayscale = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(class_idx_for_cam)])
            
            rgb_img = np.float32(image.resize((380, 380))) / 255
            vis = show_cam_on_image(rgb_img, grayscale[0, :], use_rgb=True)
            
            buff = io.BytesIO()
            Image.fromarray(vis).save(buff, format="JPEG")
            heatmap_base64 = base64.b64encode(buff.getvalue()).decode("utf-8")

        # 3. HYBRID LOGIC
        hybrid_warnings = analyze_symptoms(diagnosis, pain, vision, itch)
        details = MEDICAL_INFO.get(diagnosis, {}).copy()
        
        if not details:
            details = {"description": "N/A", "severity": "Unknown", "advice": "Consult Doctor", "treatment": [], "symptoms": []}

        if hybrid_warnings:
            details['advice'] = str(details.get('advice', '')) + " " + " ".join(hybrid_warnings)

        return {
            "group_name": spec_data['group_name'],
            "diagnosis": diagnosis,
            "confidence": confidence,
            "heatmap": f"data:image/jpeg;base64,{heatmap_base64}" if heatmap_base64 else None,
            "details": details,
            "hybrid_warnings": hybrid_warnings,
            "probabilities": probs_dict
        }

    except Exception as e:
        print(f"Prediction Error: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    # Note: Uvicorn setup is managed by the application initialization order now
    os.environ['OMP_NUM_THREADS'] = '4' 
    uvicorn.run(app, host="0.0.0.0", port=8000)
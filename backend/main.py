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
import gc 
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from medical_data import MEDICAL_INFO
except ImportError:
    from backend.medical_data import MEDICAL_INFO

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
MODELS_DIR = os.path.join(project_root, "models")
DEVICE = torch.device("cpu") 

HIERARCHY = {
    0: {'name': 'Adnexal Oculoplastic', 'model_file': 'specialist_eyelid.pth', 'path': 'Adnexal Oculoplastic'},
    1: {'name': 'Anterior Segment Pathology', 'model_file': 'specialist_anterior.pth', 'path': 'Anterior Segment Pathology'},
    2: {'name': 'Ocular Surface Disorders', 'model_file': 'specialist_surface.pth', 'path': 'Ocular Surface Disorders'}
}

CLASS_MAP = {
    0: ['Eyelid'],
    1: ['Cataract', 'Uveitis'],
    2: ['Conjunctivitis', 'Jaundice', 'Normal', 'Pterygium']
}

def build_router():
    model = models.mobilenet_v3_large(weights=None)
    model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, len(HIERARCHY))
    return model

def build_specialist(num_classes):
    model = models.efficientnet_b3(weights=None)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
    return model

ROUTER_MODEL = None

@app.on_event("startup")
async def startup_event():
    global ROUTER_MODEL
    print("⏳ Loading Router Model...")
    try:
        router = build_router()
        router_path = os.path.join(MODELS_DIR, 'router.pth')
        if os.path.exists(router_path):
            router.load_state_dict(torch.load(router_path, map_location=DEVICE))
            router.to(DEVICE).eval()
            ROUTER_MODEL = router
            print("✅ Router Loaded (Memory Safe Mode)")
        else:
            print(f"❌ Router missing at {router_path}")
    except Exception as e:
        print(f"❌ Failed to load router: {e}")

def load_specialist_on_demand(group_idx):
    info = HIERARCHY.get(group_idx)
    if not info: return None, None, None

    classes = CLASS_MAP.get(group_idx, [])
    
    if len(classes) <= 1:
        return 'direct', classes[0], info['name']

    print(f"⏳ Lazy Loading Specialist: {info['name']}...")
    model = build_specialist(len(classes))
    spec_path = os.path.join(MODELS_DIR, info['model_file'])
    
    if os.path.exists(spec_path):
        model.load_state_dict(torch.load(spec_path, map_location=DEVICE))
        model.to(DEVICE).eval()
        return model, classes, info['name']
    else:
        print(f"❌ Specialist missing: {info['model_file']}")
        return None, None, None

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def analyze_symptoms(diagnosis, pain_level, vision_loss, itchiness):
    warnings = []
    if diagnosis == "Conjunctivitis" and pain_level == "Severe":
        warnings.append("⚠️ Pain Mismatch: Severe pain is unusual for Pink Eye. Rule out Glaucoma.")
    if vision_loss == "Yes" and diagnosis in ["Conjunctivitis", "Blepharitis", "Hemorrhage"]:
         warnings.append("⚠️ Vision Loss Warning: Surface conditions usually don't affect vision. Check for Uveitis.")
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
    if ROUTER_MODEL is None: return {"error": "Router is offline"}

    gc.collect()

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        input_tensor = preprocess(image).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            router_out = ROUTER_MODEL(input_tensor)
            router_probs = torch.nn.functional.softmax(router_out[0], dim=0)
            group_idx = torch.argmax(router_probs).item()
            group_conf = router_probs[group_idx].item()
        
        specialist, classes, group_name = load_specialist_on_demand(group_idx)
        
        if not specialist:
            return {"error": "Specialist model could not be loaded."}

        heatmap_base64 = None
        probs_dict = {}

        if specialist == 'direct':
            diagnosis = classes
            confidence = group_conf * 100
            probs_dict = {diagnosis: 1.0}
        else:
            with torch.no_grad():
                out = specialist(input_tensor)
                probs = torch.nn.functional.softmax(out[0], dim=0)
                class_idx = torch.argmax(probs).item()
            
            diagnosis = classes[class_idx]
            confidence = probs[class_idx].item() * 100
            probs_dict = {classes[i]: float(probs[i].item()) for i in range(len(classes))}

            try:
                target_layer = [specialist.features[-1]]
                cam = GradCAM(model=specialist, target_layers=target_layer)
                grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(class_idx)])
                rgb_img = np.float32(image.resize((224, 224))) / 255
                vis = show_cam_on_image(rgb_img, grayscale_cam[0, :], use_rgb=True)
                
                buff = io.BytesIO()
                Image.fromarray(vis).save(buff, format="JPEG")
                heatmap_base64 = base64.b64encode(buff.getvalue()).decode("utf-8")
            except Exception as e:
                print(f"Heatmap skipped to save memory: {e}")

            del specialist
            del cam
            gc.collect()

        hybrid_warnings = analyze_symptoms(diagnosis, pain, vision, itch)
        details = MEDICAL_INFO.get(diagnosis, {}).copy()
        
        if not details:
            details = {"description": "N/A", "severity": "Unknown", "advice": "Consult Doctor", "treatment": [], "symptoms": []}

        if hybrid_warnings:
            details['advice'] = str(details.get('advice', '')) + " " + " ".join(hybrid_warnings)

        return {
            "group_name": group_name,
            "diagnosis": diagnosis,
            "confidence": confidence,
            "heatmap": f"data:image/jpeg;base64,{heatmap_base64}" if heatmap_base64 else None,
            "details": details,
            "hybrid_warnings": hybrid_warnings,
            "probabilities": probs_dict
        }

    except Exception as e:
        print(f"Error: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
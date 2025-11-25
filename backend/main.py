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
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from medical_data import MEDICAL_INFO

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
DATA_ROOT = os.path.join(project_root, "dataset")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

HIERARCHY = {
    0: {'name': 'Adnexal Oculoplastic', 'model_file': 'specialist_adnexal.pth', 'path': 'Adnexal Oculoplastic'},
    1: {'name': 'Anterior Segment Pathology', 'model_file': 'specialist_anterior.pth', 'path': 'Anterior Segment Pathology'},
    2: {'name': 'Ocular Surface Disorders', 'model_file': 'specialist_surface.pth', 'path': 'Ocular Surface Disorders'}
}

def load_model_architecture(model_type, num_classes):
    if model_type == 'router':
        model = models.mobilenet_v3_large(weights=None)
        model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, num_classes)
        return model
    else:
        model = models.efficientnet_b3(weights=None)
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
        return model

def load_hierarchical_models():
    print("Loading models...")
    router_model = load_model_architecture('router', len(HIERARCHY))
    router_path = os.path.join(MODELS_DIR, 'router.pth')
    if os.path.exists(router_path):
        router_model.load_state_dict(torch.load(router_path, map_location=DEVICE))
        router_model.to(DEVICE).eval()
    else:
        return None, None

    specialist_models = {}
    for idx, info in HIERARCHY.items():
        spec_path = os.path.join(MODELS_DIR, info['model_file'])
        class_dir = os.path.join(DATA_ROOT, info['path'])
        if os.path.exists(class_dir):
            classes = sorted([d for d in os.listdir(class_dir) if os.path.isdir(os.path.join(class_dir, d))])
            info['classes'] = classes
            model = load_model_architecture('specialist', len(classes))
            if os.path.exists(spec_path):
                model.load_state_dict(torch.load(spec_path, map_location=DEVICE))
                model.to(DEVICE).eval()
                specialist_models[idx] = {'model': model, 'classes': classes, 'group_name': info['name']}
    return router_model, specialist_models

router, specialists = load_hierarchical_models()

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def analyze_symptoms(diagnosis, pain_level, vision_loss, itchiness):
    """Adjusts advice based on user symptoms vs AI diagnosis."""
    warnings = []
    
    if diagnosis == "Conjunctivitis" and pain_level == "Severe":
        warnings.append("⚠️ AI Diagnosis Mismatch: Conjunctivitis typically does not cause severe pain. This could be Glaucoma or Scleritis. Seek urgent care.")
    
    if vision_loss == "Yes" and diagnosis in ["Conjunctivitis", "Blepharitis", "Hemorrhage"]:
         warnings.append("⚠️ Vision Loss Warning: Surface conditions usually don't affect vision. This might be Uveitis or Keratitis.")

    if itchiness == "Yes" and diagnosis == "Conjunctivitis":
        warnings.append("✅ Symptom Match: Itchiness strongly supports Allergic Conjunctivitis.")

    return warnings

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    pain: str = Form(...),  
    vision: str = Form(...), 
    itch: str = Form(...)    
):
    if router is None: return {"error": "AI system offline"}

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        input_tensor = preprocess(image).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            router_out = router(input_tensor)
            group_idx = torch.argmax(router_out).item()
            group_conf = torch.nn.functional.softmax(router_out[0], dim=0)[group_idx].item()

        spec_data = specialists.get(group_idx)
        if not spec_data: return {"error": "Specialist unavailable"}

        with torch.no_grad():
            spec_out = spec_data['model'](input_tensor)
            probs = torch.nn.functional.softmax(spec_out[0], dim=0)
            class_idx = torch.argmax(probs).item()
            
        diagnosis = spec_data['classes'][class_idx]
        confidence = probs[class_idx].item() * 100

        target_layer = [spec_data['model'].features[-1]]
        cam = GradCAM(model=spec_data['model'], target_layers=target_layer)
        grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(class_idx)])
        rgb_img = np.float32(image.resize((224, 224))) / 255
        visualization = show_cam_on_image(rgb_img, grayscale_cam[0, :], use_rgb=True)
        
        buff = io.BytesIO()
        Image.fromarray(visualization).save(buff, format="JPEG")
        heatmap_base64 = base64.b64encode(buff.getvalue()).decode("utf-8")

        hybrid_warnings = analyze_symptoms(diagnosis, pain, vision, itch)
        
        details = MEDICAL_INFO.get(diagnosis, {}).copy()
        if hybrid_warnings:
            details['advice'] += " " + " ".join(hybrid_warnings)

        return {
            "group_name": spec_data['group_name'],
            "diagnosis": diagnosis,
            "confidence": confidence,
            "heatmap": f"data:image/jpeg;base64,{heatmap_base64}",
            "details": details,
            "hybrid_warnings": hybrid_warnings,
            "probabilities": {spec_data['classes'][i]: float(probs[i].item()) for i in range(len(spec_data['classes']))}
        }

    except Exception as e:
        print(f"Error: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
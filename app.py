# app.py
import streamlit as st
import torch
import torch.nn.functional as F
from torchvision.models.detection import FasterRCNN
import torch.serialization as serialization
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as TF
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Download model
@st.cache_resource
def load_wbc_model():
    return load_model("models/resnet50_wbc_model-v1.h5")

@st.cache_resource
def load_bcd_model():
    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    in_feats = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_feats, 3)
    with serialization.safe_globals([FasterRCNN]):
        model.load_state_dict(torch.load("models/faster_rcnn_bcd_model-v1.pth", map_location="cpu", weights_only=True))
    model.eval()
    return model

wbc_model = load_wbc_model()
bcd_model = load_bcd_model()

IMAGE_SIZE = 224
CLASS_NAMES = {1: "rbc", 2: "wbc", 0: "background"}
wbc_class_labels = {0: 'Basophil', 1: 'Eosinophil', 2: 'Lymphocyte', 3: 'Monocyte', 4: 'Neutrophil'}

# helper function
def preprocess_image(image_file):
    image = Image.open(image_file).convert("RGB")
    tensor = TF.to_tensor(image).unsqueeze(0)
    return image, tensor

def predict(model, image_tensor, threshold=0.5):
    with torch.no_grad():
        preds = model(image_tensor)
    return preds[0]

def plot_predictions(image, predictions, threshold=0.5, margin=0.1):
    fig, ax = plt.subplots(1, figsize=(12, 9))
    ax.imshow(image)
    img_w, img_h = image.size
    stats = {'RBC': 0, 'WBC:Basophil': 0, 'WBC:Eosinophil': 0, 
             'WBC:Lymphocyte': 0, 'WBC:Monocyte': 0, 'WBC:Neutrophil': 0, 'UNK': 0}

    for box, score, label in zip(predictions['boxes'], predictions['scores'], predictions['labels']):
        if score >= threshold:
            class_name = CLASS_NAMES.get(label.item(), "unknown")
            xmin, ymin, xmax, ymax = box.cpu().numpy()
            width, height = xmax - xmin, ymax - ymin
            xmin = max(0, xmin - margin * width)
            ymin = max(0, ymin - margin * height)
            xmax = min(img_w, xmax + margin * width)
            ymax = min(img_h, ymax + margin * height)

            if class_name == "wbc":
                crop = image.crop((int(xmin), int(ymin), int(xmax), int(ymax)))
                crop_np = np.array(crop)
                crop_resized = cv2.resize(crop_np, (IMAGE_SIZE, IMAGE_SIZE))
                crop_array = np.expand_dims(crop_resized / 255.0, axis=0)
                pred = wbc_model.predict(crop_array)
                subclass = np.argmax(pred)
                subclass_name = wbc_class_labels[subclass]
                stats[f"WBC:{subclass_name}"] += 1

                rect = patches.Rectangle((xmin, ymin), width, height, linewidth=2, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
                ax.text(xmin, ymin-10, f"{class_name}:{subclass_name} ({pred[0][subclass]:.2f})",
                        color="red", fontsize=12, bbox=dict(facecolor="yellow", alpha=0.5))
            else:
                if class_name == "rbc":
                    stats["RBC"] += 1
                else:
                    stats["UNK"] += 1
                rect = patches.Rectangle((xmin, ymin), width, height, linewidth=2, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
                ax.text(xmin, ymin-10, f"{class_name}:{score:.2f}", 
                        color="red", fontsize=12, bbox=dict(facecolor="yellow", alpha=0.5))

    return fig, stats

# ---------------- Streamlit UI ----------------
st.title("🩸 Blood Cell Analysis to Assess the Risk of Disease in Patients")

uploaded_file = st.file_uploader("Upload blood cells photo(.jpg, .png)", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image, image_tensor = preprocess_image(uploaded_file)
    preds = predict(bcd_model, image_tensor)

    fig, stats = plot_predictions(image, preds)
    st.pyplot(fig)

    st.subheader("📊 Detection statistics")
    st.json(stats)

    # วิเคราะห์การติดเชื้อแบคทีเรีย (Monocyte)
    wbc_counts = {k: v for k, v in stats.items() if k.startswith("WBC:")}
    monocyte_count = wbc_counts.get("WBC:Monocyte", 0)
    other_counts = {k: v for k, v in wbc_counts.items() if k != "WBC:Monocyte"}
    is_high = all(monocyte_count > c for c in other_counts.values())

    if is_high and monocyte_count > 0:
        st.error("⚠️ Analysis: Possible bacterial infection (Monocyte count higher than other WBC types)")
    else:
        st.success("✅ Analysis: No clear evidence of bacterial infection detected")

    # วิเคราะห์การติดเชื้อพยาธิ (Eosinophil)
    eosinophil_count = wbc_counts.get("WBC:Eosinophil", 0)
    other_counts = {k: v for k, v in wbc_counts.items() if k != "WBC:Eosinophil"}
    is_high = all(eosinophil_count > c for c in other_counts.values())

    if is_high and eosinophil_count > 0:
        st.error("⚠️ Analysis: Possible parasitic infection (Eosinophil count higher than other WBC types)")
    else:
        st.success("✅ Analysis: No clear evidence of parasitic infection detected")

    # วิเคราะห์ภูมิแพ้ (Basophil)
    basophil_count = wbc_counts.get("WBC:Basophil", 0)
    other_counts = {k: v for k, v in wbc_counts.items() if k != "WBC:Basophil"}
    is_high = all(basophil_count > c for c in other_counts.values())

    if is_high and basophil_count > 0:
        st.error("⚠️ Analysis: Possible allergic condition (Basophil count higher than other WBC types)")
    else:
        st.success("✅ Analysis: No clear evidence of allergic condition detected")

          
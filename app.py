"""
DermoScope AI — Intelligent Skin Lesion Analysis
=================================================
Run:  streamlit run app.py
Requires `skin_lesion_resnet18.pt` next to this file.
Optional: `examples/` folder with sample dermatoscopy images.
"""

import os
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms, models
import matplotlib.cm as cm

# ----------------------------------------------------------------------
# Clinical metadata
# ----------------------------------------------------------------------
CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
CLASS_NAMES = {
    "akiec": "Actinic keratoses",
    "bcc":   "Basal cell carcinoma",
    "bkl":   "Benign keratosis",
    "df":    "Dermatofibroma",
    "mel":   "Melanoma",
    "nv":    "Melanocytic nevus (benign mole)",
    "vasc":  "Vascular lesion",
}
CLASS_INFO = {
    "akiec": ("🟠", "Pre-cancerous", "#e67e22",
              "Rough, scaly patches caused by sun damage. A fraction can evolve into squamous cell carcinoma — worth a dermatologist's look."),
    "bcc":   ("🔴", "Malignant", "#c0392b",
              "The most common skin cancer. Slow-growing and rarely spreads, but requires medical removal."),
    "bkl":   ("🟢", "Benign", "#1e8449",
              "Harmless age-related keratosis. Can visually resemble melanoma, which is why professional confirmation matters."),
    "df":    ("🟢", "Benign", "#1e8449",
              "A firm, fibrous nodule, often on the legs, frequently following a minor injury or insect bite."),
    "mel":   ("🔴", "Malignant", "#c0392b",
              "The most dangerous skin cancer. Highly treatable when detected early — urgent dermatological review is essential."),
    "nv":    ("🟢", "Benign", "#1e8449",
              "A common mole — a benign growth of pigment-producing cells."),
    "vasc":  ("🟢", "Benign", "#1e8449",
              "Blood-vessel lesion (angioma). Typically red or purple in appearance."),
}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
MODEL_PATH = "skin_lesion_resnet18.pt"
EXAMPLES_DIR = "examples"

# ----------------------------------------------------------------------
# Page setup & clinical design system
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="DermoScope AI",
    page_icon="🩺",
    layout="wide",
)

st.markdown("""
<style>
    .stApp { background-color: #f6fafb; }
    #MainMenu, footer { visibility: hidden; }

    .hero {
        background: linear-gradient(120deg, #045c66 0%, #028090 60%, #02a891 100%);
        border-radius: 16px; padding: 2rem 2.4rem; color: white;
        margin-bottom: 1.2rem;
    }
    .hero h1 { color: white !important; margin: 0; font-size: 2.3rem; }
    .hero p  { margin: 0.4rem 0 0; opacity: 0.92; font-size: 1.05rem; }

    h2, h3 { color: #045c66 !important; }
    section[data-testid="stSidebar"] { background-color: #eaf4f5; }

    .diagnosis-card {
        background: white; border-radius: 14px;
        padding: 1.4rem 1.7rem;
        box-shadow: 0 2px 10px rgba(4,92,102,0.10);
    }
    .severity-pill {
        display: inline-block; padding: 0.15rem 0.75rem;
        border-radius: 999px; color: white;
        font-size: 0.85rem; font-weight: 600;
        vertical-align: middle; margin-left: 0.4rem;
    }
    .prob-row { margin-bottom: 0.45rem; }
    .prob-bar-bg { background: #e2eef0; border-radius: 6px; height: 13px; width: 100%; }
    .prob-bar-fill {
        background: linear-gradient(90deg, #028090, #02C39A);
        border-radius: 6px; height: 13px;
    }
    .footer-credit {
        text-align: center; color: #7d949a; font-size: 0.85rem;
        margin-top: 2.2rem;
    }
    .disclaimer {
        background: #fdf6ec; border-left: 4px solid #e67e22;
        border-radius: 8px; padding: 0.7rem 1rem;
        color: #6b5b45; font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------
# Model loading
# ----------------------------------------------------------------------
@st.cache_resource
def load_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, len(CLASSES)),
    )
    state = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


# ----------------------------------------------------------------------
# Grad-CAM — visual explanation of the AI's focus
# ----------------------------------------------------------------------
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.activations, self.gradients = None, None
        target_layer.register_forward_hook(self._save_act)
        target_layer.register_full_backward_hook(self._save_grad)

    def _save_act(self, m, i, o):
        self.activations = o.detach()

    def _save_grad(self, m, gi, go):
        self.gradients = go[0].detach()

    def __call__(self, x, class_idx):
        logits = self.model(x)
        self.model.zero_grad()
        logits[0, class_idx].backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.activations).sum(dim=1))
        cam = cam / (cam.max() + 1e-8)
        return cam[0].numpy()


def overlay_heatmap(pil_img, cam, alpha):
    cam_img = Image.fromarray(np.uint8(cam * 255)).resize(pil_img.size)
    heat = cm.jet(np.array(cam_img) / 255.0)[..., :3]
    base = np.array(pil_img).astype(float) / 255.0
    blend = (1 - alpha) * base + alpha * heat
    return Image.fromarray(np.uint8(blend * 255))


# ----------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🩺 DermoScope AI")
    st.caption("Intelligent skin lesion analysis")

    st.markdown("""
**How it works**
1. Upload a dermatoscopy image
2. The AI analyses the lesion
3. Review the result and the AI's visual focus map
    """)

    st.subheader("Display settings")
    cam_alpha = st.slider("Focus map opacity", 0.0, 0.9, 0.45, 0.05)

    st.divider()
    st.markdown(
        "**Team**  \nAmira Ouaked  \nManal Soulane"
    )

# ----------------------------------------------------------------------
# Main page
# ----------------------------------------------------------------------
st.markdown("""
<div class="hero">
  <h1>🩺 DermoScope AI</h1>
  <p>AI-assisted analysis of skin lesions across 7 diagnostic categories —
  with a visual focus map showing exactly what the AI examined.</p>
</div>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="disclaimer">⚕️ DermoScope AI is an educational screening-support '
    'demo and <b>not a medical device</b>. Results are not a diagnosis. '
    'Always consult a dermatologist for any skin concern.</div>',
    unsafe_allow_html=True,
)
st.write("")

if not os.path.exists(MODEL_PATH):
    st.error(
        f"Model file `{MODEL_PATH}` not found — place it in the same folder as `app.py`."
    )
    st.stop()

model = load_model()
gradcam = GradCAM(model, model.layer4[-1])

# --- Input: upload OR example gallery ---------------------------------
img = None
upload_col, gallery_col = st.columns([2, 1])

with upload_col:
    uploaded = st.file_uploader(
        "Upload a dermatoscopy image (JPG/PNG)", type=["jpg", "jpeg", "png"]
    )
    if uploaded:
        img = Image.open(uploaded).convert("RGB")

with gallery_col:
    if os.path.isdir(EXAMPLES_DIR):
        examples = sorted(
            f for f in os.listdir(EXAMPLES_DIR)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        )
        if examples:
            choice = st.selectbox("…or try a sample image", ["—"] + examples)
            if choice != "—":
                img = Image.open(os.path.join(EXAMPLES_DIR, choice)).convert("RGB")

# --- Inference & display ----------------------------------------------
if img is not None:
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    x = tf(img).unsqueeze(0)

    with st.spinner("Analysing lesion…"):
        with torch.no_grad():
            probs = F.softmax(model(x), dim=1)[0]
        order = torch.argsort(probs, descending=True)
        top_idx = int(order[0])
        top_class = CLASSES[top_idx]
        badge, severity, sev_color, info = CLASS_INFO[top_class]
        cam = gradcam(x, top_idx)

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Your image")
        st.image(img, use_container_width=True)
    with c2:
        st.subheader("AI focus map")
        st.image(overlay_heatmap(img, cam, cam_alpha), use_container_width=True)
        st.caption(
            "Warm regions are the areas the AI examined most closely for this result."
        )

    st.divider()
    st.subheader("Analysis result")
    st.markdown(
        f"""
        <div class="diagnosis-card">
        <h2 style="margin:0">{badge} {CLASS_NAMES[top_class]}
        <span class="severity-pill" style="background:{sev_color}">{severity}</span></h2>
        <p style="margin:0.5rem 0 0.3rem; font-size:1.05rem;">
        Confidence: <b>{probs[top_idx].item():.1%}</b></p>
        <p style="margin:0; color:#5a6b70;">{info}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if probs[top_idx].item() < 0.5:
        st.info(
            "ℹ️ The analysis is uncertain — several categories are close. "
            "This case would especially benefit from expert review."
        )

    st.write("")
    st.markdown("#### Most likely categories")
    for rank in range(3):
        i = int(order[rank])
        c = CLASSES[i]
        p = probs[i].item()
        st.markdown(
            f"""
            <div class="prob-row">
              <span>{CLASS_INFO[c][0]} <b>{CLASS_NAMES[c]}</b> — {p:.1%}</span>
              <div class="prob-bar-bg">
                <div class="prob-bar-fill" style="width:{max(p*100,2):.0f}%"></div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with st.expander("See all 7 categories"):
        st.bar_chart(
            {CLASS_NAMES[c]: float(probs[i]) for i, c in enumerate(CLASSES)}
        )

st.markdown(
    '<p class="footer-credit">DermoScope AI · Developed by '
    'Amira Ouaked &amp; Manal Soulane · 2026</p>',
    unsafe_allow_html=True,
)

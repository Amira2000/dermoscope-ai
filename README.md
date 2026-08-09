# 🩺 DermoScope AI — Skin Lesion Classification with Explainable Deep Learning

**Live demo:** https://dermoscope-ai.streamlit.app/

A deep learning application that classifies dermatoscopy images of skin
lesions into **7 diagnostic categories** and shows *where the model looked*
using **Grad-CAM** explainability.



> ⚕️ **Disclaimer:** DermoScope AI is an educational project, not a medical
> device. Its results are not a diagnosis. Always consult a dermatologist
> for any skin concern.


## 🎯 The Problem

Skin cancer is one of the most common cancers worldwide. Melanoma is deadly
when detected late but highly treatable when caught early — and screening is
a visual task, making it a natural fit for computer vision. Our goal:
classify dermatoscopy images into 7 diagnostic categories with a model that
can **justify its predictions visually**, because in medicine, a black box
is not enough.

## 📊 The Dataset

**[HAM10000](https://doi.org/10.7910/DVN/DBW86T)** — 10,015 clinical
dermatoscopy images collected over ~20 years at the Medical University of
Vienna (Austria) and a skin-cancer practice in Queensland (Australia).
Published by Tschandl, Rosendahl & Kittler (2018) in *Scientific Data*;
training set of the ISIC 2018 challenge.

Why we chose it:
- **Clinical-grade labels** — over 50% of diagnoses confirmed by biopsy
  (histopathology), the rest by follow-up, expert consensus, or confocal
  microscopy
- **Rich metadata** (`lesion_id`, age, sex, body site) enabling rigorous
  methodology
- **A real challenge**: severe class imbalance (67% benign nevi) and
  visually similar classes (melanoma vs benign moles)

## 🔬 Approach — every choice driven by the data

| EDA finding | Design decision |
|---|---|
| 67% of images are benign nevi | **Class-weighted CrossEntropyLoss** `N/(K·n_c)` |
| Multiple photos of the same lesion | **Split by `lesion_id`** — leakage-free evaluation |
| Only 10k images | **Transfer learning**: ImageNet-pretrained ResNet-18 |
| Lesions have no natural orientation | Flip/rotation augmentation |
| Color is a diagnostic signal | Only *mild* ColorJitter |
| mel/nv visually similar | Evaluate with per-class recall + confusion matrix |

**Architecture:** ResNet-18 backbone with a custom head
`Linear(512→256) → ReLU → Dropout(0.4) → Linear(256→7)`

**Training:** two phases — (1) head only, backbone frozen, 3 epochs @ lr
1e-3; (2) fine-tune `layer4` + head, 5 epochs @ lr 1e-4, cosine schedule,
AdamW, best-epoch checkpointing.

## 📈 Results

| Metric | Value |
|---|---|
| Validation accuracy (lesion-level split) | **72.3%** |
| Weighted avg F1 | **0.745** |
| Melanoma recall | **0.569** |
| BCC (carcinoma) recall | **0.770** |

Context: a naive model predicting "benign mole" for everything scores 67%
accuracy and catches **zero** cancers. Our class-weighted training
deliberately trades precision for **recall on malignant classes** — in a
screening context, a false alarm costs a consultation; a missed melanoma
costs a life.

The confusion matrix confirms the failure mode our EDA predicted: melanoma
and benign nevi (the same cell type) account for the main confusion — a
distinction dermatologists themselves find difficult.

## 🖥️ The App

Streamlit application featuring:
- Image upload + one-click sample gallery
- Prediction with severity badge (malignant / pre-cancerous / benign) and
  confidence
- **AI focus map (Grad-CAM)** — a heatmap of the regions that drove the
  prediction, with adjustable opacity
- Top-3 probability breakdown and low-confidence flagging

## 📁 Repository Structure

```
├── 01_EDA.ipynb              # Exploratory data analysis (run first)
├── 02_Training.ipynb         # Model training pipeline (Colab, ~25 min on T4)
├── app.py                    # DermoScope AI — Streamlit application
├── skin_lesion_resnet18.pt   # Trained model weights
├── examples/                 # Sample dermatoscopy images for the demo
├── requirements.txt
└── README.md
```

## 🚀 Run It Yourself

```bash
git clone https://github.com/YOUR_USERNAME/dermoscope-ai.git
cd dermoscope-ai
pip install -r requirements.txt
streamlit run app.py
```

To retrain: open `02_Training.ipynb` in Google Colab (T4 GPU runtime),
run all cells (~25 min), download the generated `skin_lesion_resnet18.pt`.

## ⚠️ Limitations & Future Work

- Dataset patients are predominantly light-skinned (Austria/Australia) —
  generalization to darker skin tones is unverified, a known fairness
  issue in dermatology AI
- Trained on dermatoscope-quality images; not suited to smartphone photos
- Rare classes (df: 115 images, vasc: 142) have limited support
- Next steps: larger backbone (EfficientNet), oversampling/focal loss,
  test-time augmentation, external validation on ISIC data

## 📚 References & AI Assistance

- Tschandl, P., Rosendahl, C. & Kittler, H. *The HAM10000 dataset*,
  Scientific Data 5, 180161 (2018)
- Selvaraju, R. R. et al. *Grad-CAM: Visual Explanations from Deep
  Networks*, ICCV 2017


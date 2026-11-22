"""Knuckle liveness check: Streamlit test app for the three replay-fine-tuned models.

Setup (once):
    pip install -r requirements.txt
    unzip knuckle_models.zip into weights/       (from section 4 of Knuckle_Replay_Test_Colab.ipynb)
    git clone --depth 1 https://github.com/gurayozgur/TPO TPO_repo     (only for FoundPAD)
        If TPO_repo/pretrained/foundpad_tpo.pth is only a few hundred bytes, run `git lfs pull` inside TPO_repo.
Run:
    streamlit run app.py
"""
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
import torch
from peft import LoraConfig, inject_adapter_in_model
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torchvision.models import mobilenet_v3_large
from torchvision.transforms import v2

HERE = Path(__file__).parent
WEIGHTS, CAPTURES, TPO_REPO = HERE / "weights", HERE / "captures", HERE / "TPO_repo"
LOG = CAPTURES / "log.csv"
CONFIG = json.loads((WEIGHTS / "config.json").read_text())
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class L2Norm(nn.Module):              # DINOv2's head was trained on length-1 features
    def forward(self, x):
        return F.normalize(x, dim=1)


def make_model(name):
    # the same architectures as the notebook
    if name == "MobileNetV3":
        model = mobilenet_v3_large()
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 2)
        return model
    if name == "DINOv2 + LoRA":
        backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        lora = LoraConfig(r=8, lora_alpha=8, lora_dropout=0.4, use_rslora=True, target_modules=["qkv"])
        backbone = inject_adapter_in_model(lora, backbone)
        return nn.Sequential(backbone, L2Norm(), nn.Linear(backbone.embed_dim, 2))
    sys.path.insert(0, str(TPO_REPO / "src"))
    from models import load_detector       # the TPO authors' code
    model, _ = load_detector(str(TPO_REPO / "pretrained" / "foundpad_tpo.pth"), device=str(DEVICE))
    return model


@st.cache_resource(show_spinner="Loading model…")
def load_model(name):
    model = make_model(name)
    state = torch.load(WEIGHTS / CONFIG["models"][name]["file"], map_location="cpu", weights_only=True)
    # MobileNet file = whole model; DINOv2 / FoundPAD files = only the fine-tuned LoRA + head, loaded on top
    missing, unexpected = model.load_state_dict(state, strict=(name == "MobileNetV3"))
    assert not unexpected, f"{name}: checkpoint doesn't match the model: {unexpected[:3]}"
    return model.to(DEVICE).eval()


def available(name):
    if not (WEIGHTS / CONFIG["models"][name]["file"]).exists():
        return False
    return name != "FoundPAD (CLIP)" or (TPO_REPO / "pretrained" / "foundpad_tpo.pth").exists()


@torch.inference_mode()
def p_real(name, images):
    settings, side, size = CONFIG["models"][name], CONFIG["resize_first"], CONFIG["input_size"]
    preprocess = v2.Compose([v2.ToImage(), v2.Resize(size), v2.CenterCrop(size), v2.ToDtype(torch.float32, scale=True),
                             v2.Normalize(settings["mean"], settings["std"])])
    x = torch.stack([preprocess(img.convert("RGB").resize((side, side))) for img in images]).to(DEVICE)
    return torch.softmax(load_model(name)(x).float(), dim=1)[:, 1].cpu().numpy()


st.set_page_config(page_title="Knuckle liveness", page_icon="🖐️", layout="wide")
st.title("Knuckle liveness check")
st.caption("Real hand, or a replay on a screen? Show the back of the hand with the whole fingers in view.")

names = [n for n in CONFIG["models"] if available(n)]
missing_models = [n for n in CONFIG["models"] if n not in names]
with st.sidebar:
    chosen = st.multiselect("Models", names, default=names)
    st.caption("Threshold on P(real): at or above it counts as LIVE.")
    thresholds = {n: st.slider(n, 0.0, 1.0, float(CONFIG["models"][n]["threshold"]), 0.01) for n in chosen}
    if missing_models:
        st.info("Not available: " + ", ".join(missing_models) + ". See the setup steps at the top of app.py.")

camera, upload = st.tabs(["📷 Camera", "📁 Upload 1–5 frames of one attempt"])
shot = camera.camera_input("Take a photo")
files = upload.file_uploader("Frames", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
images = [Image.open(shot)] if shot else [Image.open(f) for f in (files or [])[:5]]

if images and chosen:
    for col, img in zip(st.columns(len(images)), images):
        col.image(img)
    rows = []
    for name in chosen:
        scores = p_real(name, images)
        score = float(scores.mean())
        rows.append({"model": name, "P(real)": round(score, 3), "threshold": round(thresholds[name], 3),
                     "verdict": "✅ LIVE" if score >= thresholds[name] else "⛔ ATTACK",
                     "per frame": ", ".join(f"{s:.3f}" for s in scores)})
    results = pd.DataFrame(rows)
    st.dataframe(results, hide_index=True)

    kind = st.radio("What was actually shown? (saved to captures/ for testing)", ["real", "print", "replay"], horizontal=True)
    if st.button("Save attempt"):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        (CAPTURES / kind).mkdir(parents=True, exist_ok=True)
        for i, img in enumerate(images):
            img.convert("RGB").save(CAPTURES / kind / f"{stamp}_{i}.png")      # PNG: no extra compression
        log = results[["model", "P(real)"]].rename(columns={"P(real)": "p_real"}).assign(time=stamp, true_kind=kind)
        log.to_csv(LOG, mode="a", header=not LOG.exists(), index=False)
        st.toast(f"Saved as {kind}")

if LOG.exists() and chosen:
    log = pd.read_csv(LOG)
    summary = []
    for name in chosen:
        part = log[log.model == name]
        if len(part):
            accepted = part.p_real >= thresholds[name]
            attack, real = part.true_kind != "real", part.true_kind == "real"
            summary.append({"model": name, "attempts": len(part),
                            "attacks accepted (APCER)": f"{accepted[attack].mean():.0%}" if attack.any() else "–",
                            "real rejected (BPCER)": f"{(~accepted[real]).mean():.0%}" if real.any() else "–"})
    if summary:
        st.subheader("Your labelled attempts (at the current thresholds)")
        st.dataframe(pd.DataFrame(summary), hide_index=True)

"""Streamlit research demo: pneumothorax segmentation (AA-HybridUNet / baseline U-Net).

Run:   streamlit run app.py
Layout next to this file:   aa_pneumo/ (code)   models/<model_name>/{weights.pt, model_config.json, operating_point.json}
RESEARCH PROTOTYPE - not a clinically validated diagnostic device. No fake predictions: if a model cannot be loaded, an error is shown.
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import streamlit as st

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "src"))
import app_logic as L  # noqa: E402

st.set_page_config(page_title="Pneumothorax segmentation (research prototype)", page_icon="🫁", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 1.4rem; max-width: 1400px;}
.warnbox {border-left: 5px solid #d9480f; background: #fff4e6; color:#5f370e; padding: .7rem 1rem; border-radius: 6px; font-size: .92rem;}
@media (prefers-color-scheme: dark) {.warnbox {background:#3a2a1a; color:#ffd8a8;}}
</style>
""", unsafe_allow_html=True)
st.title("🫁 Pneumothorax segmentation")
# st.markdown('<div class="warnbox"><b>Research prototype - not a medical device.</b> Outputs are not clinically validated and must not be used '
#             'for diagnosis or treatment decisions. Models were trained on a public research dataset (SIIM-ACR) and may fail on images that differ from it.</div>',
#             unsafe_allow_html=True)

MODELS_DIR = Path(os.environ.get("AA_MODELS_DIR", HERE / "models"))
models = L.discover_models(MODELS_DIR)
if not models:
    st.error(f"No models found in `{MODELS_DIR}`. Unzip `deployment_bundle.zip` (from the training notebook) so that this folder contains "
             "`models/<name>/weights.pt`, `model_config.json`, `operating_point.json`, and an `aa_pneumo/` folder next to `app.py`.")
    st.stop()


@st.cache_resource(show_spinner="Loading model ...")
def get_model(folder: str):
    from aa_pneumo.infer import load_bundle      # strict state-dict load: an incompatible checkpoint raises instead of guessing
    return load_bundle(folder)


with st.sidebar:
    st.header("Settings")
    label = {k: f"{k}  ({v['kind']})" for k, v in models.items()}
    name = st.selectbox("Model", list(models), format_func=lambda k: label[k])
    info = models[name]
    try:
        model, meta, op, device = get_model(str(info["path"]))
    except Exception as e:  # noqa: BLE001
        st.error(f"Could not load model '{name}': {e}")
        st.stop()
    st.caption(f"run: `{info['run']}` · input {meta.get('img_size', 512)}px · device: `{device}`")
    use_pp = st.toggle("Use tuned post-processing", True, help="Drops tiny regions using the minimum-area settings tuned on the validation split.")
    thr = st.slider("Probability threshold", 0.05, 0.95, float(op.get("thr", 0.5)), 0.01,
                    help=f"Validation-tuned default: {op.get('thr', 0.5)}")
    alpha = st.slider("Overlay opacity", 0.0, 1.0, 0.45, 0.05)
    show_outline = st.checkbox("Draw outline", True)
    if info["test"]:
        t = info["test"]
        st.divider(); st.caption("Official-test metrics of this model (clean images, from the training notebook)")
        st.caption(f"Dice (lesion images) {t.get('dice_pos', float('nan')):.3f} · Dice (all) {t.get('dice_all', float('nan')):.3f} · "
                   f"Sensitivity {t.get('sensitivity', float('nan')):.3f} · Precision {t.get('precision', float('nan')):.3f}")

up = st.file_uploader("Upload a chest X-ray (PNG / JPG / DICOM)", type=[e.strip(".") for e in sorted(L.ALLOWED)])
if up is None:
    st.info("Upload a frontal chest X-ray to run the model.")
    st.stop()

try:
    gray = L.decode_upload(up.name, up.getvalue())
except L.InputError as e:
    st.error(str(e)); st.stop()

from aa_pneumo.infer import predict_image  # noqa: E402
try:
    with st.spinner("Running inference ..."):
        o = dict(op) if use_pp else dict(thr=thr, min_comp=0, min_img=0)
        o["thr"] = thr
        r = predict_image(model, meta, o, device, gray)
except Exception as e:  # noqa: BLE001
    st.error(f"Inference failed: {e}"); st.stop()

mask, prob = r["mask_full"], r["prob_full"]
stats = L.mask_stats(mask, prob)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Inference time", f"{r['inference_ms']:.0f} ms"); c2.metric("Predicted area", f"{stats['area_pct']:.2f} %", help="Share of the image area covered by the predicted mask")
c3.metric("Regions", stats["n_regions"]); c4.metric("Max probability", f"{stats['max_prob']:.2f}")
if stats["area_px"] == 0:
    st.success("No pneumothorax region predicted at this threshold (this is a model output, not a diagnosis).")

tabs = ["Overlay", "Probability map", "Original"] + (["Artifact map"] if r["artifact_prob"] is not None else [])
tt = st.tabs(tabs)
ov = L.overlay(gray, mask, alpha, outline=show_outline)
with tt[0]:
    st.image(ov, caption="Predicted pneumothorax (red) on the uploaded image", use_container_width=True)
with tt[1]:
    st.image(L.heatmap(gray, prob), caption="Pixel probability (before thresholding)", use_container_width=True)
with tt[2]:
    st.image(gray, caption=f"Uploaded image ({gray.shape[1]}x{gray.shape[0]})", use_container_width=True, clamp=True)
if r["artifact_prob"] is not None:
    with tt[3]:
        import cv2
        ap = cv2.resize(r["artifact_prob"], (gray.shape[1], gray.shape[0]))
        st.image(L.heatmap(gray, ap), caption="Predicted device/artifact map (trained on SYNTHETIC devices only; may miss real ones)", use_container_width=True)

d1, d2 = st.columns(2)
d1.download_button("⬇ Download mask (PNG, 0/255)", L.png_bytes((mask * 255).astype(np.uint8)), f"{Path(up.name).stem}_mask.png", "image/png")
d2.download_button("⬇ Download overlay (PNG)", L.png_bytes(ov), f"{Path(up.name).stem}_overlay.png", "image/png")
st.caption("Research use only. Not a clinically validated diagnostic device.")

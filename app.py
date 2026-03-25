"""
app.py — HAM10000 Skin Lesion Classifier
Streamlit web application.  Run with:  streamlit run app.py
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import streamlit as st
import plotly.graph_objects as go
from PIL import Image
from streamlit_cropper import st_cropper

# ---------------------------------------------------------------------------
# Import inference utilities (fail gracefully if checkpoints are absent)
# ---------------------------------------------------------------------------
try:
    import inference_utils as iu

    _IMPORT_OK = True
except FileNotFoundError as _e:
    _IMPORT_OK = False
    _IMPORT_ERROR = str(_e)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    layout="wide",
    page_title="HAM10000 Classifier",
    page_icon="🔬",
)

# ---------------------------------------------------------------------------
# Custom CSS — clean, professional, minimal
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Remove default top padding */
    .block-container { padding-top: 1.5rem; }

    /* Result card */
    .result-card {
        border-radius: 12px;
        padding: 1.4rem 1.8rem;
        margin-bottom: 1rem;
    }
    .result-card-red  { background: rgba(217, 54, 62, 0.12); border-left: 6px solid #d9363e; }
    .result-card-amber { background: rgba(249, 168, 37, 0.12); border-left: 6px solid #f9a825; }
    .result-card-green { background: rgba(46, 125, 50, 0.12); border-left: 6px solid #2e7d32; }

    .card-title { font-size: 1.75rem; font-weight: 700; margin-bottom: 0.2rem; }
    .card-subtitle { font-size: 1.1rem; opacity: 0.7; margin-bottom: 0.5rem; }
    .badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        color: white;
        margin-top: 0.4rem;
    }
    .badge-red   { background: #d9363e; }
    .badge-amber { background: #f9a825; color: #1a1a1a; }
    .badge-blue  { background: #1565c0; }

    /* Model cards */
    .model-card {
        border: 1px solid rgba(128, 128, 128, 0.3);
        border-radius: 10px;
        padding: 1rem 1.1rem;
        margin-bottom: 0.5rem;
    }
    .model-name { font-size: 1rem; font-weight: 700; }
    .model-pred { font-size: 0.9rem; opacity: 0.7; margin-bottom: 0.4rem; }

    /* Probability table */
    .prob-table { font-size: 0.78rem; }

    /* Crop preview panel */
    .crop-preview-label {
        font-size: 0.78rem;
        font-weight: 600;
        opacity: 0.6;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.3rem;
    }
    .crop-hint {
        font-size: 0.8rem;
        opacity: 0.55;
        margin-top: 0.5rem;
        line-height: 1.5;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
HIGH_RISK_LABELS = {"mel", "bcc", "akiec"}
CHECKPOINT_KEYS = {
    "HAMCNN": "processed/best_hamcnn.pth",
    "EnhancedCNN": "processed/best_enhanced_cnn.pth",
    "DoubleCNN": "processed/best_double_cnn.pth",
    "ViT-DINOv2": "processed/best_vit_dinov2.pth",
    "EfficientNetB3": "processed/best_efficientnet_b3.pth",
}
ENSEMBLE_WEIGHT_NOTE = (
    "Weights: HAMCNN=0.03 · EnhancedCNN=0.03 · DoubleCNN=0.14 "
    "· ViT-DINOv2=0.60 · EfficientNetB3=0.20"
)

# Colour palette for the 7 classes (alphabetical order: akiec bcc bkl df mel nv vasc)
CLASS_COLORS = {
    "akiec": "#f9a825",  # amber — high risk
    "bcc": "#f9a825",  # amber — high risk
    "bkl": "#78909c",  # blue-grey — benign
    "df": "#78909c",
    "mel": "#d9363e",  # red — highest risk
    "nv": "#78909c",
    "vasc": "#78909c",
}


def _bar_color(label: str) -> str:
    return CLASS_COLORS.get(label, "#78909c")


def _card_class(label: str) -> str:
    if label == "mel":
        return "result-card-red"
    if label in {"bcc", "akiec"}:
        return "result-card-amber"
    return "result-card-green"


def _badge_class(label: str) -> str:
    if label == "mel":
        return "badge-red"
    if label in {"bcc", "akiec"}:
        return "badge-amber"
    return "badge-blue"


# ---------------------------------------------------------------------------
# Cached model loading
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def _load_models():
    """Load all five models once and cache across sessions."""
    return iu.load_all_models(verbose=False)


# ---------------------------------------------------------------------------
# Plotly chart helpers
# ---------------------------------------------------------------------------
def _prob_chart(result_entry: dict, show_ci: bool = False) -> go.Figure:
    """Return a horizontal bar chart (Plotly) for a single model/ensemble entry."""
    r = result_entry
    labels = [iu.IDX_TO_LABEL[i] for i in range(iu.CONFIG["num_classes"])]
    names = [iu.CLASS_NAMES[lbl] for lbl in labels]
    probs = r["probs"]

    # Sort descending by probability
    order = np.argsort(probs)[::-1]
    labels = [labels[i] for i in order]
    names = [names[i] for i in order]
    probs = probs[order]
    colors = [_bar_color(lbl) for lbl in labels]

    fig = go.Figure()

    if show_ci:
        ci_low = r["ci_low"][order]
        ci_high = r["ci_high"][order]
        error_x = dict(
            type="data",
            array=ci_high - probs,
            arrayminus=probs - ci_low,
            visible=True,
            color="#666",
            thickness=1.5,
            width=4,
        )
    else:
        error_x = None

    fig.add_trace(
        go.Bar(
            x=probs,
            y=names,
            orientation="h",
            marker_color=colors,
            error_x=error_x,
            hovertemplate="%{y}: %{x:.3f}<extra></extra>",
        )
    )

    fig.update_layout(
        margin=dict(l=0, r=10, t=4, b=4),
        xaxis=dict(
            range=[0, 1],
            showgrid=False,
            tickformat=".0%",
            title="Probability",
            title_font_size=11,
        ),
        yaxis=dict(showgrid=False, autorange="reversed"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=220,
        font=dict(size=11),
        showlegend=False,
    )
    return fig


def _prob_table_md(result_entry: dict) -> str:
    """Return a Markdown table of class | prob | 95% CI."""
    r = result_entry
    header = "| Class | Prob | CI 95% |\n|---|---|---|\n"
    rows = []
    for i in range(iu.CONFIG["num_classes"]):
        lbl = iu.IDX_TO_LABEL[i]
        name = iu.CLASS_NAMES[lbl]
        p = r["probs"][i]
        lo = r["ci_low"][i]
        hi = r["ci_high"][i]
        rows.append(f"| {name} | {p:.3f} | [{lo:.3f}, {hi:.3f}] |")
    return header + "\n".join(rows)


# ---------------------------------------------------------------------------
# Clinical high-risk warning
# ---------------------------------------------------------------------------
def _maybe_show_warning(results: dict) -> None:
    """Show st.error if any model predicts a high-risk class with conf > 0.4."""
    triggered = False
    for entry in results.values():
        if entry["pred_label"] in HIGH_RISK_LABELS and entry["confidence"] > 0.4:
            triggered = True
            break
    if triggered:
        st.error(
            "⚠️ High-risk lesion type detected. "
            "This tool is for research only — please consult a qualified dermatologist."
        )


# ---------------------------------------------------------------------------
# Simple View
# ---------------------------------------------------------------------------
def _render_simple(pil_img: Image.Image, results: dict, models: dict) -> None:
    ens = results["Ensemble"]
    pred_label = ens["pred_label"]
    pred_name = ens["pred_name"]
    confidence = ens["confidence"]

    # --- Result card ---
    card_cls = _card_class(pred_label)
    badge_cls = _badge_class(pred_label)
    risk_text = (
        "High-risk class" if pred_label in HIGH_RISK_LABELS else "Low-risk class"
    )

    st.markdown(
        f"""
        <div class="result-card {card_cls}">
            <div class="card-title">{pred_name}</div>
            <div class="card-subtitle">Ensemble confidence: <strong>{confidence:.1%}</strong></div>
            <span class="badge {badge_cls}">{risk_text}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- Probability bar ---
    st.markdown("#### Ensemble class probabilities")
    st.plotly_chart(_prob_chart(ens, show_ci=False), use_container_width=True)

    # --- Grad-CAM ---
    st.markdown("#### Activation map (Grad-CAM · EfficientNetB3)")
    with st.spinner("Computing Grad-CAM…"):
        overlay, _ = iu.get_gradcam_overlay(pil_img, models["EfficientNetB3"])

    col_orig, col_cam = st.columns(2)
    with col_orig:
        st.image(pil_img, caption="Uploaded image", use_container_width=True)
    with col_cam:
        st.image(overlay, caption="Grad-CAM overlay", use_container_width=True)

    with st.expander("How to read this map"):
        st.write(iu.GRADCAM_EXPLANATION)

    # --- Disclaimer ---
    st.warning(iu.DISCLAIMER)


# ---------------------------------------------------------------------------
# Expert View
# ---------------------------------------------------------------------------
def _model_card(col, model_name: str, entry: dict, show_ci: bool = True) -> None:
    with col:
        st.markdown(
            f"""<div class="model-card">
                <div class="model-name">{model_name}</div>
                <div class="model-pred">
                    {entry["pred_name"]} &mdash; {entry["confidence"]:.1%}
                </div>
            </div>""",
            unsafe_allow_html=True,
        )
        st.plotly_chart(_prob_chart(entry, show_ci=show_ci), use_container_width=True)
        with st.expander("Probability table"):
            st.markdown(_prob_table_md(entry), unsafe_allow_html=False)


def _render_expert(pil_img: Image.Image, results: dict, models: dict) -> None:
    # Reuse simple view for the ensemble top card + grad-cam
    _render_simple(pil_img, results, models)

    st.divider()
    st.markdown("## Per-model results")

    # Row 1: HAMCNN · EnhancedCNN · DoubleCNN
    row1 = st.columns(3)
    for col, name in zip(row1, ["HAMCNN", "EnhancedCNN", "DoubleCNN"]):
        _model_card(col, name, results[name])

    # Row 2: ViT-DINOv2 · EfficientNetB3 — centred via blank padding columns
    _, c1, c2, _ = st.columns([1, 2, 2, 1])
    for col, name in zip([c1, c2], ["ViT-DINOv2", "EfficientNetB3"]):
        _model_card(col, name, results[name])

    # Ensemble panel
    st.divider()
    st.markdown("## Ensemble")
    st.plotly_chart(
        _prob_chart(results["Ensemble"], show_ci=True), use_container_width=True
    )
    with st.expander("Ensemble probability table"):
        st.markdown(_prob_table_md(results["Ensemble"]))
    st.caption(ENSEMBLE_WEIGHT_NOTE)

    # Agreement indicator
    top_preds = [results[m]["pred_label"] for m in iu.MODEL_NAMES]
    ens_label = results["Ensemble"]["pred_label"]
    agreement = sum(1 for lbl in top_preds if lbl == ens_label)
    st.metric(
        label="Model agreement (top-1 class)",
        value=f"{agreement} / {len(iu.MODEL_NAMES)} models agree: {iu.CLASS_NAMES[ens_label]}",
    )


# ===========================================================================
# Main layout
# ===========================================================================

# --- Sidebar ---
with st.sidebar:
    st.title("🔬 HAM10000 Classifier")
    st.caption("Dermoscopic lesion classification · Research use only")

    st.divider()

    view_mode = st.radio(
        "View mode",
        options=["Simple View", "Expert View"],
        index=0,
        horizontal=False,
    )

    st.divider()

    n_tta = st.slider(
        label="TTA passes (speed ↔ precision)",
        min_value=4,
        max_value=16,
        value=8,
        step=4,
        help="More passes produce tighter 95% confidence intervals at the cost of longer inference time.",
    )

    st.divider()

    # Device info
    if _IMPORT_OK:
        dev = iu.device
        if dev.type == "cuda":
            import torch

            gpu_name = torch.cuda.get_device_name(0)
            st.info(f"🖥 Device: **CUDA** — {gpu_name}")
        else:
            st.info("🖥 Device: **CPU**")

    st.divider()

    # Author info
    st.divider()
    st.markdown("**Author**")
    st.markdown(
        """
        **Giuseppe Di Pasquale**
        <br>✉️ [1.dipasquale.giuseppe@gmail.com](mailto:1.dipasquale.giuseppe@gmail.com)
        <br>🐙 [github.com/username](https://github.com/Dipa1999)
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    # Checkpoint status
    st.markdown("**Checkpoint files**")
    for label, path_str in CHECKPOINT_KEYS.items():
        exists = Path(path_str).exists()
        icon = "✓" if exists else "✗"
        color = "green" if exists else "red"
        st.markdown(
            f"<span style='color:{color};font-weight:600'>{icon}</span> "
            f"`{path_str}`",
            unsafe_allow_html=True,
        )

# --- Handle import failure ---
if not _IMPORT_OK:
    st.error(
        f"**Failed to import `inference_utils`:** {_IMPORT_ERROR}\n\n"
        "Make sure all checkpoint `.pth` files are present in the `processed/` "
        "directory and that `inference_utils.py` is in the same folder as `app.py`."
    )
    st.stop()

# --- Main panel ---
st.markdown("## Skin Lesion Classification")
st.markdown(
    "Upload a dermoscopic image, adjust the **crop square** to frame the lesion, "
    "then click **Analyse**."
)

uploaded = st.file_uploader(
    label="Upload dermoscopic image",
    type=["jpg", "jpeg", "png", "tif", "tiff"],
    accept_multiple_files=False,
)

if uploaded is not None:
    pil_full = Image.open(io.BytesIO(uploaded.read())).convert("RGB")
    w_orig, h_orig = pil_full.size

    st.divider()

    # -----------------------------------------------------------------------
    # Crop selector
    # -----------------------------------------------------------------------
    st.markdown("### 1 · Select the region of interest")
    st.markdown(
        "<p class='crop-hint'>"
        "🖱 <b>Drag</b> the crop box to reposition it &nbsp;·&nbsp; "
        "<b>Scroll</b> inside the image to zoom in/out &nbsp;·&nbsp; "
        "The selection is always <b>square</b> (1 : 1) — what you see is exactly "
        "what enters the model at 224 × 224 px."
        "</p>",
        unsafe_allow_html=True,
    )

    col_crop, col_preview = st.columns([3, 1], gap="large")

    with col_crop:
        # st_cropper renders an interactive Cropper.js widget.
        # realtime_update=True → preview on the right refreshes while dragging.
        # aspect_ratio=(1,1)  → force square selection.
        # box_color           → crop-box border colour (matches our palette).
        _cropper_key = f"cropper_{uploaded.name}_{uploaded.size}"
        cropped_img: Image.Image = st_cropper(
            pil_full,
            realtime_update=True,
            box_color="#1565c0",
            aspect_ratio=(1, 1),
            return_type="image",
            key=_cropper_key,
        )

    with col_preview:
        st.markdown(
            "<p class='crop-preview-label'>Preview (224 × 224)</p>",
            unsafe_allow_html=True,
        )
        # Show a downscaled version of what will enter the model
        thumb = cropped_img.resize((224, 224), Image.LANCZOS)
        st.image(thumb, use_container_width=True)

        # Metadata
        cw, ch = cropped_img.size
        st.markdown(
            f"<p class='crop-hint'>"
            f"Original: {w_orig} × {h_orig} px<br>"
            f"Crop: {cw} × {ch} px<br>"
            f"TTA passes: {n_tta}"
            f"</p>",
            unsafe_allow_html=True,
        )

    st.divider()

    # -----------------------------------------------------------------------
    # Analyse button
    # -----------------------------------------------------------------------
    st.markdown("### 2 · Run classification")
    analyse = st.button("🔍 Analyse", type="primary")

    if analyse:
        # Load models (cached — runs only once per session)
        with st.spinner("Loading models…"):
            try:
                models = _load_models()
            except FileNotFoundError as e:
                st.error(
                    f"**Checkpoint not found:** {e}\n\n"
                    "Please place all `.pth` files inside the `processed/` directory."
                )
                st.stop()

        # Run inference on the cropped region (not the full photo)
        with st.spinner("Running inference…"):
            results = iu.predict_single(cropped_img, models, n_tta=n_tta)

        # Clinical warnings (both views)
        _maybe_show_warning(results)

        st.divider()

        if view_mode == "Simple View":
            _render_simple(cropped_img, results, models)
        else:
            _render_expert(cropped_img, results, models)

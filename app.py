import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
from pathlib import Path
import random
import zipfile
import io
import cv2

# =====================================================
# Page Config
# =====================================================
st.set_page_config(
    page_title="Knife Detection System",
    layout="wide"
)

st.title("🔪 Knife Detection System")
st.markdown(
    "YOLOv8 vs YOLOv11 — pixel-exact knife detection "
    "(no resizing, no color modification)"
)

# =====================================================
# Paths
# =====================================================
PRE_IMAGES_DIR = Path("pre_images")
MODEL_8_PATH = "best_8s.pt"
MODEL_11_PATH = "best_11s.pt"

# =====================================================
# Load Models (Cached)
# =====================================================
@st.cache_resource
def load_models():
    return {
        "YOLOv8-S": YOLO(MODEL_8_PATH),
        "YOLOv11-S": YOLO(MODEL_11_PATH),
    }

models = load_models()

# =====================================================
# Sidebar Controls
# =====================================================
st.sidebar.header("Configuration")

conf_threshold = st.sidebar.slider(
    "Confidence Threshold",
    0.1, 1.0, 0.30, 0.05
)

iou_threshold = st.sidebar.slider(
    "NMS IoU Threshold",
    0.1, 0.9, 0.40, 0.05
)

max_det = st.sidebar.selectbox(
    "Max Detections Per Image",
    [1, 2, 3],
    index=0
)

image_source = st.sidebar.radio(
    "Image Source",
    ["Sample Images", "Upload Images"]
)

compare_mode = st.sidebar.checkbox(
    "A/B Compare YOLOv8 vs YOLOv11",
    value=True
)

selected_model_name = None
if not compare_mode:
    selected_model_name = st.sidebar.radio(
        "Select Detection Model",
        ["YOLOv8-S", "YOLOv11-S"]
    )

# =====================================================
# Helpers (UI-ONLY thumbnails)
# =====================================================
def make_thumbnail(pil_img, max_size=220):
    thumb = pil_img.copy()
    thumb.thumbnail((max_size, max_size))  # UI only
    return thumb

# =====================================================
# Load Sample Images
# =====================================================
sample_images = []
if PRE_IMAGES_DIR.exists():
    sample_images = sorted([
        p for p in PRE_IMAGES_DIR.iterdir()
        if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
    ])

selected_images = []

# =====================================================
# Sample Image Gallery (Uniform Thumbnails — UI ONLY)
# =====================================================
if image_source == "Sample Images":

    st.subheader("Sample Image Gallery (Uniform Thumbnails)")

    if not sample_images:
        st.warning("No images found in pre_images/")
    else:
        selected = []
        cols = st.columns(4)

        for idx, img_path in enumerate(sample_images):
            with cols[idx % 4]:
                img = Image.open(img_path)
                thumb = make_thumbnail(img)

                st.image(
                    thumb,
                    caption=img_path.name,
                    use_container_width=False
                )

                if st.checkbox(
                    "Select",
                    key=f"select_{img_path.name}"
                ):
                    selected.append(img_path)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🎲 Random Image"):
                selected = [random.choice(sample_images)]
        with col2:
            if st.button("📦 Select All"):
                selected = sample_images

        selected_images = selected

# =====================================================
# Upload Images (NO resizing, NO color change)
# =====================================================
else:
    uploaded_files = st.file_uploader(
        "Upload image(s)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )
    selected_images = uploaded_files

# =====================================================
# Inference
# =====================================================
if selected_images:

    st.subheader("Detection Results")

    zip_buffer = io.BytesIO()
    zip_file = zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED)

    for img_item in selected_images:

        # Load original image (unchanged)
        if isinstance(img_item, Path):
            image = Image.open(img_item)
            img_name = img_item.name
            file_size = img_item.stat().st_size / 1024
        else:
            image = Image.open(img_item)
            img_name = img_item.name
            file_size = img_item.size / 1024

        image_np = np.array(image)

        # Metadata
        st.markdown(f"### 🖼️ {img_name}")
        st.markdown(
            f"""
            **Resolution:** {image.width} × {image.height}  
            **Format:** {image.format}  
            **File Size:** {file_size:.1f} KB
            """
        )

        # -------------------------------------------------
        # A/B Comparison Mode
        # -------------------------------------------------
        if compare_mode:
            cols = st.columns(3)

            with cols[0]:
                st.markdown("**Original**")
                st.image(image, use_container_width=False)

            for idx, (name, model) in enumerate(models.items(), start=1):
                results = model.predict(
                    source=image_np,
                    conf=conf_threshold,
                    iou=iou_threshold,
                    max_det=max_det,
                    agnostic_nms=True,
                    verbose=False
                )

                annotated = results[0].plot()  # already RGB

                with cols[idx]:
                    st.markdown(f"**{name}**")
                    st.image(annotated, use_container_width=False)

                _, buf = cv2.imencode(".jpg", annotated)
                zip_file.writestr(f"{name}_{img_name}", buf.tobytes())

        # -------------------------------------------------
        # Single Model Mode
        # -------------------------------------------------
        else:
            model = models[selected_model_name]

            cols = st.columns(2)
            with cols[0]:
                st.markdown("**Original**")
                st.image(image, use_container_width=False)

            results = model.predict(
                source=image_np,
                conf=conf_threshold,
                iou=iou_threshold,
                max_det=max_det,
                agnostic_nms=True,
                verbose=False
            )

            annotated = results[0].plot()  # already RGB

            with cols[1]:
                st.markdown(f"**{selected_model_name}**")
                st.image(annotated, use_container_width=False)

            _, buf = cv2.imencode(".jpg", annotated)
            zip_file.writestr(f"{selected_model_name}_{img_name}", buf.tobytes())

        st.divider()

    zip_file.close()
    zip_buffer.seek(0)

    st.download_button(
        "⬇️ Download Annotated Results (ZIP)",
        data=zip_buffer,
        file_name="knife_detection_results.zip",
        mime="application/zip"
    )

else:
    st.info("Select sample images or upload images to begin detection.")

import streamlit as st
from transformers import BLIPProcessor, BLIPForConditionalGeneration
from PIL import Image
import torch
import io
import zipfile

# ---------------------------------------------------------
# App Configuration
# ---------------------------------------------------------
st.set_page_config(page_title="BLIP Caption Generator",
                   page_icon="🖼️",
                   layout="centered")

st.markdown("""
    <style>
        .big-dropzone {
            border: 3px dashed #4B9CD3;
            padding: 40px;
            border-radius: 12px;
            text-align: center;
            font-size: 18px;
            color: #FFFFFF88;
            transition: all 0.2s ease;
        }
        .big-dropzone:hover {
            border-color: #76c7ff;
            background-color: #1a1f26;
            color: #FFFFFF;
        }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Load BLIP Model
# ---------------------------------------------------------
@st.cache_resource
def load_blip():
    processor = BLIPProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BLIPForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

processor, model = load_blip()

# ---------------------------------------------------------
# UI Header
# ---------------------------------------------------------
st.title("🖼️ BLIP Image Caption Generator")
st.write("Drop images → Generate captions → Download zipped dataset.")

# ---------------------------------------------------------
# Trigger Word Input
# ---------------------------------------------------------
trigger = st.text_input(
    "Optional Trigger Word (used for LoRA datasets)",
    placeholder="example: sbanks25"
)

# ---------------------------------------------------------
# File Upload Section
# ---------------------------------------------------------
st.markdown('<div class="big-dropzone">Drag & drop up to 60 images</div>', unsafe_allow_html=True)

uploaded_files = st.file_uploader(
    " ",
    type=["png", "jpg", "jpeg", "webp"],
    accept_multiple_files=True,
)

# enforce max 60 images
if uploaded_files:
    if len(uploaded_files) > 60:
        st.error("You can upload a maximum of 60 images.")
        st.stop()
    else:
        st.success(f"Loaded {len(uploaded_files)} images ✔️")

# ---------------------------------------------------------
# Process Button
# ---------------------------------------------------------
if st.button("Generate Captions", type="primary"):
    if not uploaded_files:
        st.error("Upload at least one image first.")
        st.stop()

    # Create zip buffer
    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:

        for file in uploaded_files:
            image = Image.open(file).convert("RGB")
            st.image(image, caption=file.name, use_column_width=True)

            # Generate caption
            inputs = processor(images=image, return_tensors="pt")
            with torch.no_grad():
                output = model.generate(**inputs)
                caption = processor.decode(output[0], skip_special_tokens=True)

            # Add trigger token before caption
            if trigger.strip():
                final_caption = f"{trigger.strip()}. {caption}"
            else:
                final_caption = caption

            # ---- Save image ----
            img_bytes = io.BytesIO()
            image.save(img_bytes, format="PNG")
            zipf.writestr(f"images/{file.name}", img_bytes.getvalue())

            # ---- Save caption with matching filename ----
            txt_name = file.name.rsplit(".", 1)[0] + ".txt"
            zipf.writestr(f"captions/{txt_name}", final_caption)

    st.success("Captions generated successfully! 🎉")

    # Download ZIP
    st.download_button(
        label="Download Dataset ZIP",
        data=zip_buffer.getvalue(),
        file_name="captions_dataset.zip",
        mime="application/zip"
    )

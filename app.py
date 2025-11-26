import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
import io
import zipfile

# -------------------------------------------------
# Page Config
# -------------------------------------------------
st.set_page_config(
    page_title="Image Caption Generator",
    layout="centered",
)

# -------------------------------------------------
# Load BLIP Model Once
# -------------------------------------------------
@st.cache_resource
def load_blip():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

processor, model = load_blip()

# -------------------------------------------------
# Custom CSS (drag highlight)
# -------------------------------------------------
st.markdown(
    """
    <style>
        .stFileUploader:hover {
            border: 2px solid #4B9CD3 !important;
            background-color: #1a2027 !important;
            transition: 0.2s ease;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------------------------------
# UI Title
# -------------------------------------------------
st.title("Image Caption Generator")
st.write("Upload one or more images and generate BLIP captions.")

# -------------------------------------------------
# Trigger Word
# -------------------------------------------------
trigger = st.text_input(
    "Optional Trigger Word (for LoRA training)",
    value=""
)

# -------------------------------------------------
# File Uploader — MULTIPLE FILES
# -------------------------------------------------
uploaded_files = st.file_uploader(
    "Drag and drop up to 60 images here",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

# -------------------------------------------------
# Generate Button
# -------------------------------------------------
run_button = st.button("Generate Captions")

# -------------------------------------------------
# CAPTION GENERATION
# -------------------------------------------------
captions_output = []

if run_button:
    if not uploaded_files:
        st.error("Please upload at least one image.")
    else:
        # Limit for safety (as requested)
        uploaded_files = uploaded_files[:60]

        for file in uploaded_files:
            image = Image.open(file).convert("RGB")

            # Show image preview
            st.image(image, width=250)

            # Create caption
            inputs = processor(images=image, return_tensors="pt")
            with torch.no_grad():
                output = model.generate(**inputs)
                caption = processor.decode(output[0], skip_special_tokens=True)

            # Add trigger
            if trigger.strip():
                final_caption = f"{trigger.strip()}. {caption}"
            else:
                final_caption = caption

            # Save for ZIP
            captions_output.append((file.name, final_caption))

            # Show caption UI block
            st.subheader(f"Caption for: {file.name}")
            st.markdown(
                f"""
                <div style="
                    background-color:#173a2f;
                    padding:15px;
                    border-radius:10px;
                    color:white;
                    font-size:16px;
                ">
                    {final_caption}
                </div>
                """,
                unsafe_allow_html=True
            )

        # -------------------------------------------------
        # ZIP DOWNLOAD BUTTON
        # -------------------------------------------------
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zipf:
            for filename, cap in captions_output:
                txt_name = filename.rsplit('.', 1)[0] + ".txt"
                zipf.writestr(txt_name, cap)

        st.download_button(
            label="Download Captions as ZIP",
            data=zip_buffer.getvalue(),
            file_name="captions.zip",
            mime="application/zip"
        )

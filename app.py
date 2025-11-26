import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

# ---------------------------------------------------------
# Load BLIP model
# ---------------------------------------------------------
@st.cache_resource
def load_blip():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

processor, model = load_blip()

# ---------------------------------------------------------
# Page Setup
# ---------------------------------------------------------
st.set_page_config(page_title="Image Caption Generator", layout="centered")

st.markdown(
    """
    <style>
        .uploadbox:hover {
            border: 2px solid #4B9CD3 !important;
            background-color: #1a2027 !important;
            transition: 0.2s ease;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Title
# ---------------------------------------------------------
st.title("Image Caption Generator")
st.write("Upload an image and generate a BLIP caption.")

# ---------------------------------------------------------
# Trigger Word Input
# ---------------------------------------------------------
trigger = st.text_input("Optional Trigger Word (for LoRA datasets)", value="")

# ---------------------------------------------------------
# Image Upload
# ---------------------------------------------------------
uploaded_file = st.file_uploader(
    "Drag and drop an image here",
    type=["jpg", "jpeg", "png"],
    help="Upload a single image for captioning."
)

# ---------------------------------------------------------
# Caption Generation Button
# ---------------------------------------------------------
if uploaded_file and st.button("Generate Caption"):

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Process image using BLIP
    inputs = processor(image, return_tensors="pt")
    out = model.generate(**inputs)
    blip_caption = processor.decode(out[0], skip_special_tokens=True)

    # Apply trigger word if provided
    if trigger.strip():
        final_caption = f"{trigger.strip()}. {blip_caption}"
    else:
        final_caption = blip_caption

    # Display
    st.subheader("Generated Caption")
    st.markdown(
        f"""
        <div style="
            background-color:#1f3b25;
            padding:15px;
            border-radius:10px;
            border:1px solid #2e5d3f;
            color:white;
        ">
        {final_caption}
        </div>
        """,
        unsafe_allow_html=True,
    )

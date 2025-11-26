import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

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
# UI Title
# -------------------------------------------------
st.title("Image Caption Generator")

st.write("Upload **one or more images** and click *Generate Captions*.")

# -------------------------------------------------
# File Uploader — multiple images
# -------------------------------------------------
uploaded_files = st.file_uploader(
    "Drag and drop images here",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

# Trigger word input
trigger = st.text_input("Optional trigger token (e.g., kstat25)", value="")

# Button
run_button = st.button("Generate Captions")

# -------------------------------------------------
# Generate Captions
# -------------------------------------------------
if run_button and uploaded_files:
    for file in uploaded_files:

        image = Image.open(file).convert("RGB")

        # Display image
        st.image(image, width=300)

        # Generate caption
        inputs = processor(images=image, return_tensors="pt")

        with torch.no_grad():
            output = model.generate(**inputs)
            caption = processor.decode(output[0], skip_special_tokens=True)

        # Apply trigger token formatting
        if trigger.strip():
            final_caption = f"{trigger.strip()}. {caption}"
        else:
            final_caption = caption

        # Display results
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
            unsafe_allow_html=True,
        )

elif run_button and not uploaded_files:
    st.error("Please upload at least one image.")

import streamlit as st
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

st.set_page_config(page_title="Image Caption Generator", layout="centered")

# ------------------------------
# Custom CSS (bigger box + glow)
# ------------------------------
st.markdown("""
<style>

    /* Base uploader box (bigger + padding) */
    div[data-testid="stFileUploader"] > section {
        padding: 40px !important;
        border: 3px dashed #666 !important;
        border-radius: 14px !important;
        transition: all 0.25s ease-out;
        background-color: #1f1f1f10;
    }

    /* Glow on hover */
    div[data-testid="stFileUploader"]:hover > section {
        border-color: #4fa3ff !important;
        box-shadow: 0px 0px 18px #4fa3ff55;
        transform: scale(1.02);
    }

    /* Drag-over glow (simulated via focus-within) */
    div[data-testid="stFileUploader"] > section:focus-within {
        border-color: #00eaff !important;
        box-shadow: 0px 0px 25px #00eaffaa;
        background-color: #1f1f1f22;
        transform: scale(1.03);
    }

</style>
""", unsafe_allow_html=True)

# ------------------------------
# Load BLIP Model
# ------------------------------
@st.cache_resource
def load_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
    return processor, model

processor, model = load_model()

st.title("🖼️ Image Caption Generator")
st.write("Clean, simple, WAN-ready captions for your datasets.")

# ------------------------------
# Detail Level Selector
# ------------------------------
detail = st.selectbox(
    "Caption detail level:",
    ["Basic", "Descriptive", "Highly Detailed"],
)

# Settings for each mode
settings = {
    "Basic": {
        "max_length": 20,
        "num_beams": 1,
        "temperature": 1.0,
        "repetition_penalty": 1.0
    },
    "Descriptive": {
        "max_length": 40,
        "num_beams": 3,
        "temperature": 0.8,
        "repetition_penalty": 1.1
    },
    "Highly Detailed": {
        "max_length": 80,
        "num_beams": 5,
        "temperature": 0.7,
        "repetition_penalty": 1.2
    }
}

# Optional custom trigger token
trigger = st.text_input("Custom trigger token (optional):", placeholder="e.g., sbanks25")

# ------------------------------
# Upload Area
# ------------------------------
uploaded_file = st.file_uploader(
    "Drag & drop your image here",
    type=["jpg", "jpeg", "png"]
)

# ------------------------------
# Caption Generation
# ------------------------------
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded image", use_column_width=True)

    if st.button("Generate Caption", use_container_width=True):
        with st.spinner("Generating caption..."):
            s = settings[detail]

            inputs = processor(image, return_tensors="pt")
            output = model.generate(
                **inputs,
                max_length=s["max_length"],
                num_beams=s["num_beams"],
                temperature=s["temperature"],
                repetition_penalty=s["repetition_penalty"]
            )

            caption = processor.decode(output[0], skip_special_tokens=True)

            if trigger.strip() != "":
                caption = f"{trigger.strip()}: {caption}"

        st.success("Caption generated!")
        st.write("### Caption:")
        st.write(caption)

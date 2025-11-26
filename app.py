import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
import base64

# -------------------------------
# Load model once
# -------------------------------
@st.cache_resource
def load_blip():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
    return processor, model

processor, model = load_blip()

# -------------------------------
# Custom CSS Styles
# -------------------------------
st.markdown("""
<style>

[data-testid="stFileUploader"] section {
    border: 2px dashed #4B9CD3 !important;
    padding: 2.5rem !important;
    border-radius: 14px;
    transition: background-color 0.3s ease, border-color 0.3s ease;
}

[data-testid="stFileUploader"] section:hover {
    background-color: #112233 !important;
    border-color: #77C3FF !important;
}

.caption-box {
    background-color: #193b27;
    padding: 1.2rem;
    border-radius: 10px;
    color: white;
    font-size: 1.1rem;
    line-height: 1.5rem;
    border: 1px solid #2c5942;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# Caption generator logic
# -------------------------------
def generate_caption(image, mode, length, trigger_token):
    # Step 1 — Raw BLIP caption
    inputs = processor(image, return_tensors="pt")
    out = model.generate(**inputs, max_new_tokens=60)
    caption = processor.decode(out[0], skip_special_tokens=True)

    # --------------------------
    # Apply Caption Mode
    # --------------------------
    if mode == "Simple / Direct":
        pass

    elif mode == "Descriptive":
        caption = caption + ". Add more visual descriptive detail."

    elif mode == "Character LoRA (WAN 2.2 Style)":
        caption = (
            caption
            + ". Describe only physical appearance. No emotions. No actions. No story."
        )

    # --------------------------
    # Apply Caption Length
    # --------------------------
    if length == "Short (under 10 words)":
        caption = " ".join(caption.split()[:10])

    elif length == "Medium":
        pass

    elif length == "Long":
        caption = caption + ". Provide additional clear physical attributes."

    # --------------------------
    # Trigger Token
    # --------------------------
    trigger_token = trigger_token.strip()
    if trigger_token:
        caption = f"{trigger_token} — {caption}"

    return caption


# -------------------------------
# UI Layout
# -------------------------------
st.title("📷 Image Caption Generator")
st.write("Upload an image and generate WAN 2.2–friendly captions for dataset creation.")

uploaded_file = st.file_uploader(
    "Drag and drop or browse an image",
    type=["jpg", "jpeg", "png"]
)

# Sidebar controls
st.sidebar.header("Caption Settings")

caption_mode = st.sidebar.selectbox(
    "Caption Mode",
    ["Simple / Direct", "Descriptive", "Character LoRA (WAN 2.2 Style)"]
)

caption_length = st.sidebar.selectbox(
    "Caption Length",
    ["Short (under 10 words)", "Medium", "Long"]
)

trigger_token = st.sidebar.text_input(
    "Optional Trigger Word",
    placeholder="e.g., kstat25"
)

# -------------------------------
# Process Image
# -------------------------------
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Generate Caption"):
        with st.spinner("Generating caption…"):
            final_caption = generate_caption(image, caption_mode, caption_length, trigger_token)

        st.subheader("Generated Caption")
        st.markdown(f"<div class='caption-box'>{final_caption}</div>", unsafe_allow_html=True)

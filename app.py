import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

# ---------------------------
# Load CSS
# ---------------------------
def load_css():
    with open("styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# ---------------------------
# Load BLIP model
# ---------------------------
@st.cache_resource
def load_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
    return processor, model

processor, model = load_model()

# ---------------------------
# Caption generation function
# ---------------------------
def generate_caption(image, mode, length, trigger_token):
    text_prompt = ""

    # Mode instructions
    if mode == "Simple / Direct":
        text_prompt = "a clear simple caption of the image"
    elif mode == "Descriptive":
        text_prompt = "a detailed descriptive caption of the image"
    elif mode == "Character LoRA (WAN 2.2 Style)":
        text_prompt = (
            "describe the person in the image focusing on physical attributes only, "
            "no emotions, no actions, no story, pure objective appearance"
        )

    # Length instructions
    if length == "Short":
        length_prompt = "short caption, under 10 words"
    elif length == "Medium":
        length_prompt = "medium-length caption"
    else:
        length_prompt = "long detailed caption"

    final_prompt = f"{text_prompt}, {length_prompt}"

    # Trigger token (optional)
    if trigger_token.strip():
        final_prompt = f"{trigger_token.strip()} — {final_prompt}"

    inputs = processor(image, final_prompt, return_tensors="pt")
    out = model.generate(**inputs, max_new_tokens=40)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption


# ---------------------------
# UI
# ---------------------------
st.title("📸 Image Caption Generator")
st.write("Drag an image below and generate WAN 2.2–style captions quickly.")

uploaded_image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

# UI options
mode = st.selectbox(
    "Caption Mode",
    [
        "Simple / Direct",
        "Descriptive",
        "Character LoRA (WAN 2.2 Style)"
    ]
)

length = st.radio(
    "Caption Length",
    ["Short", "Medium", "Long"],
    horizontal=True
)

trigger_token = st.text_input(
    "Optional Trigger Token (e.g., sbanks25)",
    placeholder="Leave blank if not needed..."
)

generate_button = st.button("✨ Generate Caption")

# ---------------------------
# Handle captioning
# ---------------------------
if generate_button:
    if uploaded_image is None:
        st.error("Please upload an image first.")
    else:
        image = Image.open(uploaded_image).convert("RGB")

        with st.spinner("Generating caption…"):
            cap = generate_caption(image, mode, length, trigger_token)

        st.subheader("Generated Caption")
        st.success(cap)

        st.image(image, caption="Uploaded Image", use_column_width=True)

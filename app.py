import streamlit as st
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

# Page Setup
st.set_page_config(
    page_title="Image Caption Generator",
    page_icon="🖼️",
    layout="centered",
)

# Load BLIP-Large (cached so Streamlit Cloud doesn’t reload every time)
@st.cache_resource
def load_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
    return processor, model

st.title("Image Caption Generator")
st.write("Upload an image, and this app will generate a caption for it using BLIP-Large.")

# Upload UI
uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"],
    help="Drag and drop or browse images (JPG, PNG)",
)

if uploaded_file is not None:
    # Show preview
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Button
    if st.button("Generate Caption"):
        with st.spinner("Generating caption..."):
            processor, model = load_model()
            inputs = processor(image, return_tensors="pt")
            output = model.generate(**inputs, max_new_tokens=50)
            caption = processor.decode(output[0], skip_special_tokens=True)

        # Output box
        st.success("Caption Generated:")
        st.markdown(f"### **{caption}**")
else:
    st.info("⬆️ Upload an image to get started.")

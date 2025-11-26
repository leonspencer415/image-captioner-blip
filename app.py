import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

# ---------------------------------------------------------
# MUST BE FIRST STREAMLIT COMMAND
# ---------------------------------------------------------
st.set_page_config(page_title="Image Caption Generator", layout="centered")

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
# Custom CSS (animated hover + smooth glow)
# ---------------------------------------------------------
st.markdown(
    """
    <style>
        /* Animated hover for upload box */
        .stFileUploader:hover {
            border: 2px solid #4B9CD3 !important;
            background: linear-gradient(135deg, #0e1117, #111827) !important;
            box-shadow: 0 0 20px #4B9CD355;
            transition: 0.3s ease-in-out;
        }

        /* Copy button style */
        .copy-btn {
            background-color: #4B9CD3;
            padding: 8px 16px;
            border-radius: 6px;
            color: white;
            cursor: pointer;
            display: inline-block;
            margin-top: 10px;
            font-size: 14px;
        }
        .copy-btn:hover {
            background-color: #6bb7ea;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# UI
# ---------------------------------------------------------
st.title("Image Caption Generator")
st.write("Upload an image and generate a BLIP caption.")

# Trigger token
trigger = st.text_input("Optional Trigger Word (for LoRA training)", value="")

# Upload box
uploaded_file = st.file_uploader(
    "Drag and drop an image here",
    type=["jpg", "jpeg", "png"],
)

# Where the final caption will be stored
final_caption = ""

# ---------------------------------------------------------
# Generate Caption
# ---------------------------------------------------------
if uploaded_file and st.button("Generate Caption"):

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    inputs = processor(image, return_tensors="pt")
    out = model.generate(**inputs)
    blip_caption = processor.decode(out[0], skip_special_tokens=True)

    # Prepend trigger if needed
    if trigger.strip():
        final_caption = f"{trigger.strip()}. {blip_caption}"
    else:
        final_caption = blip_caption

    st.subheader("Generated Caption")
    st.markdown(
        f"""
        <div style="
            background-color:#1f3b25;
            padding:15px;
            border-radius:10px;
            border:1px solid #2e5d3f;
            color:white;
            font-size:16px;
        ">
            {final_caption}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---------------------------------------------------------
    # Copy to clipboard button (JS hack)
    # ---------------------------------------------------------
    st.markdown(
        f"""
        <script>
            function copyText() {{
                navigator.clipboard.writeText(`{final_caption}`);
            }}
        </script>

        <div class="copy-btn" onclick="copyText()">Copy Caption</div>
        """,
        unsafe_allow_html=True,
    )

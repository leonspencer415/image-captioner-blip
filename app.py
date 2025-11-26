import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import streamlit.components.v1 as components

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="Image Caption Generator",
    layout="centered",
)

# -------------------------------------------------
# CUSTOM CSS (big dropzone + glow animation)
# -------------------------------------------------
st.markdown(
    """
    <style>
        /* Increase width of uploader */
        .uploadedFile {
            width: 100% !important;
        }

        /* Glow drag-hover effect */
        [data-testid="stFileUploader"] div {
            border: 2px dashed #4B9CD3 !important;
            padding: 40px !important;
            border-radius: 15px;
            transition: 0.3s ease;
        }

        [data-testid="stFileUploader"]:hover div {
            border-color: #76c8ff !important;
            box-shadow: 0 0 20px rgba(118, 200, 255, 0.4);
        }

        /* Copy button hover animation */
        .copy-btn:hover {
            background-color: #6ab8e8 !important;
            transform: scale(1.05);
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------
# LOADING BLIP MODEL
# -------------------------------------------------
@st.cache_resource
def load_blip():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
    return processor, model

processor, model = load_blip()

# -------------------------------------------------
# COPY BUTTON COMPONENT (always works on Streamlit Cloud)
# -------------------------------------------------
def copy_to_clipboard_button(text_to_copy: str):
    safe_text = text_to_copy.replace("`", "\\`")
    components.html(
        f"""
        <button 
            class="copy-btn"
            onclick="navigator.clipboard.writeText(`{safe_text}`)"
            style="
                background-color:#4B9CD3;
                padding:10px 16px;
                border:none;
                border-radius:6px;
                color:white;
                cursor:pointer;
                margin-top:12px;
                font-size:14px;
                transition:0.2s ease;
            "
        >
            Copy Caption
        </button>
        """,
        height=60,
    )

# -------------------------------------------------
# UI TITLE
# -------------------------------------------------
st.title("Image Caption Generator")
st.write("Upload an image and generate a caption using BLIP.")

# -------------------------------------------------
# TRIGGER WORD INPUT
# -------------------------------------------------
trigger = st.text_input("Optional Trigger / Token", placeholder="example: kstat25")

# -------------------------------------------------
# FILE UPLOADER
# -------------------------------------------------
uploaded_image = st.file_uploader(
    "Drag and drop file here",
    type=["jpg", "jpeg", "png"],
)

# -------------------------------------------------
# CAPTION GENERATION
# -------------------------------------------------
if uploaded_image:

    image = Image.open(uploaded_image).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Generate Caption"):
        inputs = processor(image, return_tensors="pt")
        output = model.generate(**inputs)
        blip_caption = processor.decode(output[0], skip_special_tokens=True)

        # Apply trigger if exists
        if trigger.strip() != "":
            final_caption = f"{trigger.strip()}. {blip_caption}"
        else:
            final_caption = blip_caption

        # Display result
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

        # Working copy-to-clipboard button
        copy_to_clipboard_button(final_caption)

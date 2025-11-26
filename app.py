import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import pickle

# Load tokenizer + model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("mymodel.h5", compile=False)
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

model, tokenizer = load_model()

# Caption generator
def generate_caption(img, max_len=20):
    image = img.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)

    # start token "<start>"
    caption = ["startseq"]

    for _ in range(max_len):
        seq = tokenizer.texts_to_sequences([" ".join(caption)])[0]
        seq = tf.keras.preprocessing.sequence.pad_sequences([seq], maxlen=34, padding="post")

        y_pred = model.predict([image, seq], verbose=0)
        y_pred = np.argmax(y_pred)

        word = tokenizer.index_word.get(y_pred, None)
        if word is None:
            break
        caption.append(word)
        if word == "endseq":
            break

    final_caption = " ".join(caption[1:-1])  # remove startseq/endseq
    return final_caption


# Streamlit UI
st.title("🖼️ Image Caption Generator")
st.markdown("Upload an image, choose caption style, and generate a custom caption.")

# Load custom CSS
with open("style.css") as css:
    st.markdown(f"<style>{css.read()}</style>", unsafe_allow_html=True)

# UI Controls
length_mode = st.selectbox("Caption Length", ["Short", "Medium", "Long"])
trigger_word = st.text_input("Optional Trigger Token (ex: sbanks25)")

length_map = {
    "Short": 12,
    "Medium": 20,
    "Long": 30,
}

max_length = length_map[length_mode]

# Upload area
st.markdown('<div class="upload-box"> Drag & Drop Image Here </div>', unsafe_allow_html=True)
uploaded = st.file_uploader(" ", type=["png", "jpg", "jpeg"])

# Display + Caption
if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    if st.button("Generate Caption"):
        with st.spinner("Analyzing..."):
            cap = generate_caption(img, max_length)
            if trigger_word.strip():
                cap = f"{trigger_word.strip()} {cap}"

        st.success("Caption Generated:")
        st.write(cap)

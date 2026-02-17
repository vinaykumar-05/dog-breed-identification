import streamlit as st
import numpy as np
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Load Model
model = load_model("dogbreed_model.keras")

# Load Class Names
with open("class_names.json","r") as f:
    class_names = json.load(f)

st.title("🐶 Dog Breed Identification")

uploaded_file = st.file_uploader("Upload Dog Image", type=["jpg","png","jpeg"])

if uploaded_file is not None:

    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    img = img.resize((224,224))
    img_array = np.array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array)
    confidence = np.max(pred)

    if confidence < 0.6:
        st.error("Not a Dog Image")
    else:
        breed = class_names[np.argmax(pred)]
        st.success(f"{breed} ({confidence*100:.2f}%)")

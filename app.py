import streamlit as st
import tensorflow as tf
model = tf.keras.models.load_model("profile_fake_detection_model.h5", compile=False)
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# model = tf.keras.models.load_model("profile_fake_detection_model.h5")

st.title("Profile Picture Analysis for Fake Account Detection")

uploaded_file = st.file_uploader(
    "Upload Profile Picture",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    img = Image.open(uploaded_file)

    st.image(img, caption="Uploaded Image", use_column_width=True)

    img = img.resize((224, 224))

    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]

    st.write("Prediction Score:", prediction)

    if prediction > 0.30:
        st.success("REAL Profile Picture")
    else:
        st.error("FAKE / Suspicious Profile Picture")

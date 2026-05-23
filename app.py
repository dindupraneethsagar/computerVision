import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="Profile Picture Analysis", page_icon="🕵️")

@st.cache_resource
def load_my_model():
    base_model = tf.keras.applications.MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3)
    )

    base_model.trainable = False

    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    output = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs=base_model.input, outputs=output)

    model.load_weights("profile_model.weights.h5")

    return model

model = load_my_model()

st.title("Profile Picture Analysis for Fake Account Detection")
st.write("Upload a profile picture to check whether it looks real or suspicious.")

uploaded_file = st.file_uploader(
    "Upload Profile Picture",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")

    st.image(img, caption="Uploaded Image", use_container_width=True)

    img = img.resize((224, 224))

    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]
    prediction_score = float(prediction)

    st.write("Prediction Score:", prediction_score)

    if prediction_score > 0.30:
        st.success("REAL Profile Picture")
    else:
        st.error("FAKE / Suspicious Profile Picture")

import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# LOAD MODEL
model = tf.keras.models.load_model(
    'models/crop_disease_model.h5'
)

# CORRECT CLASS ORDER
classes = [
    'Tomato_Early_blight',
    'Tomato_Late_blight',
    'Tomato_Leaf_Mold',
    'Tomato_healthy'
]

# RECOMMENDATIONS
recommendations = {

    'Tomato_Early_blight':
    'Use copper fungicide and avoid overwatering.',

    'Tomato_Late_blight':
    'Remove infected leaves and apply fungicide immediately.',

    'Tomato_Leaf_Mold':
    'Improve air circulation and reduce humidity.',

    'Tomato_healthy':
    'Crop appears healthy. Maintain proper irrigation.'
}

# PAGE CONFIG
st.set_page_config(
    page_title="AgriAI",
    layout="centered"
)

# TITLE
st.title("🌱 AgriAI")

st.subheader(
    "Smart Crop Disease Detection Using Deep Learning"
)

st.write(
    "Upload a tomato leaf image to detect crop diseases using MobileNetV2 transfer learning."
)

# FILE UPLOAD
uploaded_file = st.file_uploader(
    "Upload Leaf Image",
    type=['jpg', 'jpeg', 'png']
)

if uploaded_file is not None:

    # READ IMAGE
    image = Image.open(uploaded_file).convert("RGB")

    # DISPLAY IMAGE
    st.image(
        image,
        caption="Uploaded Leaf Image",
        use_container_width=True
    )

    # CONVERT TO NUMPY
    img = np.array(image)

    # RESIZE
    img = cv2.resize(img, (224,224))

    # DENOISE
    img = cv2.GaussianBlur(img, (3,3), 0)

    # NORMALIZE
    img = img / 255.0

    # EXPAND DIMS
    img = np.expand_dims(img, axis=0)

    # PREDICTION
    prediction = model.predict(img)

    predicted_index = np.argmax(prediction)

    predicted_class = classes[predicted_index]

    confidence = prediction[0][predicted_index] * 100

    # FORMATTED CLASS NAME
    display_name = predicted_class.replace("_", " ")

    # RESULTS
    st.success(f"Prediction: {display_name}")

    st.info(f"Confidence: {confidence:.2f}%")

    st.warning(
        recommendations[predicted_class]
    )

    # CONFIDENCE SCORES
    st.subheader("AI Confidence Scores")

    for i in range(len(classes)):

        class_name = classes[i].replace("_", " ")

        score = prediction[0][i] * 100

        st.write(f"{class_name} : {score:.2f}%")

    # PROGRESS BAR
    st.subheader("Prediction Confidence")

    st.progress(int(confidence))

# FOOTER
st.markdown("---")

st.caption(
    "Developed using MobileNetV2 Transfer Learning and Streamlit."
)
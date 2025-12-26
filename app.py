import streamlit as st
import pickle
import numpy as np
import pandas as pd
import cv2

with open("Bones_fractured_notfractured.pkl", "rb") as f:
    model = pickle.load(f)

st.set_page_config(page_title="Bone Fracture Detection", layout="centered")
st.header("🦴 Bone Fracture Detection System")
st.write("**Upload an X-ray image to check whether the bone is fractured or not.**")

def preprocess_uploaded_image(uploaded_file):
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("Invalid image. Please upload a valid JPG/PNG image.")

    image = cv2.resize(image, (64, 64), interpolation=cv2.INTER_LINEAR)
    img_flatten = image.flatten()
    df = pd.DataFrame([img_flatten])

    return df

uploaded_file = st.file_uploader(
    "Upload X-ray Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    st.image(
        uploaded_file.getvalue(),
        caption="Uploaded X-ray Image",
        use_container_width=True
    )

    if st.button("Predict"):
        with st.spinner("Analyzing X-ray..."):
            input_df = preprocess_uploaded_image(uploaded_file)
            prediction = model.predict(input_df)

        result = prediction[0]

        if result == "Fractured":
            st.error("❌ Fracture Detected")
        else:
            st.success("✅ No Fracture Detected")

        if hasattr(model, "predict_proba"):
            confidence = model.predict_proba(input_df)[0].max() * 100
            st.info(f"Confidence: {confidence:.2f}%")

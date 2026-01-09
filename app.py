import streamlit as st
from PIL import Image
from predict import predict_image

st.title("🐶🐱 Pet Classification App")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        label = predict_image(image)
        st.success(f"Prediction: **{label}**")

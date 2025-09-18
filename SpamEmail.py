import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
import numpy as np

model = tf.keras.models.load_model("text_model.h5", custom_objects={'KerasLayer': hub.KerasLayer})

st.title("📩 Email Classification (SPAM OR NOT SPAM)")
user_input = st.text_area("✍️ Enter the email content below:")

if st.button("🔍 Predict"):
    if not user_input.strip():
        st.warning("⚠️ Text cannot be empty.")
    else:
        prob = float(model.predict([user_input])[0])

        if prob >= 0.5:
            label = "SPAM"
            confidence = prob
            label_color = "🔴"
        else:
            label = "NOT SPAM"
            confidence = 1 - prob
            label_color = "🟢"

        st.markdown(f"### ✅ Prediction: `{label_color} {label}`")
        st.markdown(f"🧠 Confidence Level: `{confidence:.2%}`")
        st.markdown(f"📊 Raw Probability (SPAM class): `{prob:.2f}`")

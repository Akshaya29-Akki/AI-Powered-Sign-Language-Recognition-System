import streamlit as st
import cv2
import pickle
import numpy as np

st.set_page_config(
    page_title="AI Sign Language Recognition",
    page_icon="🤟",
    layout="centered"
)

st.title("🤟 AI Sign Language Recognition")
st.write("Real-time recognition of basic sign language gestures.")

# Load trained model
with open("sign_language_model.pkl", "rb") as file:
    model = pickle.load(file)

st.success("AI model loaded successfully!")

st.subheader("Supported Signs")
st.write("HELLO  |  YES  |  NO  |  THANK YOU")

start = st.button("📷 Start Camera")

if start:
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("Camera could not be opened.")
    else:
        st.info("Camera started. Press Stop Camera when finished.")

        frame_placeholder = st.empty()
        prediction_placeholder = st.empty()

        stop = st.button("⛔ Stop Camera")

        while cap.isOpened() and not stop:
            ret, frame = cap.read()

            if not ret:
                st.error("Could not read camera frame.")
                break

            frame = cv2.flip(frame, 1)

            height, width, _ = frame.shape

            x1 = int(width * 0.25)
            y1 = int(height * 0.15)
            x2 = int(width * 0.75)
            y2 = int(height * 0.85)

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            roi = frame[y1:y2, x1:x2]

            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (64, 64))
            normalized = resized / 255.0

            features = normalized.flatten().reshape(1, -1)

            prediction = model.predict(features)[0]

            cv2.putText(
                frame,
                "Sign: " + str(prediction),
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            frame_placeholder.image(
                frame_rgb,
                channels="RGB"
            )

            prediction_placeholder.success(
                "Recognized Sign: " + str(prediction)
            )

        cap.release()
        cv2.destroyAllWindows()
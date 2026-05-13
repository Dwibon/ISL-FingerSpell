import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av
import cv2
import mediapipe as mp
import numpy as np
import joblib

# ---------------------------------------------------
# Streamlit page config
# ---------------------------------------------------

st.set_page_config(
    page_title="ISL FingerSpell Recognition",
    layout="centered"
)

st.title("🤟 ISL FingerSpell Recognition")
st.write("Real-time Indian Sign Language alphabet recognition")

# ---------------------------------------------------
# Load model
# ---------------------------------------------------

model = joblib.load("/Users/dwibon/Desktop/ISL-FingerSpell/models/classes.pkl")
classes = joblib.load("/Users/dwibon/Desktop/ISL-FingerSpell/models/classes.pkl")

# ---------------------------------------------------
# MediaPipe setup
# ---------------------------------------------------

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    min_detection_confidence=0.8,
    min_tracking_confidence=0.5,
    max_num_hands=2
)

# ---------------------------------------------------
# Feature extraction
# ---------------------------------------------------

def extract_landmarks_two_hands(
    multi_hand_landmarks,
    multi_handedness
):

    left_features = [0.0] * 63
    right_features = [0.0] * 63

    if multi_hand_landmarks:

        for hand_landmarks, handedness in zip(
            multi_hand_landmarks,
            multi_handedness
        ):

            label = handedness.classification[0].label

            wrist = hand_landmarks.landmark[0]

            features = []

            for lm in hand_landmarks.landmark:

                features.append(round(lm.x - wrist.x, 6))
                features.append(round(lm.y - wrist.y, 6))
                features.append(round(lm.z - wrist.z, 6))

            if label == "Left":
                left_features = features
            else:
                right_features = features

    return left_features + right_features

# ---------------------------------------------------
# Video Processor
# ---------------------------------------------------

class VideoProcessor(VideoTransformerBase):

    frame_count = 0

    def recv(self, frame):

        img = frame.to_ndarray(format="bgr24")

        img = cv2.flip(img, 1)

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        result = hands.process(rgb)

        prediction = "?"
        confidence = 0.0

        # Frame skipping for performance
        self.frame_count += 1

        if result.multi_hand_landmarks:

            # Draw landmarks
            for hand_lm in result.multi_hand_landmarks:

                mp_drawing.draw_landmarks(
                    img,
                    hand_lm,
                    mp_hands.HAND_CONNECTIONS
                )

            # Only predict every 2nd frame
            if self.frame_count % 2 == 0:

                features = extract_landmarks_two_hands(
                    result.multi_hand_landmarks,
                    result.multi_handedness
                )

                proba = model.predict_proba([features])[0]

                prediction = classes[np.argmax(proba)]

                confidence = np.max(proba)

            # Display prediction
            cv2.rectangle(img, (0, 0), (500, 80), (0, 0, 0), -1)

            cv2.putText(
                img,
                f"{prediction}",
                (20, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0, 255, 0),
                3
            )

            cv2.putText(
                img,
                f"Confidence: {confidence:.0%}",
                (220, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (200, 200, 200),
                2
            )

        return av.VideoFrame.from_ndarray(
            img,
            format="bgr24"
        )

# ---------------------------------------------------
# Webcam Stream
# ---------------------------------------------------

webrtc_streamer(
    key="isl-demo",
    video_processor_factory=VideoProcessor,
    media_stream_constraints={
        "video": True,
        "audio": False
    },
    async_processing=True
)
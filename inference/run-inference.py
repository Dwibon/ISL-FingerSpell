import cv2
import mediapipe as mp
import numpy as np
import joblib
import collections

# Load model
model = joblib.load('/Users/dwibon/Desktop/ISL-FingerSpell/models/isl_model.pkl')
classes = joblib.load('/Users/dwibon/Desktop/ISL-FingerSpell/models/classes.pkl')

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def extract_landmarks_two_hands(multi_hand_landmarks, multi_handedness):
    left_features = [0.0] * 63
    right_features = [0.0] * 63

    if multi_hand_landmarks:
        for hand_landmarks, handedness in zip(
                multi_hand_landmarks, multi_handedness):
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

def draw_text(frame, text, pos, scale=0.8,
              color=(255,255,255), thickness=2):
    cv2.putText(frame, text, pos,
                cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)

# ── Word formation state ──────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.70
DWELL_FRAMES = 25        # frames held before confirming letter
RELEASE_FRAMES = 10      # frames of uncertainty before allowing same letter again

prediction_buffer = collections.deque(maxlen=15)
dwell_counter = 0
release_counter = 0
last_confirmed = None
current_word = ""
sentence = []

cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    min_detection_confidence=0.8,
    min_tracking_confidence=0.5,
    max_num_hands=2
) as hands:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        result = hands.process(rgb)
        rgb.flags.writeable = True

        hand_detected = result.multi_hand_landmarks is not None

        # ── Draw landmarks ────────────────────────────────────────
        if hand_detected:
            for hand_lm in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_lm, mp_hands.HAND_CONNECTIONS)

        # ── Predict ───────────────────────────────────────────────
        predicted_letter = None
        confidence = 0.0

        if hand_detected:
            features = extract_landmarks_two_hands(
                result.multi_hand_landmarks,
                result.multi_handedness)
            proba = model.predict_proba([features])[0]
            confidence = max(proba)
            if confidence >= CONFIDENCE_THRESHOLD:
                predicted_letter = classes[proba.argmax()]

        prediction_buffer.append(predicted_letter)

        # Most common prediction in buffer
        from collections import Counter
        valid = [p for p in prediction_buffer if p is not None]
        smoothed = Counter(valid).most_common(1)[0][0] if valid else None

        # ── Dwell + word formation ────────────────────────────────
        if smoothed is not None:
            if smoothed == last_confirmed:
                release_counter += 1
                dwell_counter = 0
                if release_counter >= RELEASE_FRAMES:
                    last_confirmed = None
                    release_counter = 0
            else:
                dwell_counter += 1
                release_counter = 0
                if dwell_counter >= DWELL_FRAMES:
                    current_word += smoothed
                    last_confirmed = smoothed
                    dwell_counter = 0
        else:
            dwell_counter = 0
            release_counter = 0

        # ── Progress bar for dwell ────────────────────────────────
        if dwell_counter > 0 and smoothed is not None:
            bar_len = int((dwell_counter / DWELL_FRAMES) * 200)
            cv2.rectangle(frame,
                          (w//2 - 100, h//2 + 160),
                          (w//2 - 100 + bar_len, h//2 + 175),
                          (0, 255, 255), -1)

        # ── Display predicted letter ──────────────────────────────
        cv2.rectangle(frame, (0, 0), (w, 80), (0, 0, 0), -1)

        if smoothed:
            draw_text(frame, smoothed,
                      (w//2 - 30, 65), 2.5, (0, 255, 0), 4)
            draw_text(frame, f"{confidence:.0%}",
                      (w//2 + 50, 60), 0.8, (180, 180, 180))
        else:
            draw_text(frame, "?",
                      (w//2 - 15, 65), 2.5, (100, 100, 100), 4)

        # ── Word and sentence display ─────────────────────────────
        cv2.rectangle(frame, (0, h-90), (w, h), (0, 0, 0), -1)
        draw_text(frame, f"Word:     {current_word}",
                  (10, h-60), 0.8, (255, 255, 0))
        draw_text(frame, f"Sentence: {' '.join(sentence)}",
                  (10, h-30), 0.7, (200, 200, 200))

        cv2.imshow("ISL-FingerSpell", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord(' '):       # space — confirm word
            if current_word:
                sentence.append(current_word)
                current_word = ""
        elif key == ord('c'):       # clear word
            current_word = ""
        elif key == ord('r'):       # reset everything
            current_word = ""
            sentence = []

cap.release()
cv2.destroyAllWindows()
import cv2
import mediapipe as mp
import csv
import os
import time

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# ── H, J, Y excluded — dynamic signs requiring motion ────────────
SIGNS = [c for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" if c not in ('H', 'J', 'Y')]
SAMPLES_PER_SIGN = 300
OUTPUT_FILE = "isl_dataset.csv"

if not os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        header = []
        for side in ['left', 'right']:
            for i in range(21):
                header += [f"{side}_lm{i}_x", f"{side}_lm{i}_y", f"{side}_lm{i}_z"]
        header.append("label")
        writer.writerow(header)

def extract_landmarks_two_hands(multi_hand_landmarks, multi_handedness):
    left_features = [0.0] * 63
    right_features = [0.0] * 63

    if multi_hand_landmarks:
        for hand_landmarks, handedness in zip(multi_hand_landmarks, multi_handedness):
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

def save_row(features, label):
    with open(OUTPUT_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(features + [label])

def draw_text(frame, text, pos, scale=0.8, color=(255, 255, 255), thickness=2):
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness)

cap = cv2.VideoCapture(0)
sign_index = 0
samples_collected = 0
collecting = False
flash = False
flash_time = 0
AUTO_CAPTURE = False

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

        current_sign = SIGNS[sign_index]
        hands_found = len(result.multi_hand_landmarks) if result.multi_hand_landmarks else 0
        hand_detected = hands_found > 0

        if hand_detected:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

        if AUTO_CAPTURE and collecting and hand_detected:
            features = extract_landmarks_two_hands(
                result.multi_hand_landmarks,
                result.multi_handedness
            )
            save_row(features, current_sign)
            samples_collected += 1
            flash = True
            flash_time = time.time()
            if samples_collected >= SAMPLES_PER_SIGN:
                collecting = False

        rect_color = (0, 255, 0)
        if flash:
            if time.time() - flash_time < 0.1:
                rect_color = (0, 255, 255)
            else:
                flash = False

        box_x1 = w // 2 - 320
        box_y1 = h // 2 - 220
        box_x2 = w // 2 + 320
        box_y2 = h // 2 + 220
        cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), rect_color, 2)
        draw_text(frame, "SIGN HERE", (box_x1 + 220, box_y1 - 10), 0.7, rect_color)

        cv2.rectangle(frame, (0, 0), (w, 75), (0, 0, 0), -1)
        draw_text(frame, f"Sign: {current_sign}  [{sign_index + 1}/{len(SIGNS)}]",
                  (10, 25), 0.8, (255, 255, 0))
        draw_text(frame, f"Samples: {samples_collected}/{SAMPLES_PER_SIGN}",
                  (10, 58), 0.8, (255, 255, 255))

        mode_text = "AUTO" if AUTO_CAPTURE else "MANUAL"
        mode_color = (0, 255, 0) if AUTO_CAPTURE else (0, 165, 255)
        draw_text(frame, f"Mode: {mode_text}", (w - 220, 25), 0.7, mode_color)

        if hands_found == 2:
            hand_color = (0, 255, 0)
        elif hands_found == 1:
            hand_color = (0, 165, 255)
        else:
            hand_color = (0, 0, 255)
        draw_text(frame, f"Hands: {hands_found}/2", (w - 220, 58), 0.7, hand_color)

        cv2.rectangle(frame, (0, h - 35), (w, h), (0, 0, 0), -1)
        draw_text(frame, "S=capture  A=auto  N=next  B=back  Q=quit",
                  (10, h - 10), 0.55, (180, 180, 180), 1)

        cv2.imshow("ISL Data Collection", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        elif key == ord('s'):
            if hand_detected and samples_collected < SAMPLES_PER_SIGN:
                features = extract_landmarks_two_hands(
                    result.multi_hand_landmarks,
                    result.multi_handedness
                )
                save_row(features, current_sign)
                samples_collected += 1
                flash = True
                flash_time = time.time()
                if samples_collected >= SAMPLES_PER_SIGN:
                    collecting = False

        elif key == ord('a'):
            AUTO_CAPTURE = not AUTO_CAPTURE
            collecting = AUTO_CAPTURE

        elif key == ord('n'):
            if sign_index < len(SIGNS) - 1:
                sign_index += 1
                samples_collected = 0
                collecting = False
                AUTO_CAPTURE = False

        elif key == ord('b'):
            if sign_index > 0:
                sign_index -= 1
                samples_collected = 0
                collecting = False
                AUTO_CAPTURE = False

cap.release()
cv2.destroyAllWindows()
print(f"Collection complete. Data saved to {OUTPUT_FILE}")
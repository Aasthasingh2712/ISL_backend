import cv2
import mediapipe as mp
import numpy as np
import os

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

LABELS = [
    "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "A", "B", "C", "D", "E", "F", "G", "H","I", "J",
    "K", "L"
]
SAMPLES_PER_LABEL = 100
SAVE_DIR = "static_dataset"
os.makedirs(SAVE_DIR, exist_ok=True)

def extract_hand_features(results):
    lh = np.zeros(63)
    rh = np.zeros(63)
    if results.multi_hand_landmarks and results.multi_handedness:
        for lm, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            coords = np.array([[p.x, p.y, p.z] for p in lm.landmark]).flatten()
            if handedness.classification[0].label == "Left":
                lh = coords
            else:
                rh = coords
    return np.concatenate([lh, rh])  # 126-dim

cap = cv2.VideoCapture(0)

with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_hands=2) as hands:
    for label in LABELS:
        collected = []
        print(f"\n=== Ready for sign: '{label}' ===")
        print("Press 'c' to start capturing this sign, 'q' to quit early.")

        capturing = False
        while len(collected) < SAMPLES_PER_LABEL:
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            if results.multi_hand_landmarks:
                for hlm in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hlm, mp_hands.HAND_CONNECTIONS)

            status = f"Sign: {label} | Captured: {len(collected)}/{SAMPLES_PER_LABEL} | {'CAPTURING' if capturing else 'Press c to start'}"
            cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Collect Static Signs", frame)

            key = cv2.waitKey(10) & 0xFF
            if key == ord('c'):
                capturing = True
            if key == ord('q'):
                break

            if capturing and results.multi_hand_landmarks:
                features = extract_hand_features(results)
                collected.append(features)

        np.save(os.path.join(SAVE_DIR, f"{label}.npy"), np.array(collected))
        print(f"Saved {len(collected)} samples for '{label}'")

cap.release()
cv2.destroyAllWindows()
print("\nDONE collecting all labels.")
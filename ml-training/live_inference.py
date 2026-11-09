"""
live_inference.py
Real-time ISL sentence recognition using webcam + MediaPipe (pose + hands, NO face) + trained LSTM model.
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import mediapipe as mp
from collections import deque

SEQUENCE_LENGTH = 30
NUM_POSE_LANDMARKS = 33
NUM_HAND_LANDMARKS = 21
POSE_FEATURES = NUM_POSE_LANDMARKS * 4
HAND_FEATURES = NUM_HAND_LANDMARKS * 3
FEATURES_PER_FRAME = POSE_FEATURES + (HAND_FEATURES * 2)

HIDDEN_SIZE = 128
CONFIDENCE_THRESHOLD = 0.5
PREDICT_EVERY_N_FRAMES = 5

MODEL_PATH = "sign_lstm_model.pth"
CLASSES_PATH = "label_classes.npy"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SignLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size // 2, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(hidden_size // 2, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out, (h_n, _) = self.lstm2(out)
        last_hidden = h_n[-1]
        out = self.dropout(last_hidden)
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        return out


mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils


def extract_keypoints(results):
    pose = np.zeros(POSE_FEATURES)
    if results.pose_landmarks:
        pose = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in results.pose_landmarks.landmark]).flatten()

    left_hand = np.zeros(HAND_FEATURES)
    if results.left_hand_landmarks:
        left_hand = np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]).flatten()

    right_hand = np.zeros(HAND_FEATURES)
    if results.right_hand_landmarks:
        right_hand = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]).flatten()

    return np.concatenate([pose, left_hand, right_hand])


def main():
    classes = np.load(CLASSES_PATH, allow_pickle=True)
    num_classes = len(classes)
    print(f"Loaded {num_classes} classes: {list(classes)}")

    model = SignLSTM(FEATURES_PER_FRAME, HIDDEN_SIZE, num_classes).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print("Model loaded successfully.\n")

    sequence_buffer = deque(maxlen=SEQUENCE_LENGTH)
    current_prediction = "No sign detected"
    current_confidence = 0.0
    frame_count = 0

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(frame_rgb)

            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
            mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

            hands_visible = results.left_hand_landmarks is not None or results.right_hand_landmarks is not None

            keypoints = extract_keypoints(results)
            sequence_buffer.append(keypoints)
            frame_count += 1

            if not hands_visible:
                current_prediction = "No sign detected"
                current_confidence = 0.0
            elif len(sequence_buffer) == SEQUENCE_LENGTH and frame_count % PREDICT_EVERY_N_FRAMES == 0:
                seq_array = np.array(sequence_buffer)
                seq_tensor = torch.tensor(seq_array, dtype=torch.float32).unsqueeze(0).to(DEVICE)

                with torch.no_grad():
                    outputs = model(seq_tensor)
                    probs = torch.softmax(outputs, dim=1)
                    confidence, pred_idx = torch.max(probs, dim=1)

                confidence = confidence.item()
                pred_idx = pred_idx.item()

                if confidence >= CONFIDENCE_THRESHOLD:
                    current_prediction = classes[pred_idx]
                    current_confidence = confidence
                else:
                    current_prediction = "..."
                    current_confidence = confidence

            cv2.rectangle(frame, (0, 0), (frame.shape[1], 60), (0, 0, 0), -1)
            text = f"{current_prediction} ({current_confidence*100:.1f}%)"
            cv2.putText(frame, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            cv2.imshow("ISL Sentence Recognition", frame)

            if cv2.waitKey(10) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
import numpy as np
import torch
import mediapipe as mp
import cv2
from collections import deque
from app.ml.model import SignLSTM

# ---- Load classes ----
CLASSES = np.load("app/ml/label_classes.npy", allow_pickle=True)
NUM_CLASSES = len(CLASSES)

# ---- Load model (must match training: input_size=258, hidden=128) ----
DEVICE = torch.device("cpu")
model = SignLSTM(input_size=258, hidden_size=128, num_classes=NUM_CLASSES)
model.load_state_dict(torch.load("app/ml/sign_lstm_model.pth", map_location=DEVICE))
model.eval()

# ---- MediaPipe setup (Hands + Pose, matches 258-dim training data) ----
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
hands_model = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_hands=2)
pose_model = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

SEQ_LEN = 30

def extract_landmarks(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    pose_results = pose_model.process(rgb)
    pose = np.array([[r.x, r.y, r.z, r.visibility] for r in pose_results.pose_landmarks.landmark]).flatten() \
        if pose_results.pose_landmarks else np.zeros(33 * 4)

    hand_results = hands_model.process(rgb)
    lh = np.zeros(63)
    rh = np.zeros(63)
    if hand_results.multi_hand_landmarks and hand_results.multi_handedness:
        for lm, handedness in zip(hand_results.multi_hand_landmarks, hand_results.multi_handedness):
            coords = np.array([[p.x, p.y, p.z] for p in lm.landmark]).flatten()
            if handedness.classification[0].label == "Left":
                lh = coords
            else:
                rh = coords

    return np.concatenate([pose, lh, rh])  # 258-dim


class SentencePredictor:
    def __init__(self):
        self.buffer = deque(maxlen=SEQ_LEN)

    def add_frame(self, frame):
        landmarks = extract_landmarks(frame)
        self.buffer.append(landmarks)

    def predict(self, threshold=0.4):
        if len(self.buffer) < SEQ_LEN:
            return None, 0.0, len(self.buffer)  # not enough frames yet

        x = torch.tensor(np.array(self.buffer), dtype=torch.float32).unsqueeze(0)  # (1, 30, 258)
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            conf, idx = torch.max(probs, dim=1)

        if conf.item() > threshold:
            return str(CLASSES[idx.item()]), conf.item(), len(self.buffer)
        return None, conf.item(), len(self.buffer)

    def reset(self):
        self.buffer.clear()
import cv2
import numpy as np
import torch
import torch.nn as nn
import mediapipe as mp
from collections import deque

DEVICE = torch.device("cpu")

# ---------- NORMALIZATION (must match train_static.py exactly) ----------
def normalize_hand(hand_63):
    if np.all(hand_63 == 0):
        return hand_63
    pts = hand_63.reshape(21, 3)
    wrist = pts[0].copy()
    pts = pts - wrist
    scale = np.max(np.linalg.norm(pts, axis=1))
    if scale > 1e-6:
        pts = pts / scale
    return pts.flatten()

def normalize_features(feat_126):
    lh = feat_126[:63]
    rh = feat_126[63:]
    return np.concatenate([normalize_hand(lh), normalize_hand(rh)])

# ---------- STATIC MODEL (numbers + alphabets) ----------
class StaticClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        return self.fc3(x)

static_classes = np.load("static_label_classes.npy", allow_pickle=True)
static_model = StaticClassifier(126, len(static_classes)).to(DEVICE)
static_model.load_state_dict(torch.load("static_sign_model.pth", map_location=DEVICE))
static_model.eval()

# ---------- SENTENCE MODEL (LSTM) ----------
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
        return self.fc2(out)

sentence_classes = np.load("label_classes.npy", allow_pickle=True)
sentence_model = SignLSTM(258, 128, len(sentence_classes)).to(DEVICE)
sentence_model.load_state_dict(torch.load("sign_lstm_model.pth", map_location=DEVICE))
sentence_model.eval()

# ---------- MediaPipe ----------
mp_hands = mp.solutions.hands
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

SEQ_LEN = 30
STATIC_THRESHOLD = 0.6
SENTENCE_THRESHOLD = 0.5
STATIC_PREDICT_EVERY_N_FRAMES = 3

def extract_hand_features_from_hands_model(results):
    lh = np.zeros(63)
    rh = np.zeros(63)
    if results.multi_hand_landmarks and results.multi_handedness:
        for lm, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            coords = np.array([[p.x, p.y, p.z] for p in lm.landmark]).flatten()
            if handedness.classification[0].label == "Left":
                lh = coords
            else:
                rh = coords
    return np.concatenate([lh, rh])

def extract_pose_hands_from_holistic(results):
    pose = np.zeros(33 * 4)
    if results.pose_landmarks:
        pose = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in results.pose_landmarks.landmark]).flatten()
    lh = np.zeros(63)
    rh = np.zeros(63)
    if results.left_hand_landmarks:
        lh = np.array([[p.x, p.y, p.z] for p in results.left_hand_landmarks.landmark]).flatten()
    if results.right_hand_landmarks:
        rh = np.array([[p.x, p.y, p.z] for p in results.right_hand_landmarks.landmark]).flatten()
    return np.concatenate([pose, lh, rh])

sequence_buffer = deque(maxlen=SEQ_LEN)
static_text = "No sign"
sentence_text = "..."
frame_count = 0

cap = cv2.VideoCapture(0)

with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_hands=2) as hands_detector, \
     mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic_detector:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        hands_results = hands_detector.process(rgb)
        holistic_results = holistic_detector.process(rgb)

        if hands_results.multi_hand_landmarks:
            for hlm in hands_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hlm, mp_hands.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(frame, holistic_results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

        hands_visible = bool(hands_results.multi_hand_landmarks)
        frame_count += 1

        if hands_visible:
            if frame_count % STATIC_PREDICT_EVERY_N_FRAMES == 0:
                raw_feats = extract_hand_features_from_hands_model(hands_results)
                norm_feats = normalize_features(raw_feats)
                hand_feats = torch.tensor(norm_feats, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    out = static_model(hand_feats)
                    probs = torch.softmax(out, dim=1)
                    conf, idx = torch.max(probs, dim=1)
                if conf.item() > STATIC_THRESHOLD:
                    static_text = f"{static_classes[idx.item()]} ({conf.item()*100:.0f}%)"
                else:
                    static_text = "..."

            sequence_buffer.append(extract_pose_hands_from_holistic(holistic_results))
            if len(sequence_buffer) == SEQ_LEN and frame_count % 5 == 0:
                seq = torch.tensor(np.array(sequence_buffer), dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    out = sentence_model(seq)
                    probs = torch.softmax(out, dim=1)
                    conf, idx = torch.max(probs, dim=1)
                if conf.item() > SENTENCE_THRESHOLD:
                    sentence_text = f"{sentence_classes[idx.item()]} ({conf.item()*100:.0f}%)"
                else:
                    sentence_text = "..."
        else:
            static_text = "No sign"
            sequence_buffer.clear()
            sentence_text = "..."

        cv2.rectangle(frame, (0, 0), (frame.shape[1], 80), (0, 0, 0), -1)
        cv2.putText(frame, f"Letter/Number: {static_text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Sentence: {sentence_text}", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)

        cv2.imshow("ISL Recognition - Combined", frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
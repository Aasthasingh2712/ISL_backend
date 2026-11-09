"""
main.py
FastAPI backend with a simple HTTP POST endpoint (no WebSocket) for
real-time static ISL sign recognition. Frontend polls this every ~250ms.
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import mediapipe as mp

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

STATIC_THRESHOLD = 0.6
MODEL_PATH = "static_sign_model.pth"
CLASSES_PATH = "static_label_classes.npy"

DEVICE = torch.device("cpu")


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


classes = np.load(CLASSES_PATH, allow_pickle=True)
num_classes = len(classes)

model = StaticClassifier(126, num_classes).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

print(f"Static model loaded. Classes ({num_classes}): {list(classes)}")

mp_hands = mp.solutions.hands
hands_detector = mp_hands.Hands(
    static_image_mode=True,
    min_detection_confidence=0.5,
    max_num_hands=2,
)


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
    return np.concatenate([lh, rh])


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"message": "ISL Static Sign Recognition backend is running"}


@app.post("/predict")
async def predict(request: Request):
    data = await request.body()

    np_arr = np.frombuffer(data, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if frame is None:
        return {"prediction": "No sign detected", "confidence": 0.0}

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands_detector.process(frame_rgb)

    hands_visible = bool(results.multi_hand_landmarks)

    prediction = "No sign detected"
    confidence = 0.0

    if hands_visible:
        raw_feats = extract_hand_features(results)
        norm_feats = normalize_features(raw_feats)
        feats_tensor = torch.tensor(norm_feats, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            out = model(feats_tensor)
            probs = torch.softmax(out, dim=1)
            conf_val, pred_idx = torch.max(probs, dim=1)

        confidence = conf_val.item()
        pred_idx = pred_idx.item()

        if confidence >= STATIC_THRESHOLD:
            prediction = str(classes[pred_idx])
        else:
            prediction = "..."

    return {"prediction": prediction, "confidence": confidence}
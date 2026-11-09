"""
preprocess_sentences.py
Extracts MediaPipe landmarks (pose + both hands, NO face) from
ISL-CSLRT sentence-level videos and saves training-ready .npy files
(X_sentences.npy, y_sentences.npy)
"""

import os
import cv2
import numpy as np
import mediapipe as mp
from sklearn.preprocessing import LabelEncoder

# ----------------------------
# CONFIG
# ----------------------------
BASE_DIR = os.path.join(
    "dataset_raw", "ISL_CSLRT_Corpus", "ISL_CSLRT_Corpus", "Videos_Sentence_Level"
)
SEQUENCE_LENGTH = 30

NUM_POSE_LANDMARKS = 33
NUM_HAND_LANDMARKS = 21

POSE_FEATURES = NUM_POSE_LANDMARKS * 4      # x,y,z,visibility
HAND_FEATURES = NUM_HAND_LANDMARKS * 3      # x,y,z (per hand)

FEATURES_PER_FRAME = POSE_FEATURES + (HAND_FEATURES * 2)  # 132 + 63 + 63 = 258

TEST_SENTENCES = [
    "serve the food",
    "can i help you",
    "go and sleep",
]

# ----------------------------
# MEDIAPIPE SETUP
# ----------------------------
mp_holistic = mp.solutions.holistic


def extract_keypoints(results):
    pose = np.zeros(POSE_FEATURES)
    if results.pose_landmarks:
        pose = np.array(
            [[lm.x, lm.y, lm.z, lm.visibility] for lm in results.pose_landmarks.landmark]
        ).flatten()

    left_hand = np.zeros(HAND_FEATURES)
    if results.left_hand_landmarks:
        left_hand = np.array(
            [[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]
        ).flatten()

    right_hand = np.zeros(HAND_FEATURES)
    if results.right_hand_landmarks:
        right_hand = np.array(
            [[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]
        ).flatten()

    return np.concatenate([pose, left_hand, right_hand])


def extract_landmarks_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames_landmarks = []

    with mp_holistic.Holistic(
        static_image_mode=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(frame_rgb)

            frame_features = extract_keypoints(results)
            frames_landmarks.append(frame_features)

    cap.release()

    if len(frames_landmarks) == 0:
        return None

    frames_landmarks = np.array(frames_landmarks)

    if len(frames_landmarks) < SEQUENCE_LENGTH:
        pad_count = SEQUENCE_LENGTH - len(frames_landmarks)
        padding = np.zeros((pad_count, FEATURES_PER_FRAME))
        frames_landmarks = np.vstack([frames_landmarks, padding])
    else:
        frames_landmarks = frames_landmarks[:SEQUENCE_LENGTH]

    return frames_landmarks


def main():
    X = []
    y = []

    sentence_folders = sorted(os.listdir(BASE_DIR))

    if TEST_SENTENCES is not None:
        sentence_folders = [s for s in sentence_folders if s in TEST_SENTENCES]

    print(f"Found {len(sentence_folders)} sentence folders to process.\n")

    for sentence in sentence_folders:
        folder_path = os.path.join(BASE_DIR, sentence)
        if not os.path.isdir(folder_path):
            continue

        video_files = [
            f for f in os.listdir(folder_path)
            if f.lower().endswith(".mp4")
        ]

        print(f"[{sentence}] -> {len(video_files)} video(s) found")

        for video_file in video_files:
            video_path = os.path.join(folder_path, video_file)
            print(f"  Processing: {video_file} ...", end=" ")

            landmarks = extract_landmarks_from_video(video_path)

            if landmarks is None:
                print("SKIPPED (unreadable)")
                continue

            X.append(landmarks)
            y.append(sentence)
            print("done")

    X = np.array(X)
    y_raw = np.array(y)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_raw)

    np.save("X_sentences.npy", X)
    np.save("y_sentences.npy", y_encoded)
    np.save("label_classes.npy", label_encoder.classes_)

    print("\n=== DONE ===")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y_encoded.shape}")
    print(f"Classes: {list(label_encoder.classes_)}")


if __name__ == "__main__":
    main()
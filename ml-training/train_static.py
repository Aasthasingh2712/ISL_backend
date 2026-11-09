"""
train_static.py
Trains the static sign classifier with NORMALIZED landmarks
(wrist-relative, scale-invariant) so the model learns hand SHAPE,
not screen position / distance from camera.
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import LabelEncoder

DATA_DIR = "static_dataset"
BATCH_SIZE = 16
HIDDEN_1 = 256
HIDDEN_2 = 128
NUM_EPOCHS = 60
LEARNING_RATE = 0.001

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def normalize_hand(hand_63):
    """Make a single hand's 63-dim landmark vector translation + scale invariant."""
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


class StaticDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class StaticClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, HIDDEN_1)
        self.fc2 = nn.Linear(HIDDEN_1, HIDDEN_2)
        self.fc3 = nn.Linear(HIDDEN_2, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        return self.fc3(x)


def main():
    print(f"Using device: {DEVICE}\n")

    npy_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".npy")])
    if not npy_files:
        print(f"ERROR: No .npy files found in '{DATA_DIR}'.")
        return

    print(f"Found {len(npy_files)} label files: {[f.replace('.npy', '') for f in npy_files]}\n")

    X = []
    y = []

    for file in npy_files:
        label = file.replace(".npy", "")
        data = np.load(os.path.join(DATA_DIR, file))
        print(f"  {label}: {data.shape[0]} samples")
        for sample in data:
            X.append(normalize_features(sample))
            y.append(label)

    X = np.array(X)
    y_raw = np.array(y)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_raw)

    num_classes = len(label_encoder.classes_)
    input_size = X.shape[1]

    print(f"\nTotal samples: {len(X)} | Feature size: {input_size} | Classes: {num_classes}\n")

    dataset = StaticDataset(X, y_encoded)
    val_size = max(1, int(0.15 * len(dataset)))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Train samples: {train_size} | Val samples: {val_size}\n")

    model = StaticClassifier(input_size, num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        train_loss, train_correct = 0.0, 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * X_batch.size(0)
            train_correct += (outputs.argmax(1) == y_batch).sum().item()

        train_loss /= train_size
        train_acc = train_correct / train_size

        model.eval()
        val_loss, val_correct = 0.0, 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)

                val_loss += loss.item() * X_batch.size(0)
                val_correct += (outputs.argmax(1) == y_batch).sum().item()

        val_loss /= val_size
        val_acc = val_correct / val_size

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:3d}/{NUM_EPOCHS} | "
                f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f} | "
                f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}"
            )

    torch.save(model.state_dict(), "static_sign_model.pth")
    np.save("static_label_classes.npy", label_encoder.classes_)

    print("\n=== TRAINING DONE ===")
    print("Model saved as static_sign_model.pth")
    print(f"Classes saved: {list(label_encoder.classes_)}")


if __name__ == "__main__":
    main()
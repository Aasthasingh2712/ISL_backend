"""
train_lstm.py
Trains an LSTM model on preprocessed sentence-level landmark sequences
(X_sentences.npy, y_sentences.npy) for ISL sentence recognition.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

# ----------------------------
# CONFIG
# ----------------------------
X_PATH = "X_sentences.npy"
Y_PATH = "y_sentences.npy"
CLASSES_PATH = "label_classes.npy"

BATCH_SIZE = 4          # small dataset, so small batch
HIDDEN_SIZE = 128
NUM_EPOCHS = 50
LEARNING_RATE = 0.001

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------
# DATASET
# ----------------------------
class SignLanguageDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ----------------------------
# MODEL
# ----------------------------
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
        last_hidden = h_n[-1]  # (batch, hidden_size//2)
        out = self.dropout(last_hidden)
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        return out


# ----------------------------
# MAIN
# ----------------------------
def main():
    print(f"Using device: {DEVICE}\n")

    # Load data
    X = np.load(X_PATH)
    y = np.load(Y_PATH)
    classes = np.load(CLASSES_PATH, allow_pickle=True)

    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Classes ({len(classes)}): {list(classes)}\n")

    num_samples, seq_len, num_features = X.shape
    num_classes = len(classes)

    # Train/val split (small dataset -> simple split)
    dataset = SignLanguageDataset(X, y)
    val_size = max(1, int(0.2 * len(dataset)))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Train samples: {train_size} | Val samples: {val_size}\n")

    # Model, loss, optimizer
    model = SignLSTM(num_features, HIDDEN_SIZE, num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training loop
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

        # Validation
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

    # Save model
    torch.save(model.state_dict(), "sign_lstm_model.pth")
    print("\n=== TRAINING DONE ===")
    print("Model saved as sign_lstm_model.pth")


if __name__ == "__main__":
    main()
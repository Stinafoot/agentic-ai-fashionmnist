"""
train.py — Train a CNN on FashionMNIST and save model.pt
Run: python train.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ── Reproducibility ───────────────────────────────────────────────────────────
torch.manual_seed(42)

# ── Hyper-parameters ──────────────────────────────────────────────────────────
BATCH_SIZE  = 64
EPOCHS      = 5
LR          = 1e-3
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── FashionMNIST labels ───────────────────────────────────────────────────────
CLASSES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]

# ── Data ──────────────────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530,)),   # FashionMNIST channel mean/std
])

train_set = datasets.FashionMNIST(root="./data", train=True,  download=True, transform=transform)
test_set  = datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
test_loader  = DataLoader(test_set,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# ── CNN Model ─────────────────────────────────────────────────────────────────
class FashionCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),                                       # 14×14
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),                                       # 7×7
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ── Training loop ─────────────────────────────────────────────────────────────
def train():
    model     = FashionCNN().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

    print(f"Training on {DEVICE} for {EPOCHS} epochs …")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        scheduler.step()

        # ── Validation ────────────────────────────────────────────────────────
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                preds = model(imgs).argmax(dim=1)
                correct += (preds == labels).sum().item()
                total   += labels.size(0)
        acc = 100.0 * correct / total
        print(f"  Epoch {epoch}/{EPOCHS}  loss={running_loss/len(train_loader):.4f}  test_acc={acc:.2f}%")

    # ── Save ──────────────────────────────────────────────────────────────────
    torch.save(model.state_dict(), "model.pt")
    print("Model saved → model.pt")
    return model


if __name__ == "__main__":
    train()

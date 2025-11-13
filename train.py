# train.py
from pathlib import Path
import json
import argparse

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

from dataset import ProjectFramesDataset
from model import SimpleCNN


def split_dataset(full_ds, val_ratio=0.15, seed=42):
    n = len(full_ds)
    n_val = int(n * val_ratio)
    n_train = n - n_val
    g = torch.Generator().manual_seed(seed)
    return random_split(full_ds, [n_train, n_val], generator=g)


def train_one_epoch(model, loader, criterion, optim, device):
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optim.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optim.step()

        total_loss += loss.item()

    return total_loss / max(1, len(loader))


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == y).sum().item()
        total += y.numel()
    return correct / max(1, total)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    proj_root = Path(__file__).resolve().parents[1]
    data_root = proj_root / "data" / "Project Frames"
    save_dir = proj_root / "saved"
    save_dir.mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] Device:", device)

    full_ds = ProjectFramesDataset(data_root, img_size=args.img_size)
    train_ds, val_ds = split_dataset(full_ds)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    classes = list(full_ds.class_to_idx.keys())
    num_classes = len(classes)
    model = SimpleCNN(num_classes).to(device)

    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    best_acc = -1
    for ep in range(1, args.epochs + 1):
        loss = train_one_epoch(model, train_loader, criterion, optim, device)
        acc = evaluate(model, val_loader, device)
        print(f"Epoch {ep}/{args.epochs}  loss={loss:.4f}  val_acc={acc:.3f}")

        if acc >= best_acc:
            best_acc = acc
            payload = {
                "meta": {
                    "classes": classes,
                    "num_classes": num_classes,
                    "img_size": args.img_size,
                },
                "state_dict": model.state_dict(),
            }
            torch.save(payload, save_dir / "model.pth")
            print("[INFO] Saved best model")

    print("[INFO] Best val_acc:", best_acc)


if __name__ == "__main__":
    main()








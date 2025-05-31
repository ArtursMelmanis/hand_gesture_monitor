import torch
from torch import nn, optim
from tqdm.auto import tqdm
from pathlib import Path
from collections import Counter

from dataset import get_dataloaders
from model   import get_model

DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS   = 5
LR       = 3e-5
CKPT_DIR = Path("models"); CKPT_DIR.mkdir(exist_ok=True, parents=True)


def train():
    train_loader, val_loader, num_classes = get_dataloaders(
        data_dir="data/archive/images",
        num_workers=0,
    )

    model = get_model(num_classes, pretrained=False).to(DEVICE)
    ckpt_path = CKPT_DIR / "best_cnn.pth"
    if ckpt_path.exists():
        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
        print("Loaded checkpoint, continuing fine-tune…")

    # ---------- class-weighted loss ----------
    full_targets = [y for _, y in train_loader.dataset]      # iterable over Subset
    freq = Counter(full_targets)
    class_weights = torch.tensor(
        [1.0 / freq[i] for i in range(num_classes)],
        dtype=torch.float32,
        device=DEVICE
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_val_acc = 0.0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        loss_sum = correct = total = 0

        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}"):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)
            outs = model(imgs)
            loss = criterion(outs, labels)
            loss.backward()
            optimizer.step()

            loss_sum += loss.item() * imgs.size(0)
            preds     = outs.argmax(1)
            correct  += (preds == labels).sum().item()
            total    += labels.size(0)

        tr_loss = loss_sum / total
        tr_acc  = correct / total
        val_loss, val_acc = evaluate(model, val_loader, criterion)

        print(f"[{epoch}] train_loss={tr_loss:.4f} acc={tr_acc:.3f} | "
              f"val_loss={val_loss:.4f} acc={val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), ckpt_path)
            print(f"  ✔ New best saved ({best_val_acc:.3f})")


def evaluate(model, loader, criterion):
    model.eval()
    loss_sum = correct = total = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outs = model(imgs)
            loss_sum += criterion(outs, labels).item() * imgs.size(0)
            preds = outs.argmax(1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
    return loss_sum / total, correct / total


if __name__ == "__main__":
    train()

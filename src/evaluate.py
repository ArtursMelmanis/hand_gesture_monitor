import torch
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

from dataset import get_dataloaders
from model   import get_model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CKPT   = Path("models/best_cnn.pth")

def main():
    _, val_loader, num_classes = get_dataloaders(
        data_dir="data/archive/images",
        batch_size=32,
        num_workers=0
    )

    model = get_model(num_classes, pretrained=False).to(DEVICE)
    model.load_state_dict(torch.load(CKPT, map_location=DEVICE))
    model.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            preds = model(imgs.to(DEVICE)).argmax(1).cpu()
            y_true.extend(labels.numpy())
            y_pred.extend(preds.numpy())

    print(classification_report(y_true, y_pred, digits=3))

    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion matrix"); plt.xlabel("Predicted"); plt.ylabel("True")
    Path("outputs").mkdir(exist_ok=True)
    plt.savefig("outputs/confusion_matrix1.png", dpi=200)
    plt.show()

if __name__ == "__main__":
    main()

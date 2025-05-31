# ── src/demo_viewer.py ─────────────────────────────────────────────
import sys, cv2, torch, random
from collections import defaultdict
import numpy as np
from pathlib import Path
from typing import List
from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout
from PyQt5.QtGui      import QImage, QPixmap
from PyQt5.QtCore     import Qt

from model   import get_model
from dataset import MEAN, STD, IMG_SIZE

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_ROOT = Path("data/archive/images")      # <-- при необходимости поменяйте
MAX_PER_CLASS = 40                           # None → берём ВСЕ файлы

# -------------------------------------------------------------------
def collect_paths(root: Path, limit: int | None) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    samples: list[Path] = []

    for p in root.rglob("*"):
        if p.suffix in exts:
            samples.append(p)

    if limit is not None:
        # рандомно отбираем ≤ limit на класс
        by_class: dict[str, list[Path]] = defaultdict(list)
        for p in samples:
            by_class[p.parent.name].append(p)
            
        limited: list[Path] = []
        for lst in by_class.values():
            limited.extend(random.sample(lst, min(limit, len(lst))))
        samples = limited
    
    random.shuffle(samples)
    return samples

def preprocess(bgr: np.ndarray) -> torch.Tensor | None:
    if bgr is None:
        return None
    img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)).astype("float32") / 255.0
    img = (img - np.array(MEAN, dtype="float32")) / np.array(STD, dtype="float32")
    return (
        torch.tensor(img)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(DEVICE)
        .float()
    )

# -------------------------------------------------------------------
class Viewer(QWidget):
    def __init__(self, paths: list[Path], class_names: list[str], model: torch.nn.Module):
        super().__init__()
        self.paths = paths
        self.class_names = class_names
        self.model = model
        self.idx = 0

        self.setWindowTitle("Gesture Demo Viewer")

        self.img_lbl  = QLabel()
        self.pred_lbl = QLabel()
        self.pred_lbl.setAlignment(Qt.AlignCenter)          # type: ignore[attr-defined]

        lay = QVBoxLayout(self)
        lay.addWidget(self.img_lbl)
        lay.addWidget(self.pred_lbl)

        self.show_sample()

    # клавиши навигации
    def keyPressEvent(self, e):
        if e.key() in (Qt.Key_Right, Qt.Key_Down, Qt.Key_Space):    # type: ignore[attr-defined]
            self.idx = (self.idx + 1) % len(self.paths)
        elif e.key() in (Qt.Key_Left, Qt.Key_Up):                   # type: ignore[attr-defined]
            self.idx = (self.idx - 1) % len(self.paths)
        elif e.key() == Qt.Key_Escape:                              # type: ignore[attr-defined]
            self.close()
            return
        self.show_sample()

    def show_sample(self):
    # ищем читаемое изображение
        for _ in range(len(self.paths)):
            path = self.paths[self.idx]
            bgr  = cv2.imread(str(path))
            t    = preprocess(bgr)
            if t is None:
                print("⚠️  unreadable:", path.name)
                self.idx = (self.idx + 1) % len(self.paths)
                continue
            try:
                with torch.no_grad():
                    probs = torch.softmax(self.model(t), dim=1)[0]
                pred   = int(torch.argmax(probs))
            except Exception as err:
                print("⚠️  inference error on", path.name, "→", err)
                self.idx = (self.idx + 1) % len(self.paths)
                continue
            break
        else:                       # не нашли ни одной рабочей
            self.pred_lbl.setText("No readable images")
            return

        # отображение
        pred_txt = f"{self.class_names[pred]} ({probs[pred]:.1%})"
        rgb  = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, rgb.shape[1], rgb.shape[0], QImage.Format_RGB888)
        self.img_lbl.setPixmap(QPixmap.fromImage(qimg))
        self.pred_lbl.setText(pred_txt)
        self.resize(rgb.shape[1], rgb.shape[0] + 40)   # задаём минимальный размер окна


# -------------------------------------------------------------------
def main():
    img_paths = collect_paths(DATA_ROOT, MAX_PER_CLASS)
    print("Found", len(img_paths), "images.")
    if not img_paths:
        print("❌ No images found, check DATA_ROOT path.")
        sys.exit(1)

    class_names = sorted(p.name for p in DATA_ROOT.iterdir() if p.is_dir())
    model = get_model(len(class_names), pretrained=False).to(DEVICE)
    model.load_state_dict(torch.load("models/best_cnn.pth", map_location=DEVICE))
    model.eval()

    app = QApplication(sys.argv)
    try:
        Viewer(img_paths, class_names, model).show()
        rc = app.exec()
    except Exception as e:
        print("❌ top-level error:", e)
        raise
    sys.exit(rc)

if __name__ == "__main__":
    main()

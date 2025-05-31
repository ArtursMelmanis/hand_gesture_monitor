# src/dataset_stats.py
from pathlib import Path
from collections import OrderedDict

import matplotlib.pyplot as plt

DATA_ROOT = Path("data/archive/images")      # при необходимости поменяй путь
OUT_DIR   = Path("outputs");  OUT_DIR.mkdir(exist_ok=True)

def main():
    if not DATA_ROOT.exists():
        raise FileNotFoundError(DATA_ROOT.resolve())

    # ---------- счёт файлов ----------
    class_counts: OrderedDict[str, int] = OrderedDict()
    for class_dir in sorted(DATA_ROOT.iterdir()):
        if class_dir.is_dir():
            n_imgs = sum(1 for p in class_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
            class_counts[class_dir.name] = n_imgs

    # ---------- вывод ----------
    print(f"{'Class':<20} | Images")
    print("-" * 30)
    for cls, n in class_counts.items():
        print(f"{cls:<20} | {n}")
    print("-" * 30)
    total = sum(class_counts.values())
    print(f"{'TOTAL':<20} | {total}")

    # ---------- bar chart ----------
    plt.figure(figsize=(8, 4))
    plt.bar(list(class_counts.keys()), list(class_counts.values()), color="skyblue")
    plt.title("Image count per class")
    plt.ylabel("Images")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "class_dist.png", dpi=200)
    plt.show()

if __name__ == "__main__":
    main()

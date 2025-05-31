import matplotlib.pyplot as plt, glob, random, cv2
from pathlib import Path
root = Path("data/archive/images")

samples = random.sample(list(root.rglob("*.jpg")), 8)
plt.figure(figsize=(10,5))
for i, p in enumerate(samples, 1):
    img = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
    plt.subplot(2,4,i); plt.imshow(img); plt.title(p.parent.name); plt.axis("off")
plt.tight_layout(); plt.show()
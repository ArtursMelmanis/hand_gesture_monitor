from pathlib import Path
from typing import Tuple, List
from collections import Counter

import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets, transforms

IMG_SIZE   = 224
BATCH_SIZE = 32
VAL_SPLIT  = 0.40
MEAN, STD  = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
SEED       = 42
DATA_DIR   = Path(__file__).parents[1] / "data" / "archive" / "images"

def _split_indices(n: int, val_ratio: float, seed: int) -> Tuple[List[int], List[int]]:
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(n, generator=g).tolist()
    val_len = int(n * val_ratio)
    return idx[val_len:], idx[:val_len]      # train, val


def get_dataloaders(
    data_dir: Path | str = DATA_DIR,
    batch_size: int = BATCH_SIZE,
    img_size: int = IMG_SIZE,
    val_split: float = VAL_SPLIT,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, int]:

    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(data_dir.resolve())

    tfm_train = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    tfm_val = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    base = datasets.ImageFolder(data_dir)
    num_classes = len(base.classes)

    train_idx, val_idx = _split_indices(len(base), val_split, SEED)

    train_ds = Subset(datasets.ImageFolder(data_dir, transform=tfm_train), train_idx)
    val_ds   = Subset(datasets.ImageFolder(data_dir, transform=tfm_val),   val_idx)

    # -------- oversampling -----------
    base_ds = train_ds.dataset
    if not hasattr(base_ds, "targets"):
        raise AttributeError("ImageFolder missing 'targets' attribute")
    base_targets: list[int] = getattr(base_ds, "targets")
    targets = [base_targets[i] for i in train_ds.indices]
    freq    = Counter(targets)
    weights = [1.0 / freq[t] for t in targets]
    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(train_ds),
        replacement=True
    )

    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,         # <-- единственный источник порядка
        num_workers=num_workers,
        pin_memory=pin,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
    )
    return train_loader, val_loader, num_classes

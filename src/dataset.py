import os
from glob import glob
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import datasets, transforms
import numpy as np
from PIL import Image

# Default ImageNet normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class CLAHE:
    """Pre-processing: CLAHE pada kanal L (ruang warna LAB) untuk menaikkan kontras lesi kulit."""
    def __init__(self, clip_limit=2.0, tile=8):
        self.clip_limit, self.tile = clip_limit, tile

    def __call__(self, img):
        import cv2
        lab = cv2.cvtColor(np.asarray(img.convert("RGB")), cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=(self.tile, self.tile))
        lab[..., 0] = clahe.apply(lab[..., 0])
        return Image.fromarray(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB))


def get_transforms(img_size=224, preprocess="none"):
    """
    Mengembalikan pipeline augmentasi data (train) dan preprocessing evaluasi (valid/test).
    Eksplorasi augmentasi mencakup: Random Flip, Random Rotation, Color Jitter, & Normalisasi.
    """
    pre = [CLAHE()] if preprocess == "clahe" else []
    train_transform = transforms.Compose([
        *pre,
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    eval_transform = transforms.Compose([
        *pre,
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    return train_transform, eval_transform


def resolve_dataset_root(data_dir=None):
    """
    Mendeteksi direktori dataset secara adaptif (lokal maupun Kaggle environment).
    """
    candidates = []
    if data_dir:
        candidates.append(data_dir)
    
    candidates.extend([
        "Data",
        "./Data",
        "../Data",
        "/kaggle/input/mpox-skin-lesion-dataset-version-20-msld-v20",
    ])

    for path in candidates:
        if os.path.exists(path):
            return os.path.abspath(path)

    raise FileNotFoundError(
        f"Direktori dataset tidak ditemukan di: {candidates}. Harap tentukan parameter --data_dir."
    )


def find_fold_dir(base_dir, fold, split, aug=False):
    """
    Mencari sub-folder split (Train/Valid/Test) untuk fold tertentu secara rekursif,
    baik untuk dataset Original maupun Augmented.
    """
    split_norm = split.strip().lower()
    fold_marker = f"fold{fold}"

    for root, dirs, _ in os.walk(base_dir):
        # cek "aug"/"fold" hanya pada path relatif terhadap base_dir (bukan path induk dataset)
        root_low = os.path.relpath(root, base_dir).lower().replace(" ", "").replace("_", "")
        base_name_low = os.path.basename(root).strip().lower()

        is_correct_split = (base_name_low == split_norm)
        has_fold = fold_marker in root_low
        is_aug_match = ("aug" in root_low) if aug else ("aug" not in root_low)

        if is_correct_split and has_fold and is_aug_match and dirs:
            return root

    raise FileNotFoundError(
        f"Folder '{split}' untuk Fold {fold} (Augmented={aug}) tidak ditemukan di dalam: {base_dir}"
    )


def get_fold_loaders(data_dir="Data", fold=1, batch_size=16, img_size=224, num_workers=2, preprocess="none"):
    """
    Menyiapkan DataLoader untuk Fold tertentu:
    - Train: Kombinasi Original Images 'Train' + Augmented Images 'Train'
    - Valid: Original Images 'Valid'
    - Test : Original Images 'Test'
    Sesuai ketentuan docx: Valid dan Test TIDAK dicampur ke dalam data training.
    """
    root_dir = resolve_dataset_root(data_dir)
    train_tf, eval_tf = get_transforms(img_size=img_size, preprocess=preprocess)

    # 1. Temukan direktori masing-masing split
    orig_train_dir = find_fold_dir(root_dir, fold=fold, split="Train", aug=False)
    aug_train_dir = find_fold_dir(root_dir, fold=fold, split="Train", aug=True)
    valid_dir = find_fold_dir(root_dir, fold=fold, split="Valid", aug=False)
    test_dir = find_fold_dir(root_dir, fold=fold, split="Test", aug=False)

    # 2. Inisialisasi Dataset
    orig_train_ds = datasets.ImageFolder(orig_train_dir, transform=train_tf)
    aug_train_ds = datasets.ImageFolder(aug_train_dir, transform=train_tf)
    train_dataset = ConcatDataset([orig_train_ds, aug_train_ds])

    valid_dataset = datasets.ImageFolder(valid_dir, transform=eval_tf)
    test_dataset = datasets.ImageFolder(test_dir, transform=eval_tf)

    # Kelas diambil dari urutan folder ImageFolder (konsisten secara alfabetis)
    class_names = orig_train_ds.classes

    # 3. Inisialisasi DataLoader
    # Di Windows, multiprocessing DataLoader bisa membutuhkan num_workers=0 jika ada issue IPC
    nw = num_workers if os.name != "nt" else 0

    # Gunakan pin_memory hanya jika accelerator (CUDA) tersedia
    import torch
    use_pin = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=nw,
        pin_memory=use_pin
    )
    val_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=nw,
        pin_memory=use_pin
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=nw,
        pin_memory=use_pin
    )

    return train_loader, val_loader, test_loader, class_names

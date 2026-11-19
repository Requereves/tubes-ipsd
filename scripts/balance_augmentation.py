import argparse
import os
import random
from glob import glob
import numpy as np
from PIL import Image, ImageFilter
from torchvision import transforms


def add_gaussian_noise(img: Image.Image, mean: float = 0.0, std: float = 12.0) -> Image.Image:
    """Menambahkan Gaussian Noise pada gambar PIL."""
    arr = np.array(img, dtype=np.float32)
    noise = np.random.normal(mean, std, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def get_augmenter():
    """Pipeline augmentasi PIL: Flip horizontal/vertikal, rotasi, color jitter, blur, & noise."""
    jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)

    def augment(img: Image.Image) -> Image.Image:
        # 1. Flip horizontal & vertikal acak
        if random.random() > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() > 0.5:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)

        # 2. Rotasi acak (-30 s/d +30 derajat)
        angle = random.uniform(-30, 30)
        img = img.rotate(angle, resample=Image.BICUBIC, expand=False)

        # 3. Color Jitter
        img = jitter(img)

        # 4. Gaussian Blur acak (probabilitas 30%)
        if random.random() < 0.3:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.2)))

        # 5. Gaussian Noise (probabilitas 50%) - sesuai arahan asprak
        if random.random() < 0.5:
            img = add_gaussian_noise(img, std=random.uniform(8.0, 15.0))

        return img

    return augment


def find_fold_train_dir(base_dir: str, fold: int, aug: bool = False) -> str:
    """Mencari sub-folder split Train untuk fold tertentu."""
    fold_marker = f"fold{fold}"
    for root, dirs, _ in os.walk(base_dir):
        root_low = root.lower().replace(" ", "").replace("_", "")
        base_name_low = os.path.basename(root).strip().lower()

        if base_name_low == "train" and fold_marker in root_low:
            is_aug = "aug" in root_low
            if is_aug == aug and dirs:
                return root
    raise FileNotFoundError(f"Folder Train fold {fold} (aug={aug}) tidak ditemukan di {base_dir}")


def clean_balanced_images(base_dir: str, fold: int = None):
    """Menghapus gambar hasil generate augmentasi seimbang (_balaug_)."""
    folds = [fold] if fold else range(1, 6)
    deleted_total = 0

    for f in folds:
        try:
            aug_train_dir = find_fold_train_dir(base_dir, fold=f, aug=True)
        except FileNotFoundError:
            continue

        pattern = os.path.join(aug_train_dir, "*", "*_balaug_*.*")
        files = glob(pattern)
        for file in files:
            try:
                os.remove(file)
                deleted_total += 1
            except OSError:
                pass

    print(f"Pembersihan selesai: {deleted_total} gambar augmentasi (_balaug_) dihapus.")


def balance_fold(base_dir: str, fold: int, target_count: int = None):
    """Menyeimbangkan data latih 1 Fold dengan menambah gambar augmentasi pada kelas minoritas."""
    orig_train_dir = find_fold_train_dir(base_dir, fold=fold, aug=False)
    aug_train_dir = find_fold_train_dir(base_dir, fold=fold, aug=True)

    classes = sorted([d for d in os.listdir(orig_train_dir) if os.path.isdir(os.path.join(orig_train_dir, d))])
    augment = get_augmenter()

    # Hitung jumlah awal per kelas
    counts = {}
    orig_files_map = {}
    for c in classes:
        orig_files = [
            os.path.join(orig_train_dir, c, f)
            for f in os.listdir(os.path.join(orig_train_dir, c))
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        aug_files = [
            os.path.join(aug_train_dir, c, f)
            for f in os.listdir(os.path.join(aug_train_dir, c))
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        counts[c] = len(orig_files) + len(aug_files)
        orig_files_map[c] = orig_files

    # Jika target_count tidak ditentukan, samakan dengan kelas mayoritas
    majority_count = max(counts.values())
    target = target_count if target_count is not None else majority_count

    print(f"\n{'='*20} PENYEIMBANGAN FOLD {fold} {'='*20}")
    print(f"Target jumlah per kelas: {target} gambar (Mayoritas saat ini: {majority_count})")
    print("-" * 55)
    print(f"{'Kelas':<12} | {'Awal':<8} | {'Dibuat':<8} | {'Total Akhir':<10}")
    print("-" * 55)

    for c in classes:
        current = counts[c]
        needed = target - current

        if needed <= 0:
            print(f"{c:<12} | {current:<8} | {'0 (skip)':<8} | {current:<10}")
            continue

        orig_pool = orig_files_map[c]
        if not orig_pool:
            print(f"[Warning] Tidak ada foto original untuk kelas {c}, dilewati.")
            continue

        target_class_dir = os.path.join(aug_train_dir, c)
        os.makedirs(target_class_dir, exist_ok=True)

        created = 0
        for i in range(needed):
            src_path = random.choice(orig_pool)
            base_stem = os.path.splitext(os.path.basename(src_path))[0]

            with Image.open(src_path) as img:
                img_rgb = img.convert("RGB")
                aug_img = augment(img_rgb)
                out_name = f"{base_stem}_balaug_{i:05d}.jpg"
                out_path = os.path.join(target_class_dir, out_name)
                aug_img.save(out_path, quality=95)
                created += 1

        print(f"{c:<12} | {current:<8} | {created:<8} | {current + created:<10}")

    print("-" * 55)


def main():
    parser = argparse.ArgumentParser(
        description="Targeted Augmentation Script untuk Menyeimbangkan Dataset Imbalance (Saran Asprak)"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="Data",
        help="Path ke folder root dataset (default: 'Data')",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=None,
        choices=[1, 2, 3, 4, 5],
        help="Jalankan hanya pada fold tertentu (1-5). Jika kosong, jalankan pada semua 5 fold.",
    )
    parser.add_argument(
        "--target_count",
        type=int,
        default=None,
        help="Target jumlah gambar per kelas (default: menyamakan dengan kelas mayoritas)",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Hapus semua gambar augmentasi yang pernah digenerate oleh script ini (_balaug_)",
    )

    args = parser.parse_args()

    if args.clean:
        clean_balanced_images(args.data_dir, fold=args.fold)
        return

    folds = [args.fold] if args.fold else list(range(1, 6))
    for f in folds:
        balance_fold(args.data_dir, fold=f, target_count=args.target_count)

    print("\n[SUKSES] Seluruh kelas berhasil diseimbangkan.")
    print("DataLoader di Train.py akan otomatis mendeteksi gambar tambahan ini tanpa ubah kode.")


if __name__ == "__main__":
    main()

import argparse
import os
import sys

from src.train_eval import run_5fold_cv, train_and_eval_fold, get_device
import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Mpox Skin Lesion Classification - 5-Fold Cross Validation"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="regnet_y_800mf",
        choices=["regnet_y_800mf", "convnext_tiny", "custom_cnn", "efficientnet_b0"],
        help="Arsitektur model deep learning (selain baseline Kaggle ResNet50/VGG16)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "cpu"],
        help="Pilih device eksekusi secara eksplisit (default: otomatis deteksi GPU jika tersedia)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="Data",
        help="Path ke direktori dataset (lokal atau Kaggle)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=15,
        help="Jumlah epoch per fold",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size dataloader",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate (default: 1e-3 untuk custom_cnn, 5e-5 untuk model pretrained)",
    )
    parser.add_argument(
        "--preprocess",
        type=str,
        default="none",
        choices=["none", "clahe"],
        help="Pre-processing gambar: none (resize saja) atau clahe",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--img_size",
        type=int,
        default=224,
        help="Resolusi dimensi citra (tinggi dan lebar)",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="Results",
        help="Direktori penyimpanan grafik evaluasi dan ringkasan metrik",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
        help="Jumlah worker DataLoader",
    )
    parser.add_argument(
        "--no_pretrained",
        action="store_true",
        help="Nonaktifkan bobot pretrained ImageNet jika menggunakan model pretrained",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=None,
        choices=[1, 2, 3, 4, 5],
        help="Jalankan hanya 1 fold spesifik (1-5) untuk testing cepat. Kosongkan untuk menjalankan 5-fold CV penuh.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.pretrained = not args.no_pretrained
    if args.lr is None:
        args.lr = 1e-3 if args.model == "custom_cnn" else 5e-5

    import random
    import numpy as np
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    # Pastikan folder output ada
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    if args.fold is not None:
        device, _ = get_device(args.device)
        print(f"Menjalankan evaluasi Fold {args.fold}...")
        train_and_eval_fold(args.fold, args, device)
    else:
        run_5fold_cv(args)


if __name__ == "__main__":
    main()

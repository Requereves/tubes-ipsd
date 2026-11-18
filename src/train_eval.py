import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.dataset import get_fold_loaders
from src.model import build_model
from src.utils import (
    compute_metrics,
    plot_learning_curves,
    plot_confusion_matrix,
    plot_multiclass_roc,
    save_summary_report,
)


def get_device(requested_device=None):
    """
    Mendeteksi dan menyiapkan device (GPU / CPU) beserta informasinya.
    """
    if requested_device:
        device = torch.device(requested_device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        if not torch.cuda.is_available():
            print("[PERINGATAN] CUDA diminta tetapi tidak tersedia di PyTorch! Beralih ke CPU.")
            return torch.device("cpu"), False
        
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"[DEVICE] Menggunakan GPU: {gpu_name} ({vram_gb:.2f} GB VRAM)")
        torch.backends.cudnn.benchmark = True
        return device, True
    else:
        print("[DEVICE] Menggunakan CPU. (Catatan: Untuk menggunakan GPU, pastikan PyTorch versi CUDA terpasang)")
        return device, False


def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None):
    """
    Menjalankan 1 epoch training dengan dukungan Mixed Precision (AMP) untuk efisiensi VRAM GPU.
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    use_amp = (scaler is not None and device.type == "cuda")

    for images, targets in loader:
        images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        optimizer.zero_grad()

        if use_amp:
            with torch.amp.autocast(device_type="cuda"):
                outputs = model(images)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)

    epoch_loss = total_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def evaluate(model, loader, criterion, device):
    """
    Evaluasi model: menghitung loss, mengumpulkan y_true, y_pred, dan y_prob.
    """
    model.eval()
    total_loss = 0.0
    total = 0

    all_targets = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)

            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            total_loss += loss.item() * images.size(0)
            total += targets.size(0)

            all_targets.extend(targets.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())

    eval_loss = total_loss / total
    return eval_loss, all_targets, all_preds, all_probs


def train_and_eval_fold(fold, args, device):
    """
    Melatih 1 Fold dan mengevaluasinya pada test set:
    1. Melatih pada Train (Original + Augmented)
    2. Validasi pada Valid (Original) untuk early stopping / checkpointing best model
    3. Menguji bobot terbaik pada Test (Original) dan menghitung metrik lengkap
    """
    print(f"\n{'='*25} MULAI FOLD {fold} / 5 {'='*25}")

    # 1. Dapatkan DataLoader
    train_loader, val_loader, test_loader, class_names = get_fold_loaders(
        data_dir=args.data_dir,
        fold=fold,
        batch_size=args.batch_size,
        img_size=args.img_size,
        num_workers=args.num_workers,
        preprocess=getattr(args, "preprocess", "none"),
    )
    num_classes = len(class_names)

    # 2. Inisialisasi Model, Loss, Optimizer, Scheduler
    model = build_model(
        model_name=args.model,
        num_classes=num_classes,
        pretrained=args.pretrained
    ).to(device)

    # CrossEntropyLoss dengan label smoothing untuk mereduksi overconfidence
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_val_loss = float("inf")
    ckpt_dir = "checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)
    best_model_path = os.path.join(ckpt_dir, f"{args.model}_fold{fold}_best.pt")

    history = {"train_loss": [], "train_acc": [], "valid_loss": [], "valid_acc": []}

    # 3. Training Loop dengan GradScaler jika di CUDA
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler=scaler
        )
        val_loss, val_targets, val_preds, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        val_acc = (torch.tensor(val_preds) == torch.tensor(val_targets)).float().mean().item()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["valid_loss"].append(val_loss)
        history["valid_acc"].append(val_acc)

        print(
            f"Fold {fold} | Epoch {epoch:02d}/{args.epochs:02d} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}"
        )

        # Simpan model terbaik berdasarkan validation loss terendah
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)

    # 4. Simpan plot kurva pembelajaran
    plot_learning_curves(history, fold=fold, save_dir=args.results_dir, model_name=args.model)

    # 5. Evaluasi Final pada TEST SET menggunakan checkpoint terbaik
    print(f"\n[Fold {fold}] Mengevaluasi checkpoint terbaik pada TEST SET...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    test_loss, test_targets, test_preds, test_probs = evaluate(
        model, test_loader, criterion, device
    )

    test_metrics = compute_metrics(
        test_targets, test_preds, test_probs, num_classes=num_classes
    )
    test_metrics["fold"] = fold
    test_metrics["test_loss"] = test_loss

    print(
        f"Hasil Test Fold {fold} -> "
        f"Acc: {test_metrics['accuracy']:.4f} | "
        f"Prec: {test_metrics['precision_macro']:.4f} | "
        f"Rec: {test_metrics['recall_macro']:.4f} | "
        f"F1: {test_metrics['f1_macro']:.4f} | "
        f"ROC-AUC: {test_metrics['roc_auc_macro']:.4f}"
    )

    # Simpan Gambar Hasil Evaluasi (Confusion Matrix & ROC-AUC) untuk Fold ini
    plot_confusion_matrix(
        test_metrics["confusion_matrix"],
        class_names=class_names,
        fold=fold,
        save_dir=args.results_dir,
        model_name=args.model,
    )
    plot_multiclass_roc(
        test_targets,
        test_probs,
        class_names=class_names,
        fold=fold,
        save_dir=args.results_dir,
        model_name=args.model,
    )

    return test_metrics


def run_5fold_cv(args):
    """
    Orkestrasi 5-Fold Cross Validation penuh sesuai perintah tugas.
    Menghasilkan rerata dan standar deviasi dari semua metrik pada test set.
    """
    device, is_gpu = get_device(getattr(args, "device", None))
    print(f"Model: {args.model} | Epochs: {args.epochs} | Batch Size: {args.batch_size}")

    all_fold_metrics = []

    for fold in range(1, 6):
        fold_res = train_and_eval_fold(fold, args, device)
        all_fold_metrics.append(fold_res)

    print("\n" + "=" * 60)
    print("RINGKASAN 5-FOLD CROSS VALIDATION (TEST SET)")
    print("=" * 60)

    summary_df = save_summary_report(
        all_fold_metrics, save_dir=args.results_dir, model_name=args.model
    )
    print(summary_df.to_string(index=False))

    return summary_df

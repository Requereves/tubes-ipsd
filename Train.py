import os, sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.optim import Adam
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, ConfusionMatrixDisplay

from Utils.getsData import get_train, get_test   
from Models import get_model, _REGISTRY          

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_FOLDS = 5


def get_config(model_name):
    from_scratch = model_name in ("simple", "cnn")
    return {
        "batch_size": 32 if from_scratch else 16,
        "epoch": 30 if from_scratch else 10,
        "lr": 1e-3 if from_scratch else 1e-4,
    }


def train(model_name):
    cfg = get_config(model_name)
    for fold in range(1, N_FOLDS + 1):
        train_loader, valid_loader = get_train(fold, cfg["batch_size"])
        model = get_model(model_name).to(dev)
        optimizer = Adam(model.parameters(), lr=cfg["lr"])
        loss_function = nn.CrossEntropyLoss()

        loss_train_all, loss_valid_all = [], []
        best_loss = float("inf")
        for epoch in range(cfg["epoch"]):
            train_loss, valid_loss = 0, 0

            model.train()
            for src, trg in train_loader:
                src, trg = src.to(dev), trg.to(dev)
                pred = model(src)
                loss = loss_function(pred, trg)
                train_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                for src, trg in valid_loader:
                    src, trg = src.to(dev), trg.to(dev)
                    valid_loss += loss_function(model(src), trg).item()

            loss_train_all.append(train_loss / len(train_loader))
            loss_valid_all.append(valid_loss / len(valid_loader))
            print(f"{model_name} | fold {fold} | epoch {epoch+1} | "
                  f"train loss {loss_train_all[-1]:.4f} | valid loss {loss_valid_all[-1]:.4f}")

            if loss_valid_all[-1] < best_loss:
                best_loss = loss_valid_all[-1]
                torch.save(model.state_dict(), f"checkpoints/{model_name}_fold{fold}.pt")

        plt.figure()
        plt.plot(range(cfg["epoch"]), loss_train_all, color="#931a00", label="Training")
        plt.plot(range(cfg["epoch"]), loss_valid_all, color="#3399e6", label="Validation")
        plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()
        plt.title(f"{model_name} - Fold {fold}")
        plt.savefig(f"results/{model_name}_training_fold{fold}.png")
        plt.close()


@torch.no_grad()
def test(model_name):
    """Load checkpoint terbaik tiap fold (model dibuat lewat get_model di __init__) lalu evaluasi di data test."""
    cfg = get_config(model_name)
    test_loader = get_test(cfg["batch_size"])
    rows = []

    for fold in range(1, N_FOLDS + 1):
        ckpt = f"checkpoints/{model_name}_fold{fold}.pt"
        if not os.path.exists(ckpt):
            print(f"[skip] {ckpt} tidak ditemukan")
            continue

        model = get_model(model_name).to(dev)
        model.load_state_dict(torch.load(ckpt, map_location=dev))
        model.eval()

        y_true, y_pred = [], []
        for src, trg in test_loader:
            pred = model(src.to(dev)).argmax(dim=1).cpu().numpy()
            y_pred.extend(pred)
            y_true.extend(trg.numpy())
        y_true, y_pred = np.array(y_true), np.array(y_pred)

        acc = accuracy_score(y_true, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
        rows.append({"model": model_name, "fold": fold, "accuracy": acc,
                     "precision": prec, "recall": rec, "f1": f1})
        print(f"[TEST] {model_name} | fold {fold} | acc {acc:.4f} | prec {prec:.4f} | rec {rec:.4f} | f1 {f1:.4f}")

        ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred)).plot(cmap="Blues", colorbar=False)
        plt.title(f"{model_name} - Fold {fold} (test)")
        plt.savefig(f"results/{model_name}_cm_fold{fold}.png")
        plt.close()

    return rows


def save_test_summary(all_rows):
    if not all_rows:
        print("Tidak ada checkpoint yang bisa dites.")
        return
    df = pd.DataFrame(all_rows)
    df.to_csv("results/test_per_fold.csv", index=False)

    metrics = ["accuracy", "precision", "recall", "f1"]
    summary = df.groupby("model")[metrics].agg(["mean", "std"]).round(4)
    summary.to_csv("results/test_summary.csv")
    print("\n===== RINGKASAN TEST (mean, std dari semua fold) =====")
    print(summary.to_string())

    df.groupby("model")[metrics].mean().plot(kind="bar", figsize=(9, 5), ylim=(0, 1))
    plt.title("Perbandingan model (rata-rata fold)")
    plt.ylabel("Score"); plt.xticks(rotation=0); plt.tight_layout()
    plt.savefig("results/test_comparison.png")
    plt.close()


def main():
    # python Train.py convnext          -> train + test model convnext
    # python Train.py convnext test     -> test saja (pakai checkpoint yang sudah ada)
    # python Train.py all test          -> test semua model di registry
    # python Train.py all               -> train + test semua model
    arg = sys.argv[1] if len(sys.argv) > 1 else "convnext"
    test_only = len(sys.argv) > 2 and sys.argv[2] == "test"
    names = list(_REGISTRY) if arg == "all" else [arg]

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    all_rows = []
    for name in names:
        if not test_only:
            train(name)
        all_rows += test(name)

    save_test_summary(all_rows)


if __name__ == "__main__":
    main()
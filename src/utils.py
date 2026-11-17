import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    auc,
)
from sklearn.preprocessing import label_binarize


def compute_metrics(y_true, y_pred, y_prob, num_classes=6):
    """
    Menghitung seluruh metrik evaluasi yang diwajibkan oleh docx:
    Accuracy, Precision, Recall, F1-Score, dan ROC-AUC (Macro OVR).
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    acc = accuracy_score(y_true, y_pred)
    prec_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)

    prec_weighted = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec_weighted = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    # ROC-AUC multi-class (One-vs-Rest)
    try:
        if num_classes == 2:
            roc_auc = roc_auc_score(y_true, y_prob[:, 1])
        else:
            roc_auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
    except Exception:
        roc_auc = float("nan")

    cm = confusion_matrix(y_true, y_pred)

    return {
        "accuracy": float(acc),
        "precision_macro": float(prec_macro),
        "recall_macro": float(rec_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(prec_weighted),
        "recall_weighted": float(rec_weighted),
        "f1_weighted": float(f1_weighted),
        "roc_auc_macro": float(roc_auc),
        "confusion_matrix": cm,
    }


def plot_learning_curves(history, fold, save_dir="Results", model_name="model"):
    """
    Menyimpan grafik Loss dan Akurasi (Train vs Valid) per epoch.
    """
    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot Loss
    axes[0].plot(epochs, history["train_loss"], label="Train Loss", color="#d9534f", lw=2)
    axes[0].plot(epochs, history["valid_loss"], label="Valid Loss", color="#0275d8", lw=2)
    axes[0].set_title(f"Loss Curve - Fold {fold} ({model_name})")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # Plot Accuracy
    if "train_acc" in history and "valid_acc" in history:
        axes[1].plot(epochs, history["train_acc"], label="Train Acc", color="#5cb85c", lw=2)
        axes[1].plot(epochs, history["valid_acc"], label="Valid Acc", color="#f0ad4e", lw=2)
        axes[1].set_title(f"Accuracy Curve - Fold {fold} ({model_name})")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
        axes[1].legend()
        axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    out_path = os.path.join(save_dir, f"{model_name}_learning_curve_fold{fold}.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path


def plot_confusion_matrix(cm, class_names, fold, save_dir="Results", model_name="model"):
    """
    Menyimpan heatmap Confusion Matrix pada set Test untuk fold tertentu.
    """
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(8, 6))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
    )
    plt.title(f"Confusion Matrix (Test Set) - Fold {fold}\nModel: {model_name}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    out_path = os.path.join(save_dir, f"{model_name}_confusion_matrix_fold{fold}.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path


def plot_multiclass_roc(y_true, y_prob, class_names, fold, save_dir="Results", model_name="model"):
    """
    Menyimpan kurva ROC multiclass (One-vs-Rest) untuk set Test per fold.
    """
    os.makedirs(save_dir, exist_ok=True)
    num_classes = len(class_names)
    y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))

    plt.figure(figsize=(9, 7))

    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        roc_score = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=1.5, label=f"{class_names[i]} (AUC = {roc_score:.3f})")

    plt.plot([0, 1], [0, 1], "k--", lw=1.5, label="Chance Level")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curves (Test Set) - Fold {fold}\nModel: {model_name}")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    out_path = os.path.join(save_dir, f"{model_name}_roc_curve_fold{fold}.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path


def save_summary_report(fold_metrics_list, save_dir="Results", model_name="model"):
    """
    Menghitung mean & std dari 5-fold cross validation pada test set,
    serta menyimpan hasilnya dalam format CSV dan Markdown.
    """
    os.makedirs(save_dir, exist_ok=True)

    records = []
    for item in fold_metrics_list:
        row = {
            "Fold": item["fold"],
            "Accuracy": item["accuracy"],
            "Precision (Macro)": item["precision_macro"],
            "Recall (Macro)": item["recall_macro"],
            "F1-Score (Macro)": item["f1_macro"],
            "ROC-AUC (Macro)": item["roc_auc_macro"],
        }
        records.append(row)

    df = pd.DataFrame(records)

    # Hitung rata-rata dan standar deviasi
    mean_row = df.mean(numeric_only=True).to_dict()
    std_row = df.std(numeric_only=True).to_dict()

    mean_row["Fold"] = "Mean"
    std_row["Fold"] = "Std"

    df_summary = pd.concat([df, pd.DataFrame([mean_row, std_row])], ignore_index=True)

    # Simpan CSV
    csv_path = os.path.join(save_dir, f"{model_name}_5fold_metrics_summary.csv")
    df_summary.to_csv(csv_path, index=False)

    # Simpan Markdown untuk kemudahan copy ke readme.md
    md_path = os.path.join(save_dir, f"{model_name}_5fold_metrics_summary.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# Evaluasi 5-Fold Cross Validation - Model: {model_name}\n\n")
        f.write(df_summary.to_markdown(index=False))
        f.write("\n")

    return df_summary

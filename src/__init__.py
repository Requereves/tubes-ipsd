# Package initialization for src
from src.dataset import get_fold_loaders, get_transforms, resolve_dataset_root
from src.model import build_model, CustomCNN
from src.train_eval import run_5fold_cv, train_and_eval_fold
from src.utils import compute_metrics, plot_learning_curves, plot_confusion_matrix, plot_multiclass_roc

__all__ = [
    "get_fold_loaders",
    "get_transforms",
    "resolve_dataset_root",
    "build_model",
    "CustomCNN",
    "run_5fold_cv",
    "train_and_eval_fold",
    "compute_metrics",
    "plot_learning_curves",
    "plot_confusion_matrix",
    "plot_multiclass_roc",
]

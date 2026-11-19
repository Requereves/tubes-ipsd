# Klasifikasi Mpox Skin Lesion (MSLD v2.0): ConvNeXt-Tiny vs Custom CNN

Tugas Minggu 10 – Studi Kasus Pembelajaran Mesin.
Mengklasifikasikan citra kulit ke 6 kelas (Chickenpox, Cowpox, Healthy, HFMD, Measles, Monkeypox) agar carrier Mpox bisa dideteksi lebih awal.

Dataset: [Mpox Skin Lesion Dataset v2.0](https://www.kaggle.com/datasets/joydippaul/mpox-skin-lesion-dataset-version-20-msld-v20/data) (tidak diunggah ke repo ini).

## Struktur repo
```
train.py            # entry point (argparse)
src/dataset.py      # DataLoader per fold + pre-processing + augmentasi
src/model.py        # ConvNeXt-Tiny, RegNetY-800MF, Custom CNN, EfficientNet-B0
src/train_eval.py   # loop training, validasi, testing, 5-fold CV
src/utils.py        # metrik, plot learning curve / confusion matrix / ROC, ringkasan
Results/            # gambar & tabel hasil
```

## Metode
| Komponen | Pilihan |
|---|---|
| Model utama | ConvNeXt-Tiny (pretrained ImageNet, head diganti 6 kelas) |
| Model pembanding | RegNetY-800MF (pretrained, head diganti 6 kelas) dan Custom CNN 4 blok konv (BatchNorm, Dropout 0.4), dilatih dari nol |
| Pre-processing | Resize 224×224 + normalisasi ImageNet (CLAHE tersedia sebagai opsi `--preprocess clahe`, tidak dipakai pada hasil di bawah) |
| Augmentasi online | H/V flip, rotasi ±15°, color jitter (hanya data train) |
| Loss | CrossEntropyLoss (label smoothing 0.1) |
| Optimizer | AdamW (lr 5e-5 untuk model pretrained / 1e-3 untuk Custom CNN, weight decay 1e-2) + CosineAnnealing |
| Epoch / batch | 15 / 16, mixed precision (AMP) di GPU |
| Data train | Original `Train` + Augmented `Train` per fold |
| Validasi | Original `Valid` (bobot terbaik berdasarkan validasi dipakai untuk pengujian) |
| Testing | Original `Test` (evaluasi akhir) |
| Evaluasi | Confusion Matrix, Accuracy, Precision, Recall, F1 (macro), ROC-AUC (OvR macro) |

`Valid` dan `Test` tidak pernah dipakai untuk training.

## Cara menjalankan
```bash
pip install -r requirements.txt
D=/kaggle/input/mpox-skin-lesion-dataset-version-20-msld-v20   # folder yang berisi Original Images & Augmented Images

python train.py --model convnext_tiny   --data_dir $D --results_dir Results/convnext_tiny
python train.py --model regnet_y_800mf  --data_dir $D --results_dir Results/regnet_y_800mf
python train.py --model custom_cnn      --data_dir $D --results_dir Results/custom_cnn
```
Tes cepat 1 fold: tambahkan `--fold 1 --epochs 2`.
Bobot model disimpan di `checkpoints/` dan di-ignore oleh git.

## Hasil (rerata ± std 5-fold, data Test)
> ISI setelah training selesai, salin dari `Results/<run>/<model>_5fold_metrics_summary.md`

| Metrik | ConvNeXt-Tiny | RegNetY-800MF | Custom CNN |
|---|---|---|---|
| Accuracy | 0.844854 ± 0.029537 | 0.8107 ± 0.0225 | 0.6081|
| Precision (macro) | 0.862267 ± 0.048321 | | 0.5653 |
| Recall (macro) | 0.800044 ± 0.047329 | | 0.4816 |
| F1-score (macro) | 0.814741 ± 0.045745 | | 0.4738 |
| ROC-AUC (macro OvR) | 0.961375 ± 0.015657 | | 0.8421 |  

### Gambar
> Ganti nama file sesuai hasil di folder `Results/`
![Confusion matrix fold 1](Results/convnext_tiny/convnext_tiny_confusion_matrix_fold1.png)
![Learning curve fold 1](Results/convnext_tiny/convnext_tiny_learning_curve_fold1.png)
![ROC fold 1](Results/convnext_tiny/convnext_tiny_roc_curve_fold1.png)

## Analisis
-ConvNeXt-Tiny paling menjanjikan: rata-rata accuracy 86,38% dan ROC-AUC 97,20% pada 5-fold. Fold terbaik adalah fold 5 (accuracy 92,06%). Standar deviasi ROC-AUC yang kecil (0,83%) menunjukkan kemampuan diskriminasi yang konsisten antar fold. Pada fold 3, train accuracy mendekati 1,00 sedangkan valid accuracy sekitar 0,80 dan valid loss naik.

-RegNetY-800MF mencapai rata-rata train accuracy 96,50% tetapi val accuracy hanya 81,07%. Selisih sekitar 15 poin ini mengindikasikan overfitting.
Custom CNN (dilatih dari nol, Fold 1) memperoleh accuracy 60,81% dan F1-score 47,38%. Train accuracy naik dari 42,93% ke 72,30% dengan valid accuracy tertinggi 55,56%, sehingga model sudah belajar pola tetapi performanya masih terbatas.

-Pretrained vs dari nol: model pretrained (ConvNeXt-Tiny dan RegNetY-800MF) lebih baik daripada Custom CNN, sehingga pemilihan arsitektur berpengaruh pada performa klasifikasi lesi kulit.

-Keterbatasan: data evaluasi RegNetY-800MF dan Custom CNN belum lengkap di semua fold. Langkah selanjutnya: evaluasi 5-fold penuh dan optimasi parameter.

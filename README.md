# Perbandingan Arsitektur CNN pada Dataset CIFAR-10

Proyek ini membandingkan kinerja berbagai arsitektur Convolutional Neural Networks (CNN) untuk klasifikasi gambar pada dataset CIFAR-10. Implementasi ini mencakup perbandingan dari tiga arsitektur populer: VGG16, ResNet50, dan MobileNetV2.

## Deskripsi Dataset

Dataset CIFAR-10 terdiri dari 60.000 gambar berwarna berukuran 32x32 piksel, dibagi menjadi 10 kelas:
- Pesawat (airplane)
- Mobil (automobile)
- Burung (bird)
- Kucing (cat)
- Rusa (deer)
- Anjing (dog)
- Katak (frog)
- Kuda (horse)
- Kapal (ship)
- Truk (truck)

Dataset ini dibagi menjadi 50.000 gambar untuk pelatihan dan 10.000 gambar untuk pengujian.

## Arsitektur CNN yang Dibandingkan

1. **VGG16**
   - Arsitektur yang dalam dengan 16 lapisan berbobot
   - Menggunakan filter konvolusi 3x3 berulang
   - Karakteristik: Sederhana namun dalam

2. **ResNet50**
   - Arsitektur dengan 50 lapisan dan koneksi residual (skip connection)
   - Mengatasi masalah vanishing gradient pada jaringan yang sangat dalam
   - Karakteristik: Dalam dengan koneksi residual

3. **MobileNetV2**
   - Arsitektur yang ringan dan efisien untuk perangkat mobile
   - Menggunakan konvolusi depthwise separable untuk mengurangi parameter
   - Karakteristik: Ringan, efisien, cocok untuk perangkat dengan sumber daya terbatas

## Implementasi

Implementasi mencakup:
- Preprocessing data (normalisasi dan augmentasi)
- Transfer learning menggunakan model pre-trained pada ImageNet
- Pelatihan dan evaluasi model
- Analisis perbandingan metrik (akurasi, waktu pelatihan, waktu inferensi, dll.)
- Visualisasi hasil

## Dependensi

Proyek ini membutuhkan pustaka-pustaka berikut:
- TensorFlow 2.x
- NumPy
- Matplotlib
- Pandas
- Scikit-learn
- Seaborn

## Cara Menjalankan

1. Pastikan semua dependensi telah terpasang:
   ```
   pip install -r requirements.txt
   ```

2. Ada beberapa cara untuk menjalankan eksperimen:

   a. Menjalankan seluruh pipeline (preprocessing, pelatihan, dan analisis) dengan ukuran asli 32x32:
   ```
   python run_experiment.py --original_size --epoch 5
   ```

   b. Menjalankan seluruh pipeline dengan ukuran gambar yang diubah menjadi 224x224 (untuk model pretrained):
   ```
   python run_experiment.py
   ```

   c. Menjalankan seluruh pipeline dengan jumlah epoch tertentu:
   ```
   python run_experiment.py --epochs 10
   ```

   d. Hanya menjalankan analisis hasil (tanpa pelatihan ulang):
   ```
   python run_experiment.py --only_analysis
   ```

3. Alternatif, Anda juga dapat menjalankan skrip secara terpisah:
   - Preprocessing data:
     ```
     python preprocess.py
     ```
   - Pelatihan dan evaluasi model:
     ```
     python cnn_comparison.py
     ```
   - Analisis hasil:
     ```
     python analyze_results.py
     ```

## Struktur Proyek

```
.
├── README.md              # Dokumentasi proyek
├── requirements.txt       # Dependensi
├── run_experiment.py      # Script utama untuk menjalankan seluruh pipeline
├── preprocess.py          # Script untuk preprocessing dataset
├── cnn_comparison.py      # Script untuk pelatihan dan evaluasi model
├── analyze_results.py     # Script untuk analisis lanjutan dari hasil pelatihan
├── report.md              # Laporan perbandingan arsitektur CNN
├── models/                # Direktori untuk menyimpan model terlatih
└── results/               # Direktori untuk menyimpan hasil visualisasi dan analisis
```

## Hasil

Setelah menjalankan eksperimen, beberapa file hasil akan dihasilkan di direktori `results/`:
- `training_history.png`: Grafik pelatihan yang menunjukkan akurasi dan loss
- `confusion_matrix_[model_name].png`: Confusion matrix untuk setiap model
- `predictions_visualization.png`: Visualisasi prediksi dari model terbaik
- `metrics_comparison.png`: Perbandingan metrik antar model
- `model_comparison.csv`: File CSV yang berisi hasil perbandingan metrik
- `data_samples.png`: Visualisasi sampel dataset
- `data_augmentation_examples.png`: Visualisasi hasil augmentasi data
- Visualisasi analisis tambahan dari `analyze_results.py`

Model terlatih juga akan disimpan di direktori `models/`. 
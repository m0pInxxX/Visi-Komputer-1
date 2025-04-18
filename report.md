# Laporan Perbandingan Arsitektur CNN pada Dataset CIFAR-10

## Pendahuluan

Convolutional Neural Networks (CNN) telah menjadi arsitektur pilihan untuk berbagai tugas pengolahan gambar karena kemampuannya untuk menangkap fitur spasial secara hierarkis. Dalam laporan ini, kami membandingkan tiga arsitektur CNN populer—VGG16, ResNet50, dan MobileNetV2—pada dataset klasifikasi CIFAR-10 untuk mengevaluasi kinerja mereka dalam hal akurasi, efisiensi komputasi, dan metrik lainnya.

## Dataset CIFAR-10

Dataset CIFAR-10 terdiri dari 60.000 gambar berwarna 32×32 pixel yang terbagi dalam 10 kelas objek yang berbeda. Tiap kelas direpresentasikan oleh 6.000 gambar. Data dibagi menjadi 50.000 gambar untuk pelatihan dan 10.000 gambar untuk pengujian. Kelas-kelas dalam dataset ini adalah:

1. Pesawat terbang (airplane)
2. Mobil (automobile)
3. Burung (bird)
4. Kucing (cat)
5. Rusa (deer)
6. Anjing (dog)
7. Katak (frog)
8. Kuda (horse)
9. Kapal (ship)
10. Truk (truck)

## Arsitektur CNN yang Dibandingkan

### 1. VGG16

**Karakteristik:**
- Arsitktur dalam dengan 16 lapisan berbobot (13 lapisan konvolusi + 3 lapisan fully connected)
- Filter konvolusi kecil (3x3) yang digunakan secara konsisten
- Jaringan yang relatif sederhana tapi dalam
- Jumlah parameter besar (~138 juta)

**Kelebihan:**
- Arsitektur yang sederhana dan mudah dipahami
- Kinerja yang baik pada banyak dataset klasifikasi gambar

**Kekurangan:**
- Membutuhkan banyak memori dan komputasi
- Waktu inferensi yang relatif lambat

### 2. ResNet50

**Karakteristik:**
- Menggunakan koneksi residual (skip connection) untuk mengatasi masalah vanishing gradient
- 50 lapisan dalam dengan blok residual
- Lebih dalam dari VGG16 namun dengan parameter yang lebih sedikit (~25 juta)

**Kelebihan:**
- Mampu melatih jaringan yang sangat dalam
- Memiliki akurasi yang tinggi pada dataset kompleks
- Lebih efisien dari VGG16 dalam hal ukuran model

**Kekurangan:**
- Lebih kompleks untuk diimplementasikan dan dipahami
- Masih membutuhkan komputasi yang cukup intensif

### 3. MobileNetV2

**Karakteristik:**
- Dirancang khusus untuk perangkat mobile dan edge computing
- Menggunakan konvolusi depthwise separable untuk mengurangi parameter
- Blok inverted residual dengan linear bottleneck
- Jumlah parameter jauh lebih kecil (~3,5 juta)

**Kelebihan:**
- Sangat efisien dalam hal ukuran model dan kecepatan inferensi
- Cocok untuk deployment di perangkat dengan sumber daya terbatas
- Konsumsi memori yang rendah

**Kekurangan:**
- Akurasi yang lebih rendah dibandingkan model yang lebih besar
- Trade-off antara efisiensi dan akurasi

## Metodologi

### Preprocessing dan Augmentasi Data

Semua gambar dalam dataset CIFAR-10 telah melalui preprocessing dasar:
- Normalisasi nilai pixel (pembagian dengan 255 untuk mendapatkan rentang 0-1)
- One-hot encoding untuk label kelas

Teknik augmentasi data yang diterapkan:
- Rotasi acak (±15°)
- Pergeseran horizontal dan vertikal (±10%)
- Horizontal flip
- Zoom acak (±10%)

### Transfer Learning

Semua arsitektur diimplementasikan menggunakan pendekatan transfer learning:
- Model dasar dimuat dengan bobot pre-trained dari ImageNet
- Layer klasifikasi asli dihapus dan diganti dengan layer baru yang disesuaikan untuk 10 kelas CIFAR-10
- Model dasar dibekukan (tidak dilatih) untuk memanfaatkan representasi fitur yang telah dipelajari sebelumnya

### Pelatihan

Pada setiap arsitektur:
- Optimizer: Adam
- Loss function: Categorical Cross-Entropy
- Batch size: 64
- Jumlah epoch maksimum: 20 dengan early stopping (patience=5)
- Checkpoint model untuk menyimpan model dengan validasi terbaik

## Evaluasi dan Hasil

### Metrik Evaluasi

Kinerja model dievaluasi menggunakan metrik berikut:
- Akurasi: Persentase gambar yang diklasifikasikan dengan benar
- Precision: Rasio prediksi positif yang benar terhadap semua prediksi positif
- Recall: Rasio prediksi positif yang benar terhadap semua kelas positif aktual
- F1-score: Rata-rata harmonik dari precision dan recall
- Waktu pelatihan: Total waktu yang dibutuhkan untuk melatih model
- Waktu inferensi: Rata-rata waktu per gambar untuk membuat prediksi

### Perbandingan Kinerja

**Akurasi:**
Hasil akurasi akan ditampilkan setelah menjalankan eksperimen

**Efisiensi Komputasi:**
Perbandingan waktu pelatihan dan inferensi akan ditampilkan setelah menjalankan eksperimen

**Metrik per Kelas:**
Analisis performa model pada tiap kelas akan ditampilkan setelah menjalankan eksperimen

## Visualisasi Hasil

Beberapa visualisasi yang dihasilkan:
1. Grafik pelatihan (akurasi dan loss)
2. Confusion matrix untuk setiap model
3. Visualisasi prediksi dari model terbaik
4. Perbandingan metrik antar model
5. Visualisasi contoh kesalahan klasifikasi

## Kesimpulan dan Diskusi

Analisis akan berfokus pada trade-off antara akurasi dan efisiensi untuk ketiga arsitektur. Secara umum, ekspektasi awal adalah:

- **VGG16**: Akurasi tinggi dengan penalti pada waktu pelatihan dan inferensi
- **ResNet50**: Keseimbangan antara akurasi dan performa
- **MobileNetV2**: Performa inferensi terbaik dengan kemungkinan akurasi yang lebih rendah

Kesimpulan akhir akan diberikan berdasarkan hasil eksperimen yang menunjukkan model mana yang paling cocok untuk kasus penggunaan dengan prioritas yang berbeda (akurasi tinggi vs. efisiensi). 
# Bypass Utils Image Detection: A Technical Deep-Dive

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/USERNAME/Bypass-Utils-Image-Google-Colab/blob/main/BypassDetectionAI_Image.ipynb) 
*(Catatan: Ganti `USERNAME` dengan username GitHub Anda setelah diunggah)*

## Abstrak

Proyek ini menyediakan implementasi praktis dan analisis mendalam mengenai berbagai teknik manipulasi gambar yang bertujuan untuk menguji dan melewati model deteksi gambar berbasis AI. Disajikan dalam format Google Colab notebook interaktif, repositori ini ditujukan untuk para peneliti, mahasiswa, dan praktisi di bidang *adversarial machine learning*, keamanan siber, dan forensik digital. Fokus utamanya adalah untuk tujuan pendidikan, yakni untuk memahami mekanisme internal model deteksi dan mengeksplorasi kerentanannya melalui perturbasi yang terkontrol.

## Latar Belakang Teknis

Model deteksi AI modern, terutama yang berbasis *Deep Learning*, seringkali dilatih untuk mengidentifikasi "artefak" atau "sidik jari" digital yang ditinggalkan oleh proses pembuatan gambar sintetik (misalnya, oleh model Generative Adversarial Networks atau Diffusion Models). Artefak ini dapat bermanifestasi dalam beberapa domain:

1.  **Domain Spasial (Pixel):** Pola berulang yang tidak wajar, inkonsistensi pencahayaan lokal, atau tekstur yang terlalu halus atau terlalu seragam.
2.  **Domain Frekuensi (Frequency):** Spektrum frekuensi gambar AI-generated seringkali memiliki karakteristik yang berbeda dari gambar fotografi alami. Misalnya, beberapa model menunjukkan atenuasi frekuensi tinggi yang tidak wajar atau puncak frekuensi spesifik yang terkait dengan arsitektur *upsampling* (seperti *transposed convolutions*).
3.  **Domain Statistik:** Distribusi statistik piksel, gradien, atau koefisien DCT (*Discrete Cosine Transform*) dapat menyimpang dari statistik yang ditemukan pada gambar alami.

Tujuan dari teknik-teknik dalam notebook ini adalah untuk "menyamarkan" gambar yang dimanipulasi dengan cara mengintroduksi karakteristik yang menyerupai gambar alami atau dengan menghapus artefak sintetik yang dapat dideteksi.

---

## Rincian Teknis Implementasi Metode

Notebook ini mengimplementasikan serangkaian teknik perturbasi, masing-masing menargetkan aspek yang berbeda dari jejak digital gambar.

### 1. Noise Injection
-   **Rasional:** Kamera digital di dunia nyata secara inheren menghasilkan noise (derau) karena keterbatasan sensor fisik (misalnya, *shot noise*, *read noise*). Gambar AI yang sempurna dan bebas noise bisa menjadi tanda bahaya bagi detektor. Teknik ini bertujuan untuk mensimulasikan noise sensor yang realistis.
-   **Implementasi:**
    -   **Gaussian Noise:** Derau aditif Gaussian disuntikkan ke seluruh gambar. Variansi (`sigma`) dari derau ini tidak statis; ia diskalakan secara adaptif berdasarkan luminans (kecerahan) rata-rata gambar. Area yang lebih gelap pada gambar cenderung menunjukkan noise yang lebih terlihat, dan implementasi ini meniru efek tersebut (`sigma = 0.5 + (1.0 - mean_luma**1.2) * 6.5`).
    -   **Salt-and-Pepper Noise:** Sejumlah kecil piksel secara acak diatur ke nilai minimum (0) atau maksimum (255). Ini mensimulasikan piksel "mati" atau "panas" pada sensor. Probabilitas (`p`) kejadian ini disesuaikan berdasarkan resolusi gambar untuk menjaga subtilitas.
-   **Dampak:** Menambahkan noise berfrekuensi tinggi yang dapat menutupi artefak berfrekuensi tinggi lainnya dan membuat distribusi statistik gambar lebih mirip dengan foto asli.

### 2. Pixel Perturbation
-   **Rasional:** Ini adalah bentuk serangan adversarial gradien yang disederhanakan. Tujuannya adalah untuk sedikit menggeser nilai piksel ke arah yang kemungkinan besar akan membingungkan model, tanpa harus menghitung gradien dari model target secara eksplisit.
-   **Implementasi:**
    -   **Gradient Estimation:** Gradien gambar diestimasi menggunakan operator Sobel pada versi grayscale. Ini memberikan arah perubahan intensitas tercepat di setiap piksel.
    -   **Perturbation Application:** Perturbasi (`epsilon * sign(gradient)`) ditambahkan ke gambar. Ini secara efektif mendorong piksel di sepanjang arah gradien lokal, sedikit menajamkan atau memburamkan tepi.
    -   **Post-processing:**
        -   *Bilateral Filtering:* Filter ini diterapkan untuk menghaluskan perturbasi sambil mempertahankan ketajaman tepi, mencegah noise yang tampak buatan.
        -   *Color Statistics Matching:* Untuk memastikan warna tidak bergeser secara signifikan, mean dan standar deviasi dari setiap channel warna pada gambar yang diperturbasi dicocokkan kembali dengan statistik gambar asli.
-   **Dampak:** Mengubah hubungan mikro-tekstur dan gradien lokal yang mungkin telah dipelajari oleh detektor sebagai ciri khas gambar sintetik.

### 3. Camera Simulation
-   **Rasional:** Mensimulasikan proses optik dan elektronik yang terjadi saat mengambil foto dengan kamera fisik. Ini adalah salah satu metode paling efektif karena mengintroduksi distorsi kompleks yang sulit dibedakan dari foto asli.
-   **Implementasi:**
    -   **Lens Distortion:** Mensimulasikan distorsi barel/pincushion yang disebabkan oleh ketidaksempurnaan lensa. Ini dicapai dengan pemetaan ulang (`cv2.remap`) koordinat piksel berdasarkan model distorsi radial (`1 + k1*r^2 + k2*r^4`).
    -   **Chromatic Aberration:** Mensimulasikan kegagalan lensa untuk memfokuskan semua warna pada titik yang sama. Channel warna Merah dan Biru digeser sedikit ke arah luar dari pusat gambar, meniru efek *color fringing* di tepi kontras tinggi.
    -   **ISO Noise:** Mensimulasikan noise yang bergantung pada sinyal (mirip dengan pengaturan ISO tinggi pada kamera). Variansi noise lebih tinggi di area yang lebih terang.
    -   **Optical Softening & Sharpening:** Kombinasi *Gaussian blur* (untuk mensimulasikan kelembutan optik) dan *Unsharp Masking* (diimplementasikan sebagai `cv2.addWeighted` dari gambar dengan versi buramnya) untuk meniru penajaman dalam kamera.
-   **Dampak:** Mengubah geometri gambar secara halus dan memperkenalkan korelasi antar-channel warna yang kompleks, secara efektif menimpa artefak digital tingkat rendah.

### 4. FFT Smoothing (Low-pass Filtering)
-   **Rasional:** Beberapa model generatif menghasilkan artefak berfrekuensi tinggi yang tajam dan tidak alami. Metode ini bertujuan untuk menekan frekuensi-frekuensi tersebut.
-   **Implementasi:**
    -   Gambar ditransformasikan ke domain frekuensi menggunakan 2D Fast Fourier Transform (FFT).
    -   Sebuah *low-pass filter mask* (dalam kasus ini, *Butterworth-style filter*) dibuat. Masker ini memiliki nilai 1 di pusat (frekuensi rendah) dan secara bertahap turun ke 0 di tepi (frekuensi tinggi). Parameter `cutoff` dan `rolloff` mengontrol seberapa tajam transisi ini.
    -   Spektrum frekuensi gambar dikalikan dengan masker ini, secara efektif menekan komponen frekuensi tinggi.
    -   Gambar direkonstruksi kembali ke domain spasial menggunakan Inverse FFT.
-   **Dampak:** Menghasilkan gambar yang sedikit lebih lembut, menghilangkan detail frekuensi tinggi yang mungkin menjadi indikator utama bagi detektor.

### 5. FFT Matching
-   **Rasional:** Metode ini lebih canggih daripada sekadar smoothing. Ia mencoba untuk membentuk ulang seluruh spektrum daya (power spectrum) gambar agar sesuai dengan distribusi spektral yang umum ditemukan pada gambar alami, yang seringkali mengikuti hukum pangkat (power law, `1/f` noise).
-   **Implementasi:**
    -   Spektrum amplitudo target (`target`) dibuat secara sintetis, yang memiliki daya tinggi pada frekuensi rendah dan menurun secara eksponensial menuju frekuensi tinggi. Parameter `r0`, `alpha`, dan `anisotropy` mengontrol bentuk spektrum target ini.
    -   Amplitudo gambar asli dihitung dari FFT-nya.
    -   Faktor penskalaan dihitung untuk setiap komponen frekuensi untuk mengubah amplitudo asli agar lebih sesuai dengan amplitudo target. Parameter `strength` mengontrol seberapa kuat penyesuaian ini.
    -   Amplitudo yang baru (telah diskalakan) digabungkan kembali dengan fase asli dari gambar, dan gambar direkonstruksi.
-   **Dampak:** Secara fundamental mengubah tekstur dan karakteristik frekuensi gambar agar lebih "alami" secara statistik, yang seringkali sangat efektif melawan detektor berbasis frekuensi.

### 6. Adversarial Patch Blending (Simulated)
-   **Rasional:** Mensimulasikan penambahan "patch" atau area kecil yang dirancang untuk membingungkan detektor. Dalam implementasi non-adversarial ini, patch tersebut adalah noise yang dihaluskan, bukan gradien yang dioptimalkan. Tujuannya adalah untuk memecah homogenitas statistik gambar.
-   **Implementasi:**
    -   Beberapa patch noise acak yang sangat dihaluskan (*smoothed noise*) dibuat.
    -   Setiap patch ditempatkan di lokasi acak pada gambar.
    -   Patch tersebut dibaurkan (*blended*) dengan gambar asli menggunakan masker alpha berbentuk elips yang lembut untuk transisi yang mulus.
-   **Dampak:** Memperkenalkan inkonsistensi lokal yang terkontrol yang dapat mengganggu analisis statistik global oleh detektor.

---

## Panduan Penggunaan

1.  **Kloning Repositori:**
    ```bash
    git clone https://github.com/USERNAME/Bypass-Utils-Image-Google-Colab.git
    ```
2.  **Buka di Google Colab:** Unggah file `BypassDetectionAI_Image.ipynb` ke Google Colab, atau cukup klik lencana "Open in Colab" di bagian atas README ini.
3.  **Jalankan Sel Setup:** Sel kode pertama (`@title Upload & Setup`) akan menginstal semua dependensi yang diperlukan dan menyiapkan fungsi-fungsi utilitas.
4.  **Unggah Gambar:** Jalankan sel tersebut dan gunakan antarmuka unggah untuk memilih gambar (AI-generated atau lainnya) yang ingin Anda proses.
5.  **Eksekusi Sel Metode:** Jalankan setiap sel metode secara berurutan untuk menerapkan berbagai teknik perturbasi pada gambar Anda. Setiap sel akan menampilkan perbandingan berdampingan antara gambar asli dan yang diproses, bersama dengan metrik kualitas (PSNR dan SSIM).
6.  **Analisis & Ekspor:**
    -   Sel visualisasi gabungan akan menampilkan semua hasil dalam satu kisi untuk perbandingan yang mudah.
    -   Sel ekspor (`@title Export & Download`) akan menyimpan semua gambar yang dihasilkan sebagai file PNG, membuat pipeline gabungan dari semua metode, dan mengemas semuanya ke dalam file ZIP untuk diunduh. Metadata yang berisi parameter yang digunakan akan disematkan ke dalam file PNG jika memungkinkan.
    -   Sel visualisasi teknis terakhir menyediakan analisis mendalam (FFT, PSD, histogram, DCT) untuk memahami dampak transformasi pada level yang lebih dalam.

## Disclaimer Penting

Proyek ini dibuat murni untuk tujuan pendidikan dan penelitian. Tujuannya adalah untuk memajukan pemahaman tentang kekuatan dan kelemahan model AI, dan untuk mendorong pengembangan detektor yang lebih kuat dan tangguh. Pengguna dilarang keras menggunakan kode atau teknik yang disediakan untuk aktivitas ilegal, tidak etis, atau berbahaya, termasuk pembuatan misinformasi atau konten menipu. Penulis tidak bertanggung jawab atas penyalahgunaan materi ini.

## Lisensi

Proyek ini dilisensikan di bawah [Lisensi MIT](LICENSE).

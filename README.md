# Laporan Proyek Machine Learning - Firda Aulia Rakhmah

## Project Overview

Topik yang saya pilih untuk proyek akhir ini adalah mengenai rekomendasi buku, dengan judul proyek **Book Recommendation System**.

<img width="773" alt="Book" src="https://github.com/user-attachments/assets/12764ab5-9872-4bd5-b498-cc93190e7c6f">

Perkembangan teknologi dan internet dalam beberapa tahun terakhir telah mengubah kebiasaan membaca masyarakat Indonesia. Kini, banyak orang lebih tertarik menggunakan media sosial seperti Instagram dan TikTok, atau menikmati hiburan di platform streaming seperti Netflix dan YouTube. Misalnya, seseorang yang dulu mungkin menghabiskan waktu membaca buku sejarah untuk belajar, sekarang lebih sering menonton video dokumenter di YouTube. Begitu pula dengan novel, yang dulu menjadi pilihan utama untuk hiburan, kini mulai tergantikan oleh serial drama di Netflix. Pergeseran ini menunjukkan bagaimana teknologi telah mengubah cara masyarakat mencari informasi dan menikmati hiburan. Membaca buku adalah salah satu cara efektif untuk memperoleh ilmu pengetahuan dan memperluas wawasan. Sebagai sumber informasi, buku memainkan peran penting dalam meningkatkan pemahaman tentang berbagai topik. Namun, di Indonesia, minat baca masyarakat tergolong rendah meskipun informasi tentang buku semakin mudah diakses melalui internet. Salah satu alasan utamanya adalah banyaknya pilihan buku yang tersedia, sehingga pembaca sering kesulitan menemukan bacaan yang sesuai dengan minat atau kebutuhannya. Faktor lain yang memengaruhi rendahnya minat baca adalah kurang menariknya kualitas banyak buku lokal dan keterbatasan akses ke buku berkualitas. Banyak buku yang tidak relevan dengan minat pembaca atau kurang menggugah rasa ingin tahu. Dalam beberapa kasus, masyarakat cenderung lebih tertarik pada buku terjemahan yang menawarkan gaya penulisan dan konten yang lebih menarik [[1](https://tirto.id/6-alasan-mengapa-minat-baca-masyarakat-indonesia-masih-rendah-gCNE)]. 

Berdasarkan data dari Picodi.com, permintaan buku di toko online mengalami peningkatan terbesar pada bulan Desember, dengan kontribusi sebesar 12% dari total transaksi tahunan. Sebaliknya, penurunan paling tajam terjadi pada bulan Juni, hanya mencapai 6% dari total transaksi tahunan. Dalam hal cara mendapatkan buku, survei menunjukkan bahwa 47% responden membeli buku di toko buku fisik, sementara 37% meminjam dari perpustakaan, dan 12% meminjam dari teman. Selain itu, 10% responden mengaku jarang membaca atau tidak tertarik pada buku. Meski teknologi semakin berkembang, mayoritas responden (55%) memilih memesan buku secara online, meskipun 73% tetap menggemari pembelian di toko buku konvensional [[2](https://www.picodi.com/id/mencari-penawaran/pembelian-buku-di-indonesia-dan-di-seluruh-dunia)]. Wattpad adalah salah satu platform media sosial populer yang memungkinkan pengguna untuk membaca dan menulis cerita, sekaligus membangun komunitas penggemar literasi. Platform ini mempermudah interaksi antara pembaca dan penulis, menciptakan ruang yang inklusif untuk berbagi karya. Menurut pandangan Restu I. Aji di Quora.com, beberapa cerita di Wattpad memiliki daya tarik yang lebih kuat dibandingkan dengan buku cetak tradisional. Bahkan, banyak cerita yang awalnya populer di Wattpad akhirnya diterbitkan dalam bentuk buku fisik dan dipasarkan dengan tambahan klaim seperti "dibaca ratusan ribu hingga jutaan kali di Wattpad." Fenomena ini menunjukkan bagaimana Wattpad berhasil menjadi jembatan antara karya digital dan pasar buku konvensional. [[3](https://id.quora.com/Apakah-kamu-lebih-suka-baca-buku-di-situs-Wattpad-atau-toko-buku-offline)]. 

Salah satu cara untuk meningkatkan minat baca di masyarakat adalah dengan mengembangkan sistem rekomendasi buku yang dapat membantu pembaca menemukan buku sesuai preferensi mereka. Sistem ini dapat dirancang menggunakan metode seperti Content-Based Filtering, yang menyarankan buku berdasarkan kesamaan konten dengan pilihan pembaca sebelumnya, serta Collaborative Filtering, yang merekomendasikan buku berdasarkan preferensi pengguna lain dengan minat serupa. Dengan menyediakan rekomendasi yang relevan dan personal, sistem ini diharapkan mampu memberikan pengalaman yang lebih memuaskan bagi pembaca, sekaligus memotivasi mereka untuk terus membaca dan mengeksplorasi lebih banyak buku yang menarik. Hal ini berpotensi meningkatkan minat baca secara keseluruhan di masyarakat.

## Business Understanding

### Problem Statements
Berdasarkan latar belakang yang telah dijelaskan di atas, berikut ini merupakan permasalahan yang akan diselesaikan dari proyek ini : 
- Bagaimana merancang proses pengolahan data meliputi data buku, pengguna, dan penilaian agar data tersebut siap digunakan untuk membangun sistem rekomendasi berbasis machine learning ? 
- Bagaimana cara membuat model machine learning untuk sistem rekomendasi buku ?

### Goals
Berdasarkan permasalahan yang telah dirumuskan sebelumnya, tujuan dari proyek ini adalah : 
- Melakukan proses pengolahan data sehingga data dapat diproses dan siap digunakan dalam pengembangan model machine learning untuk sistem rekomendasi.  
- Merancang dan membangun model machine learning yang mampu memberikan rekomendasi buku terbaik sesuai dengan kebutuhan dan preferensi pengguna.  

### Solution statements
Untuk mencapai tujuan yang telah diuraikan di atas, maka berikut adalah beberapa solusi yang dapat dilakukan agar dapat mencapai tujuan dari proyek ini, yaitu:
- **Content-based Filtering Recommendation**

  Sistem rekomendasi yang berbasis konten (content-based filtering) merupakan sistem rekomendasi yang memberikan rekomendasi item yang hampir sama dengan item yang disukai oleh pengguna di masa lalu. Content-based filtering akan mempelajari profil minat pengguna baru berdasarkan data dari objek yang telah dinilai oleh pengguna lain sebelumnya. Pada pendekatan menggunakan content-based filtering akan menggunakan algoritma TF-IDF Vectorizer dan Cosine Similarity.
  - TF-IDF Vectorizer
    
    Algoritma Term Frequency Inverse Document Frequency Vectorizer (TF-IDF Vectorizer) adalah algoritma yang dapat melakukan kalkulasi dan transformasi dari teks mentah menjadi representasi angka yang memiliki makna tertentu dalam bentuk matriks serta dapat digunakan dan dimengerti oleh model machine learning [[4](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)].

    Kelebihan dari teknik ini adalah tidak membutuhkan data yang diperoleh dari pengguna lain karena rekomendasi yang akan diberikan akan spesifik hanya untuk pengguna tersebut. Sedangkan kekurangan dengan menggunakan teknik ini ialah hasil rekomendasi yang hanya terbatas dari pengguna itu saja dan tidak dapat memperluas data dari penilaian pengguna lain [[5](https://towardsdatascience.com/tf-idf-simplified-aba19d5f5530)].
  - Cosine Similarity

    Teknik cosine similarity digunakan untuk melakukan perhitungan derajat kesamaan (similarity degree) antara dua sampel [[6](https://www.sciencedirect.com/topics/computer-science/cosine-similarity)].
    
- **Collaborative Filtering Recommendation**

  Sistem rekomendasi berbasis **Collaborative Filtering** berfungsi untuk memberikan saran item yang sesuai dengan preferensi pengguna di masa lalu, dengan mengandalkan data dari pengguna lain yang memiliki preferensi serupa, misalnya berdasarkan rating atau penilaian yang diberikan sebelumnya [[7](https://realpython.com/build-recommendation-engine-collaborative-filtering)]. Namun, pendekatan ini memiliki kekurangan, yakni tidak dapat merekomendasikan item yang belum pernah dinilai atau memiliki riwayat transaksi.

  Untuk menerapkan metode ini, diperlukan proses penyandian (encoding) terhadap fitur-fitur dalam dataset, mengubahnya menjadi indeks integer, dan kemudian memetakan informasi tersebut ke dalam dataframe yang sesuai. Setelah itu, dataset akan dibagi menggunakan rasio tertentu untuk memisahkan data yang digunakan untuk pelatihan (training data) dan data yang digunakan untuk pengujian (validation data), sebelum melanjutkan ke tahap pemodelan.

## Data Understanding
Data yang digunakan dalam proyek ini adalah *dataset* yang diambil dari Kaggle Dataset. Di bawah ini adalah informasi detail tentang *dataset* yang digunakan.

|                         | Keterangan                                                                                                                                                                         |
| ----------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Sumber                  | [Kaggle Dataset : Book Recommendation Dataset](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset 'Build state-of-the-art models for book recommendation system') |
| *Usability*             | 10.00                                                                                                                                                                              |
| Lisensi                 | [CC0 : Public Domain](https://creativecommons.org/publicdomain/zero/1.0 'Creative Common - CC0 1.0 Universal')                                                                      |
| Penilaian/*Rating*      | Silver                                                                                                                                                                             |
| Jenis dan Ukuran Berkas | zip (25 MB)                                                                                                                                                                        |
| Tags              | Online Communities, Literature, Art, Recommender Systems, Culture and Humanities                                                                                                                                            |

Dalam dataset tersebut berisi tiga 3 data CSV yaitu `Books.csv`, `Ratings.csv`, `Users.csv`.

- **Books.csv**, memiliki atribut sebagai berikut :
  
  <img width="364" alt="1" src="https://github.com/user-attachments/assets/1c83e39e-3192-4882-87c4-0f683d9d7b61">
  
  **Penjelasan :**
  - `RangeIndex` : Dataset memiliki 271,360 baris, dari indeks 0 hingga 271,359.
  - `Data Columns` : Terdapat 8 kolom dalam dataset.
  - `Non-Null Count` : Menunjukkan jumlah nilai yang tidak kosong dalam setiap kolom.
  - `Dtype` : Menunjukan Tipe data dari setiap kolom.
  - `Memory Usage` : Dataset menggunakan sekitar 16.6 MB memori di RAM.

  Penjelasan kolom : 
  - `ISBN` : Merupakan kode unik *International Standard Book Number* berupa 10 atau 13 digit yang digunakan untuk mengidentifikasi buku secara internasional. Setiap buku memiliki ISBN yang berbeda.
  - `Book-Title` : Berisi judul buku yang dimasukkan dalam dataset. Judul ini digunakan untuk mengidentifikasi isi atau nama buku.
  - `Book-Author` : Nama penulis buku. Bisa berupa satu penulis atau lebih jika buku ditulis oleh beberapa orang.
  - `Year-of-Publication` : Tahun di mana buku diterbitkan untuk pertama kalinya. Informasi ini membantu menentukan usia buku dan relevansinya.
  - `Publisher` : Nama penerbit yang bertanggung jawab atas publikasi buku tersebut. Penerbit biasanya mengelola produksi, distribusi, dan pemasaran buku.
  - `Image-URL-S` : URL untuk gambar sampul buku dengan ukuran kecil. Biasanya digunakan untuk pratinjau cepat atau thumbnail.
  - `Image-URL-M` : URL untuk gambar sampul buku dengan ukuran sedang. Cocok untuk tampilan standar pada aplikasi atau website.
  - `Image-URL-L` : URL untuk gambar sampul buku dengan ukuran besar. Berguna untuk tampilan detail. 
  
- **Ratings.csv**, memiliki atribut sebagai berikut :

  <img width="316" alt="2 0" src="https://github.com/user-attachments/assets/9778f6bd-e844-496c-933c-84d1159c48d5">
  
  **Penjelasan :**
  - `RangeIndex` : Dataset memiliki 1,149,780 baris, dari indeks 0 hingga 1,149,779.
  - `Data Columns` : Terdapat 3 kolom dalam dataset.
  - `Non-Null Count` : Menunjukkan jumlah nilai yang tidak kosong dalam setiap kolom. Semua kolom memiliki 1,149,780 nilai non-null, yang berarti tidak ada nilai yang hilang (NaN).
  - `Dtype` : Menunjukan Tipe data dari setiap kolom.
  -`Memory Usage` : Dataset menggunakan sekitar 26.3 MB memori di RAM.

  Penjelasan kolom : 
  - `User-ID` : Menunjukkan ID unik yang diberikan kepada setiap pengguna yang memberikan rating untuk buku tertentu. Atribut ini digunakan untuk mengidentifikasi setiap pengguna dalam dataset.
  - `ISBN` : Merupakan kode ISBN *(International Standard Book Number)* yang digunakan untuk mengidentifikasi setiap buku secara unik. Setiap ISBN merepresentasikan satu buku yang dapat dinilai oleh pengguna. Atribut ini membantu menghubungkan rating dengan buku yang relevan.
  - `Book-Rating` : Merupakan rating yang diberikan oleh pengguna untuk buku tertentu. Nilai rating bervariasi dari 0 hingga 10, di mana nilai 0 kemungkinan menunjukkan buku yang belum dibaca atau tidak dinilai, sementara nilai yang lebih tinggi mencerminkan tingkat kepuasan pengguna terhadap buku tersebut.

  Deskripsi statistik untuk *dataset* `ratings` pada fitur `Book-Rating` dapat dilihat pada gambar di bawah ini.

  <img width="167" alt="12" src="https://github.com/user-attachments/assets/845cadb9-adb6-400a-bf99-421044aa271c">


  Dari gambar di atas dapat dilihat bahwa terdapat,
  - Total jumlah data (`count`) sebanyak 1.149.780;
  - Rata-rata *rating* (`mean`) 3;
  - Simpangan baku/standar deviasi *rating* (`std`) 4;
  - *Rating* Minimal (`min`), kuartil bawah/Q1 *rating* (`25%`), kuartil tengah/Q2/median *rating* (`50%`) 0;
  - Kuartil atas/Q3 *rating* (`75%`) 7;
  - *Rating* maksimum (`max`) 10

  Berikut adalah visualisasi grafik histogram frekuensi sebaran data *rating* pengguna terhadap buku yang sudah pernah dibaca, mulai dari *rating* terendah yaitu 1 hingga *rating* tertinggi yaitu 10.

  <img width="720" alt="13" src="https://github.com/user-attachments/assets/e0f35fff-236a-4f2b-ae0f-ae458abe8184">


  Berdasarkan hasil visualisasi grafik histogram "Jumlah Rating Buku", dapat disimpulkan bahwa rating yang paling sering diberikan pada buku yang telah dibaca adalah rating 0, dengan jumlah sekitar lebih dari 700.000. Rating 0 ini dapat menimbulkan bias dan mempengaruhi hasil analisis, sehingga data dengan rating 0 sebaiknya dihapus pada tahap persiapan data.
  
- **Users.csv**, memiliki atribut sebagai berikut :
  
  <img width="293" alt="3" src="https://github.com/user-attachments/assets/c85e04a7-d63d-463f-84fa-b623d18ea5f2">
  
  **Penjelasan :**
  - `RangeIndex` : Dataset memiliki 278,858 baris, dari indeks 0 hingga 278,857.
  - `Data Columns` : Terdapat 3 kolom dalam dataset.
  - `Non-Null Count` : Menunjukkan jumlah nilai yang tidak kosong dalam setiap kolom:
  - `Dtype` : Menunjukan Tipe data dari setiap kolom.
  -`Memory Usage` : Dataset menggunakan sekitar 6.4 MB memori di RAM.

  Penjelasan kolom :
  - `User-ID` : Identitas unik pengguna berupa bilangan bulat atau integer
  - `Location` : Lokasi tempat tinggal pengguna
  - `Age` : Umur pengguna

- **Mengubah Nama Kolom**
  
  Perubahan nama kolom bertujuan untuk memudahkan proses pemanggilan dataframe dengan nama kolom yang lebih mudah diingat.
  - Books

    <img width="805" alt="5" src="https://github.com/user-attachments/assets/6def0aa4-cec3-403a-93cc-62529f0e8466">

  - Ratings

    <img width="226" alt="6" src="https://github.com/user-attachments/assets/c265f6c7-0bf6-4406-ba32-3cfdec90ccba">

  - Users

    <img width="263" alt="7" src="https://github.com/user-attachments/assets/40b070a6-b013-4682-a9af-db91ebf9845e">

- **Penggabungan Data ISBN**
  
  Penggabungan data ISBN buku dilakukan dengan menggunakan fungsi `.concatenate` yang disediakan oleh library numpy. Data ISBN ini ada pada dua dataframe, yaitu dataframe buku dan dataframe rating, dan penggabungan dilakukan berdasarkan kolom atau atribut isbn.
  
  <img width="296" alt="a" src="https://github.com/user-attachments/assets/30262250-f59d-496b-92d9-17b6ae2abbb4">

  
- **Penggabungan Data User**

  Penggabungan data user_id pada buku dilakukan dengan menggunakan fungsi `.concatenate` dari library numpy. Data user_id terdapat dalam dua dataframe, yaitu dataframe rating dan dataframe user, dan penggabungan dilakukan berdasarkan kolom atau atribut user_id.

  <img width="281" alt="b" src="https://github.com/user-attachments/assets/157e3738-fdbc-4946-8e48-8187a33befad">

- **Pengecekkan Missing Value**

  Missing value adalah nilai yang hilang atau tidak ada dalam sebuah dataset. Hal ini terjadi ketika data tidak tersedia atau tidak tercatat untuk suatu entri atau atribut tertentu. Missing value sering ditemukan dalam berbagai bentuk, seperti kosong, NaN (Not a Number), atau null, dan bisa muncul karena berbagai alasan, seperti kesalahan pengumpulan data, ketidaksesuaian antara sumber data, atau kelalaian dalam pencatatan. Pengecekan *missing value* pada *dataframe* dapat dilakukan dengan menggunakan fungsi `.isnull().sum()`, yang akan menghasilkan total jumlah data yang kosong atau hilang (*missing*).

  Pada pembuatan Book Recommendation System ini beberapa data ditemuka terdapat missing value yaitu pada **dataframe books**, sehingga data yang missing tersebut harus dihapus. Berikut ini adalah penemuan missing value pada data books :
  
  <img width="217" alt="c" src="https://github.com/user-attachments/assets/80763b64-2e3d-4321-9e1e-82a673bebb29">
  
  Dapat dilihat bahwa pada dataframe books terdapat beberapa atribut yang memiliki nilai kosong atau null, yaitu pada kolom book_author sebanyak 2 data, publisher sebanyak 2 data, dan image_l_url sebanyak 3 data. Oleh karena itu, data yang kosong tersebut dapat dihapus dengan menggunakan fungsi .dropna(). Setelah penghapusan, pengecekan ulang akan menunjukkan bahwa tidak ada lagi data yang kosong atau null. Berikut ini setelah penghapusan missing value :
  
  <img width="228" alt="d" src="https://github.com/user-attachments/assets/e056287e-3002-427c-81ce-3a825234b6ad">

  Pada **dataframe ratings**, tidak ditemukan adanya *missing value*. Dpat dilihat pada gambar dibawah ini :
  
  <img width="274" alt="e" src="https://github.com/user-attachments/assets/c5ed10e9-d145-43dc-97cf-0164aa1921d1">

  Namun, penghapusan nilai rating 0 tetap perlu dilakukan. Hal ini karena berdasarkan hasil analisis pada tahap *data understanding*, rating 0 merupakan kategori yang paling banyak muncul, yaitu sebanyak 716.109 data. Kondisi ini berpotensi menyebabkan bias dalam analisis data. Oleh karena itu, kategori rating 0 tidak disertakan, dan hasil visualisasi grafik histogram setelah penghapusan dapat dilihat pada gambar di bawah ini.
  
  <img width="689" alt="8" src="https://github.com/user-attachments/assets/8a3bcb03-eecd-4451-97c5-85fca4ec2912">

  Berdasarkan hasil visualisasi grafik histogram di atas, rating 0 telah di hapus dan distribusi frekuensi data terlihat lebih rapih dan jelas. Terutama pada rating 1 hingga rating 4.

  Missing value selanjutnya ada pada **dataframe users**. Terdapat sebanyak 110.762 missing value pada fitur `age`. Sehingga mengharuskan data tersebut untuk diisi dengan nilai modus dalam data `age` atau usia. Berikut ini adalah missing value pada data `age`.
  
  <img width="135" alt="9" src="https://github.com/user-attachments/assets/48bd1d14-c8e9-4599-a606-86da957640f2">

  Berikut ini adalah hasil visualisasi grafik histogram umur.

  <img width="371" alt="10" src="https://github.com/user-attachments/assets/ba2541d1-49a0-4d46-8b13-94872143d590">

  Dari grafik di atas dapat dilihat bahwa umur pengguna paling banyak berada pada rentang usia 20 hingga 30 tahun.

- **Pengecekkan Duplicate Data**

  Pengecekan data duplikat dilakukan untuk memastikan tidak ada baris atau entri data yang muncul lebih dari sekali, yang dapat memengaruhi hasil analisis. Untuk memeriksa adanya data duplikat atau data yang sama dalam sebuah dataframe, kita dapat menggunakan fungsi `.duplicated().sum()`. Berikut ini adalah hasil pengecekan duplicate pada setiap data yang digunakan.
  
  <img width="232" alt="11" src="https://github.com/user-attachments/assets/17018bc7-3603-4603-94da-2d61d81c231b">

   Berdasarkan gambar di atas, dapat disimpulkan bahwa data telah bersih dari duplikasi. Hal ini menunjukkan bahwa setiap baris data kini bersifat unik, tanpa adanya pengulangan entri. Dengan demikian, data siap digunakan untuk analisis atau pemrosesan lebih lanjut tanpa khawatir akan bias akibat adanya data duplikat.

- **Penggabungan Data Buku dan Rating**

  Proses penggabungan (merge) dilakukan untuk mengintegrasikan data dari dataframe buku dan dataframe rating menjadi satu dataframe yang komprehensif. Dengan langkah ini, informasi yang sebelumnya terpisah dapat digabungkan, sehingga mempermudah analisis atau pemodelan lebih lanjut.

  <img width="809" alt="18" src="https://github.com/user-attachments/assets/8363f9dc-38ad-48b8-8b04-6e494abbfc73">

## Data Preparation
Sebelum masuk ke tahap data preparation, kita harus melalui tahap preprocessing terlebih dahulu. Tahap Preprocessing data adalah langkah awal yang sangat penting dalam analisis data atau pemodelan *machine learning*. Proses ini bertujuan untuk memastikan bahwa data yang digunakan untuk pemodelan berada dalam kondisi yang optimal, bebas dari masalah yang dapat mempengaruhi hasil model. Berikut adalah beberapa tahapan umum dalam **preprocessing data** :

Setelah selesai melakukan tahap preprocessing, selanjutnya bisa melanjutkan ke tahap **data preparation**. Di tahap ini, data sudah dalam kondisi yang siap untuk digunakan dalam pemodelan. **Proses data preparation sebagai berikut** : 

## Modeling and Result
Tahap berikutnya adalah membangun model machine learning yang berfungsi sebagai sistem rekomendasi untuk memberikan rekomendasi buku terbaik kepada pengguna, menggunakan beberapa algoritma sistem rekomendasi.  

Dari hasil analisis data sebelumnya, kita telah mengetahui bahwa jumlah data pada masing-masing dataframe (data buku, rating, dan users) sangat besar, mencapai ratusan ribu hingga jutaan baris. Ukuran data yang besar ini dapat meningkatkan biaya pemrosesan, memerlukan waktu yang lebih lama, serta menghabiskan banyak resource, seperti RAM atau GPU. Oleh karena itu, untuk mempermudah dan mempercepat proses pemodelan, data yang digunakan akan dibatasi, yaitu sebanyak 10.000 baris untuk data buku dan 5.000 baris untuk data rating.

1. **Content-Based Filtering**
   
   Content-based filtering adalah teknik rekomendasi yang menganalisis konten atau fitur dari item yang ada (misalnya, genre, deskripsi, atau id dari buku) untuk memberikan rekomendasi. Pada tahap ini, sistem rekomendasi dikembangkan dengan menggunakan teknik content-based filtering. Teknik ini berfokus pada merekomendasikan item yang memiliki kesamaan dengan item yang sudah disukai atau dipilih oleh pengguna sebelumnya. Dalam hal ini, jika pengguna menyukai buku tertentu, sistem akan mencari buku lain yang memiliki kesamaan dengan buku tersebut berdasarkan konten atau fitur yang ada.

   Dalam sistem ini, representasi fitur penting dari setiap item, seperti deskripsi buku atau kategori, diubah menjadi bentuk numerik menggunakan teknik TF-IDF Vectorizer. Kemudian, tingkat kesamaan antara item dihitung menggunakan Cosine Similarity, yang mengukur seberapa mirip dua item berdasarkan vektornya.
      
   - TF-IDF Vectorizer

     TF-IDF (*Term Frequency-Inverse Document Frequency*) adalah teknik yang digunakan untuk mengubah data menjadi bentuk numerik agar dapat dianalisis oleh sistem. Dengan menggunakan TF-IDF Vectorizer, data diubah menjadi representasi angka dalam bentuk matriks. Dalam kasus ini, matriks yang dihasilkan memiliki ukuran 10.000 baris data buku dan 5.575 kolom yang merepresentasikan kata-kata unik dari penulis atau deskripsi buku.
     
     <img width="770" alt="14" src="https://github.com/user-attachments/assets/d7b04685-df4a-41d7-ae89-fd2bf9c19284">

   - Cosine Similarity
     
     Cosine Similarity digunakan untuk mengukur tingkat kesamaan antara dua vektor, dalam hal ini adalah representasi buku yang dihasilkan dari TF-IDF. Nilainya berkisar antara -1 hingga 1, di mana:
     - 1 menunjukkan kesamaan penuh
     - 0 menunjukkan tidak ada kesamaan
     - -1 menunjukkan perbedaan penuh.
     
     Metode ini sangat berguna dalam Content-Based Filtering karena membantu menghitung kemiripan antara buku yang berbeda berdasarkan atribut seperti deskripsi atau genre. Cosine Similarity akan melakukan perhitungan derajat kesamaan (similarity degree) antar judul buku. Ukuran matriks yang diperoleh adalah sebesar 10.000 data buku dan 10.000 data buku juga.
     
     <img width="767" alt="15" src="https://github.com/user-attachments/assets/3c6d9085-2b0f-47a8-82d8-01f370f3d579">

     
   - Top-N Recommendation

     Top-N Recommendation adalah langkah akhir dalam sistem rekomendasi di mana algoritma memilih sejumlah buku (N) dengan nilai kesamaan tertinggi terhadap buku yang sedang dicari atau yang sudah dinikmati oleh pengguna. Buku-buku ini disusun berdasarkan skor kesamaan, sehingga sistem dapat memberikan rekomendasi yang paling relevan dan menarik bagi pengguna. Hasil pengujian sistem rekomendasi dengan pendekatan content-based recommendation adalah sebagai berikut.

     <img width="773" alt="16" src="https://github.com/user-attachments/assets/4c53f51b-55fc-45fa-a546-f93e705d9069">

     Gambar diatas menunjukan data berdasarkan judul buku yang dipilih oleh pengguna.
  
     <img width="290" alt="17" src="https://github.com/user-attachments/assets/870679e3-26a7-4c9f-9e40-2c130afbee43">


     Sistem yang telah dibangun berhasil memberikan rekomendasi beberapa judul buku berdasarkan input judul buku "Proxies", dan menghasilkan daftar buku yang relevan berdasarkan perhitungan yang dilakukan oleh sistem.
     
  Kelebihan dan kekurangan Content-based Filtering : 
       
       Kelebihan :
       1. Rekomendasi sangat dipersonalisasi berdasarkan preferensi pengguna sebelumnya, yang memastikan relevansi yang lebih tinggi.
       2. Tidak bergantung pada perilaku pengguna lain, sehingga bisa memberikan rekomendasi meskipun data pengguna terbatas.
       3. Mudah diimplementasi.
      
       Kekurangan :
       1. Hanya bisa merekomendasikan item yang mirip dengan yang sudah disukai pengguna, sehingga tidak bisa memberikan rekomendasi yang lebih beragam atau eksploratif.
       2. Sistem cenderung merekomendasikan item yang serupa dengan yang telah ada, yang bisa membatasi variasi rekomendasi dan tidak membantu pengguna menemukan item yang berbeda atau baru.
       3. Jika item baru tidak memiliki cukup data atau deskripsi, sistem kesulitan memberikan rekomendasi yang relevan.
     
2. **Collaborative Filtering**

   Collaborative Filtering adalah teknik rekomendasi yang memberikan saran item kepada pengguna berdasarkan preferensi pengguna lain yang memiliki kesamaan pola atau perilaku dengan pengguna tersebut. Teknik ini biasanya menggunakan data seperti rating yang diberikan oleh pengguna terhadap item (misalnya buku atau film) untuk mengidentifikasi pola atau kesamaan dengan pengguna lainnya. Kemudian, item yang disukai oleh pengguna yang memiliki kesamaan preferensi akan direkomendasikan kepada pengguna yang belum memilih atau memberi rating pada item tersebut. Pada tahap pembuatan model akan menggunakan kelas `RecommenderNet` dengan `keras model class`
   - Data Preparation
     
     Data preparation dilakukan dengan mengubah fitur `user_id` dan `isbn` pada dataframe ratings menjadi angka indeks (encoding) dalam bentuk integer. Setelah itu, fitur yang telah diubah ini dipetakan kembali ke dalam dataframe ratings. Dari hasil ini, ditemukan bahwa ada 1204 pengguna, 4565 buku, dengan rating terendah sebesar 1 dan rating tertinggi sebesar 10.
     
     <img width="365" alt="19" src="https://github.com/user-attachments/assets/bf9268d3-268f-42da-ac9e-6edda5beb3bb">

   - Spliting Data (Train dan Validation)
     
     Pada tahap ini, dataframe ratings diacak terlebih dahulu, kemudian data dibagi dengan rasio 80:20, di mana 80% digunakan sebagai data latih (training data) dan 20% sisanya digunakan sebagai data uji (validation data).
     
     <img width="277" alt="20" src="https://github.com/user-attachments/assets/2c16c8fd-7d4f-4323-a064-a21b975764b0">

   - Recommendation Testing
     
     Berdasarkan model yang telah dilatih, berikut ini adalah hasil evaluasi dari sistem rekomendasi buku yang menggunakan pendekatan collaborative filtering recommendation.
     
     <img width="652" alt="21" src="https://github.com/user-attachments/assets/bed39d15-9208-4c9a-88de-5c292c387c25">

     Berdasarkan hasil di atas, dapat dilihat bahwa sistem akan mengambil pengguna secara acak, yaitu pengguna dengan `user_id` **278843**. Selanjutnya, sistem akan mencari buku dengan *rating* tertinggi dari pengguna tersebut, yaitu:
     *   **Divine Secrets of the Ya-Ya Sisterhood : A Novel** oleh **Rebecca Wells**
     *   **Icy Sparks** oleh **Gwyn Hyman Rubio**
     *   **The Bonesetter's Daughter** oleh **Amy Tan**
     *   **The Things They Carried** oleh **Tim O'Brien**  

      Kemudian, sistem akan membandingkan antara buku dengan *rating* tertinggi dari pengguna tersebut dan semua buku lainnya yang belum pernah dibaca, lalu mengurutkan buku yang akan direkomendasikan berdasarkan nilai prediksi rekomendasi tertinggi. Terdapat 10 daftar buku yang direkomendasikan oleh sistem, yaitu:
      *   **To Kill a Mockingbird** oleh **Harper Lee**  
      *   **The Secret Life of Bees** oleh **Sue Monk Kidd**  
      *   **The Bean Trees** oleh **Barbara Kingsolver**  
      *   **Life of Pi** oleh **Yann Martel**  
      *   **Chasing the Dime** oleh **Michael Connelly**  
      *   **A Walk in the Woods: Rediscovering America on the Appalachian Trail** oleh **Bill Bryson**  
      *   **The Cabinet of Curiosities** oleh **Douglas Preston**  
      *   **Wuthering Heights** oleh **Emily Bronte**  
      *   **The Visitor (Animorphs, No 2)** oleh **K. A. Applegate**  
      *   **The King of Torts** oleh **John Grisham**  

      Dapat dibandingkan antara daftar ***Book with high ratings from user*** dan ***Top 10 Books Recommendation***, terdapat beberapa kesesuaian pola rekomendasi berdasarkan preferensi pengguna. Hal ini menunjukkan bahwa sistem yang dibangun dapat memberikan rekomendasi buku kepada pengguna dengan hasil yang relevan dan sesuai prediksi.

  Kelebihan dan kekurangan Collaborative Filtering : 
       
       Kelebihan :
       1. Tidak membutuhkan pemahaman mendalam mengenai konten item yang dianalisis.
       2. Dapat Menemukan Rekomendasi Baru.
       3. Bisa memberikan rekomendasi yang lebih beragam dan tidak terbatas pada item serupa dengan yang sudah ada.
      
       Kekurangan :
       1. Sistem kesulitan memberikan rekomendasi untuk pengguna baru (user cold start) atau item baru (item cold start) yang tidak memiliki cukup data interaksi untuk dianalisis.
       2. Ketika jumlah item sangat besar, komputasi untuk mencari kesamaan antar pengguna atau item bisa menjadi sangat berat dan memerlukan waktu serta sumber daya yang besar.
       3. Jika item sedikit sistem bisa kesulitan dalam menghasilkan rekomendasi yang akurat.

       
## Evaluation
1. **Content-Based Filtering**

   Sistem rekomendasi Content-Based Filtering yang dibangun berhasil memberikan rekomendasi buku berdasarkan kemiripan konten antara buku yang telah dibaca pengguna dengan buku lainnya. Teknik Evaluasi yang digunakan untuk content-based filtering adalah dengan menggunakan precission, rumus dari teknik ini adalah :
   
   $$P = \frac{\text{Jumlah rekomendasi yang relevan}}{\text{Jumlah total item yang direkomendasikan}}$$

   **Penjelasan:**
   - **Jumlah rekomendasi yang relevan:** Ini adalah jumlah item yang direkomendasikan oleh sistem yang sesuai dengan kebutuhan atau preferensi pengguna. Artinya, item ini dianggap bermanfaat atau relevan oleh pengguna.
   - **Jumlah total** item yang direkomendasikan: Ini adalah jumlah keseluruhan item yang direkomendasikan oleh sistem, termasuk yang relevan maupun yang tidak relevan.
     
   Precision adalah salah satu metrik evaluasi yang digunakan untuk mengukur keakuratan rekomendasi dalam sistem rekomendasi. Precision menunjukkan proporsi item yang direkomendasikan dan relevan dibandingkan dengan jumlah total item yang direkomendasikan.

   Masih menggunakan data yang sama pada tahap [Modeling](#modeling 'Modeling') *content-based recommendation*, pada proses Hasil *Top-N Recommendation*, yaitu penulis buku atau `book_author` Laura, akan dilakuakn proses pencarian jumlah judul buku atau `book_title` dengan penulis atau *author* yang sama. Pencarian tersebut menggunakan variabel baru yang di mana akan mengambil sebuah data buku yang telah dibaca oleh pengguna. Hasil dari Top-N Recommendation mendapatka beberapa rekomendasi judul buku seperti pada gambar berikut :

   <img width="290" alt="17" src="https://github.com/user-attachments/assets/5656552e-87c9-413c-aad3-fc8624fa2251">
   
3. **Collaborative Filtering**

   Berdasarkan model machine learning yang telah dibangun menggunakan *embedding layer* dengan optimizer Adam dan fungsi loss binary crossentropy, metrik yang digunakan untuk evaluasi adalah Root Mean Squared Error (RMSE). Perhitungan RMSE dapat dilakukan menggunakan rumus berikut :
   
   $$RMSE=\sqrt{\sum^{n}_{i=1} \frac{y_i - y\\_pred_i}{n}}$$
   
   Di mana, nilai $n$ merupakan jumlah *dataset*, nilai $y_i$ adalah nilai sebenarnya, dan $y\\_pred$ yaitu nilai prediksinya terdahap $i$ sebagai urutan data dalam *dataset*.
   
   Nilai RMSE yang rendah menunjukkan bahwa perbedaan antara nilai yang diprediksi oleh model dan nilai observasi yang sebenarnya sangat kecil. Dengan kata lain, semakin kecil nilai RMSE, semakin akurat prediksi model dibandingkan dengan data asli. Berikut ini adalah visualisasi dari hasil training dan validation error menggunakan metrik RMSE, serta grafik yang menunjukkan training dan validation loss selama proses pelatihan.

   <img width="719" alt="22" src="https://github.com/user-attachments/assets/401e5202-2db3-443d-a082-8d834616111d">

   Secara keseluruhan, grafik diatas menunjukkan bahwa model berhasil mempelajari pola dari data dan dapat memberikan hasil yang baik pada data latih maupun data validasi. Penurunan yang stabil pada RMSE dan loss mengindikasikan bahwa model semakin akurat dalam memprediksi hasil.

## Kesimpulan
Kesimpulannya, model yang dibangun untuk merekomendasikan buku menggunakan dua pendekatan, yaitu **Content-based Recommendation** dan **Collaborative Filtering Recommendation**, telah berhasil dikembangkan dan mampu memberikan rekomendasi yang sesuai dengan preferensi pengguna. Pada pendekatan **Collaborative Filtering**, sistem membutuhkan data rating yang diberikan oleh pengguna untuk menentukan kesamaan preferensi antara pengguna yang satu dengan pengguna lainnya, dan berdasarkan informasi ini, rekomendasi dapat diberikan. Sementara itu, pada pendekatan **Content-based Filtering**, data rating tidak diperlukan. Sistem ini mengandalkan analisis terhadap atribut atau konten dari masing-masing buku, seperti genre, deskripsi, dan penulis, untuk memberikan rekomendasi yang relevan berdasarkan buku yang sudah dibaca oleh pengguna.

Kedua pendekatan tersebut memiliki kelebihan dan kekurangannya masing-masing. **Collaborative Filtering** cenderung memberikan rekomendasi yang lebih beragam karena mempertimbangkan pola preferensi pengguna lain, namun dapat menghadapi masalah ketika data rating terbatas atau baru. Di sisi lain, **Content-based Filtering** memberikan rekomendasi yang lebih spesifik berdasarkan atribut buku, namun cenderung membatasi keberagaman karena hanya merekomendasikan buku yang serupa dengan yang sudah dibaca sebelumnya. Meskipun begitu, kedua teknik ini saling melengkapi dan dapat memberikan sistem rekomendasi yang lebih efektif jika digabungkan.

## Referensi
[1]. (https://tirto.id/6-alasan-mengapa-minat-baca-masyarakat-indonesia-masih-rendah-gCNE) Sulthoni. -*6 Alasan Mengapa Minat Baca Masyarakat Indonesia Masih Rendah*. tirto.id. https://tirto.id/6-alasan-mengapa-minat-baca-masyarakat-indonesia-masih-rendah-gCNE

[2]. (https://www.picodi.com/id/mencari-penawaran/pembelian-buku-di-indonesia-dan-di-seluruh-dunia) Picodi. -*Pembelian Buku di Indonesia (dan di seluruh Dunia)*. picodi. https://www.picodi.com/id/mencari-penawaran/pembelian-buku-di-indonesia-dan-di-seluruh-dunia

[3]. (https://id.quora.com/Apakah-kamu-lebih-suka-baca-buku-di-situs-Wattpad-atau-toko-buku-offline) Restu I Aji -*Apakah kamu lebih suka baca buku di situs Wattpad atau toko buku offline?*. Quora. https://id.quora.com/Apakah-kamu-lebih-suka-baca-buku-di-situs-Wattpad-atau-toko-buku-offline

[4]. (https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html) Scikit Learn -*TfidfVectorizer*. Scikit Learn. https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html

[5]. (https://towardsdatascience.com/tf-idf-simplified-aba19d5f5530) Luthfi Ramadhan -*TF-IDF Simplified*. Medium. https://towardsdatascience.com/tf-idf-simplified-aba19d5f5530

[6]. (https://www.sciencedirect.com/topics/computer-science/cosine-similarity) ScienceDirect -*Cosine Similarity*. ScienceDirect. https://www.sciencedirect.com/topics/computer-science/cosine-similarity

[7]. (https://realpython.com/build-recommendation-engine-collaborative-filtering)  Abhinav Ajitsaria -*Build a Recommendation Engine With Collaborative Filtering*. Real Python. https://realpython.com/build-recommendation-engine-collaborative-filtering

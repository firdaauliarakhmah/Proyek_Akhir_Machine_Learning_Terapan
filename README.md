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
| Sumber                  | [Kaggle Dataset: Book Recommendation Dataset](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset 'Build state-of-the-art models for book recommendation system') |
| *Usability*             | 10.00                                                                                                                                                                              |
| Lisensi                 | [CC0: Public Domain](https://creativecommons.org/publicdomain/zero/1.0 'Creative Common - CC0 1.0 Universal')                                                                      |
| Penilaian/*Rating*      | Silver                                                                                                                                                                             |
| Jenis dan Ukuran Berkas | zip (25 MB)                                                                                                                                                                        |
| Tags              | Online Communities, Literature, Art, Recommender Systems, Culture and Humanities                                                                                                                                            |

Dalam dataset tersebut berisi tiga (3) data CSV yaitu `Books.csv`, `Ratings.csv`, `Users.csv`.

- **Books.csv**, memiliki atribut atau fitur sebagai berikut,
  
  <img width="364" alt="1" src="https://github.com/user-attachments/assets/1c83e39e-3192-4882-87c4-0f683d9d7b61">

  - `ISBN` : Merupakan kode unik *International Standard Book Number* berupa 10 atau 13 digit yang digunakan untuk mengidentifikasi buku secara internasional. Setiap buku memiliki ISBN yang berbeda.
  - `Book-Title` : Berisi judul buku yang dimasukkan dalam dataset. Judul ini digunakan untuk mengidentifikasi isi atau nama buku.
  - `Book-Author` : Nama penulis buku. Bisa berupa satu penulis atau lebih jika buku ditulis oleh beberapa orang.
  - `Year-of-Publication` : Tahun di mana buku diterbitkan untuk pertama kalinya. Informasi ini membantu menentukan usia buku dan relevansinya.
  - `Publisher` : Nama penerbit yang bertanggung jawab atas publikasi buku tersebut. Penerbit biasanya mengelola produksi, distribusi, dan pemasaran buku.
  - `Image-URL-S` : URL untuk gambar sampul buku dengan ukuran kecil. Biasanya digunakan untuk pratinjau cepat atau thumbnail.
  - `Image-URL-M` : URL untuk gambar sampul buku dengan ukuran sedang. Cocok untuk tampilan standar pada aplikasi atau website.
  - `Image-URL-L` : URL untuk gambar sampul buku dengan ukuran besar. Berguna untuk tampilan detail.
  
- **Ratings.csv**, memiliki atribut atau fitur sebagai berikut,

  <img width="316" alt="2 0" src="https://github.com/user-attachments/assets/9778f6bd-e844-496c-933c-84d1159c48d5">
  
  - `User-ID` : Menunjukkan ID unik yang diberikan kepada setiap pengguna yang memberikan rating untuk buku tertentu. Atribut ini digunakan untuk mengidentifikasi setiap pengguna dalam dataset.
  - `ISBN` : Merupakan kode ISBN *(International Standard Book Number)* yang digunakan untuk mengidentifikasi setiap buku secara unik. Setiap ISBN merepresentasikan satu buku yang dapat dinilai oleh pengguna. Atribut ini membantu menghubungkan rating dengan buku yang relevan.
  - `Book-Rating` : Merupakan rating yang diberikan oleh pengguna untuk buku tertentu. Nilai rating bervariasi dari 0 hingga 10, di mana nilai 0 kemungkinan menunjukkan buku yang belum dibaca atau tidak dinilai, sementara nilai yang lebih tinggi mencerminkan tingkat kepuasan pengguna terhadap buku tersebut.
  
- **Users.csv**, memiliki atribut atau fitur sebagai berikut,
  
  <img width="293" alt="3" src="https://github.com/user-attachments/assets/c85e04a7-d63d-463f-84fa-b623d18ea5f2">
  
  - `User-ID` : Identitas unik pengguna berupa bilangan bulat atau integer
  - `Location` : Lokasi tempat tinggal pengguna
  - `Age` : Umur pengguna

Deskripsi statistik untuk *dataset* `ratings` pada fitur `Book-Rating` dapat dilihat pada gambar di bawah ini.

<img width="209" alt="4" src="https://github.com/user-attachments/assets/c49bba5e-ada5-4323-a35a-83a898b890ad">

Dari gambar di atas dapat dilihat bahwa terdapat,
- Total jumlah data (`count`) sebanyak 1.149.780;
- Rata-rata *rating* (`mean`) 3;
- Simpangan baku/standar deviasi *rating* (`std`) 4;
- *Rating* Minimal (`min`), kuartil bawah/Q1 *rating* (`25%`), kuartil tengah/Q2/median *rating* (`50%`) 0;
- Kuartil atas/Q3 *rating* (`75%`) 7;
- *Rating* maksimum (`max`) 10

Berikut adalah visualisasi grafik histogram frekuensi sebaran data *rating* pengguna terhadap buku yang sudah pernah dibaca, mulai dari *rating* terendah yaitu 1 hingga *rating* tertinggi yaitu 10.

<img width="431" alt="grafik2" src="https://github.com/user-attachments/assets/e2f88e5e-e640-4328-bb54-0a74e744e149">

Berdasarkan hasil visualisasi grafik histogram "Jumlah Rating Buku", dapat disimpulkan bahwa rating yang paling sering diberikan pada buku yang telah dibaca adalah rating 0, dengan jumlah sekitar lebih dari 700.000. Rating 0 ini dapat menimbulkan bias dan mempengaruhi hasil analisis, sehingga data dengan rating 0 sebaiknya dihapus pada tahap persiapan data.

## Data Preparation
Sebelum masuk ke tahap data preparation, kita harus melalui tahap preprocessing terlebih dahulu. Tahap Preprocessing data adalah langkah awal yang sangat penting dalam analisis data atau pemodelan *machine learning*. Proses ini bertujuan untuk memastikan bahwa data yang digunakan untuk pemodelan berada dalam kondisi yang optimal, bebas dari masalah yang dapat mempengaruhi hasil model. Berikut adalah beberapa tahapan umum dalam preprocessing data :
- **Mengubah Nama Kolom/Atribut/Fitur**
  
  Perubahan nama kolom bertujuan untuk memudahkan proses pemanggilan dataframe dengan nama kolom yang lebih mudah diingat.
  - Books
  - Ratings
  - Users
- **Penggabungan Data ISBN**
  
  Penggabungan data ISBN buku dilakukan dengan menggunakan fungsi .concatenate yang disediakan oleh library numpy. Data ISBN ini ada pada dua dataframe, yaitu dataframe buku dan dataframe rating, dan penggabungan dilakukan berdasarkan kolom atau atribut isbn.
  
- **Penggabungan Data User**

  Penggabungan data user_id pada buku dilakukan dengan menggunakan fungsi `.concatenate` dari library numpy. Data user_id terdapat dalam dua dataframe, yaitu dataframe rating dan dataframe user, dan penggabungan dilakukan berdasarkan kolom atau atribut user_id.

Setelah selesai melakukan tahap preprocessing, selanjutnya bisa melanjutkan ke tahap data preparation. Di tahap ini, data sudah dalam kondisi yang siap untuk digunakan dalam pemodelan. Proses data preparation sebagai berikut : 

## Modeling
-
   
## Evaluation
-

## Referensi
[1]. (https://tirto.id/6-alasan-mengapa-minat-baca-masyarakat-indonesia-masih-rendah-gCNE) Sulthoni. -*6 Alasan Mengapa Minat Baca Masyarakat Indonesia Masih Rendah*. tirto.id. https://tirto.id/6-alasan-mengapa-minat-baca-masyarakat-indonesia-masih-rendah-gCNE

[2]. (https://www.picodi.com/id/mencari-penawaran/pembelian-buku-di-indonesia-dan-di-seluruh-dunia) Picodi. -*Pembelian Buku di Indonesia (dan di seluruh Dunia)*. picodi. https://www.picodi.com/id/mencari-penawaran/pembelian-buku-di-indonesia-dan-di-seluruh-dunia

[3]. (https://id.quora.com/Apakah-kamu-lebih-suka-baca-buku-di-situs-Wattpad-atau-toko-buku-offline) Restu I Aji -*Apakah kamu lebih suka baca buku di situs Wattpad atau toko buku offline?*. Quora. https://id.quora.com/Apakah-kamu-lebih-suka-baca-buku-di-situs-Wattpad-atau-toko-buku-offline

[4]. (https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html) Scikit Learn -*TfidfVectorizer*. Scikit Learn. https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html

[5]. (https://towardsdatascience.com/tf-idf-simplified-aba19d5f5530) Luthfi Ramadhan -*TF-IDF Simplified*. Medium. https://towardsdatascience.com/tf-idf-simplified-aba19d5f5530

[6]. (https://www.sciencedirect.com/topics/computer-science/cosine-similarity) ScienceDirect -*Cosine Similarity*. ScienceDirect. https://www.sciencedirect.com/topics/computer-science/cosine-similarity

[7]. (https://realpython.com/build-recommendation-engine-collaborative-filtering)  Abhinav Ajitsaria -*Build a Recommendation Engine With Collaborative Filtering*. Real Python. https://realpython.com/build-recommendation-engine-collaborative-filtering

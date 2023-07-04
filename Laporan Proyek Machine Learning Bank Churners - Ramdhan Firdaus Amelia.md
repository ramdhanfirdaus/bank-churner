# Laporan Proyek Machine Learning *Bank Churners* - Ramdhan Firdaus Amelia

## Domain Proyek

**Analisis *Churn* (Pergantian Nasabah) dalam Industri Perbankan**

**Latar Belakang**:
Industri perbankan menghadapi tantangan yang signifikan dalam mempertahankan nasabah yang ada [1, 2]. Pergantian nasabah atau *churn* dapat menyebabkan kerugian keuangan dan reputasi bagi lembaga keuangan [2]. Oleh karena itu, penting untuk memahami faktor-faktor yang mempengaruhi *churn* nasabah dan menentukan nasabah yang berkemungkinan mengalami *churn* [1, 2].

## *Business Understanding*

### *Problem Statements*

Menjelaskan pernyataan masalah latar belakang:
- Berdasarkan latar belakang yang telah disajikan, masalah yang ingin diselesaikan adalah bagaimana memprediksi *churn* nasabah dalam industri perbankan untuk mengambil langkah-langkah pencegahan yang diperlukan.
- Masalah lain yang ingin diatasi adalah bagaimana mengidentifikasi faktor-faktor utama yang berkontribusi terhadap *churn* nasabah dalam industri perbankan.

### *Goals*

Menjelaskan tujuan dari pernyataan masalah:
- Tujuan dari pernyataan masalah pertama adalah mengembangkan model prediksi *churn* nasabah yang akurat berdasarkan faktor-faktor yang relevan dalam industri perbankan. Hal ini akan membantu lembaga keuangan mengidentifikasi nasabah yang berpotensi *churn* dan mengambil tindakan yang sesuai untuk mempertahankan mereka.
- Tujuan dari pernyataan masalah kedua adalah mengidentifikasi faktor-faktor kunci yang berpengaruh terhadap *churn* nasabah dalam industri perbankan. Dengan pemahaman yang lebih baik tentang faktor-faktor ini, lembaga keuangan dapat mengambil langkah-langkah pencegahan yang lebih efektif dan mempertahankan nasabah mereka. Langkah-langkah pencegahan yang dapat diambil jika *churn* nasabah terdeteksi dalam industri perbankan adalah sebagai berikut:
    - Memberikan tawaran khusus: Ketika nasabah menunjukkan tanda-tanda akan *churn*, lembaga keuangan dapat memberikan tawaran khusus, seperti diskon suku bunga, pengurangan biaya transaksi, atau program loyalitas, untuk mendorong nasabah agar tetap bertahan.
    - Peningkatan layanan pelanggan: Menyediakan layanan pelanggan yang unggul dapat membantu mengurangi *churn* nasabah. Lembaga keuangan harus merespons keluhan dan permintaan nasabah dengan cepat, menyediakan bantuan yang efektif, dan memastikan pengalaman nasabah yang menyenangkan.
    - Personalisasi komunikasi: Mengirimkan komunikasi yang relevan dan personal kepada nasabah dapat meningkatkan keterlibatan dan mempertahankan nasabah. Lembaga keuangan dapat menggunakan data historis dan preferensi nasabah untuk mengirimkan penawaran atau informasi yang relevan secara tepat waktu.

### Solution statements
- Menggunakan teknik *machine learning* seperti regresi logistik, pohon keputusan, algoritma naive bayes, dan algoritma lainnya untuk membangun model prediksi *churn* nasabah.
- Melakukan analisis fitur untuk menentukan faktor-faktor yang paling penting dalam mempengaruhi *churn* nasabah dan menggunakan informasi ini untuk meningkatkan keputusan bisnis.
- Solusi yang diusulkan akan dievaluasi menggunakan metrik evaluasi seperti akurasi, presisi, recall, dan f1-score untuk memastikan keandalan dan kinerja model prediksi *churn* nasabah.
- Melakukan tuning hyperparameter dengan GridSearchCV pada model terbaik.

## Data Understanding
Dataset yang digunakan adalah dataset Bank Churner yang dapat diakses pada [Bank Churner Dataset](https://www.kaggle.com/datasets/manjuahuja/bank-churner?select=BankChurners.csv). Dataset ini memiliki jumlah baris 10127 baris dan kolom 21 kolom. Pada dataset ini terdapat beberapa *missing values* yang dapat dilihat pada Tabel 3.1 dan *outliers* yang dapat dilihat pada Tabel 3.2. Untuk *duplicate data* pada dataset ini tidak ada.

Tabel 3.1 Daftar *Missing Value* pada Dataset
| Missing values | Total | Percent |                    
| ------ | ------ | ------ |
| Marital_Status | 749 | 0.073961 |
| Income_Category | 1112 | 0.109805 |
| Education_Level | 1519 | 0.149995 |    

Tabel 3.2 Daftar *Outliers* pada Dataset
| Outliers | Jumlah Outliers | Percent |                    
| ------ | ------ | ------ |
| Credit_Limit | 984 | 0.097166 |
| Avg_Open_To_Buy | 963 | 0.095092 |
| Total_Trans_Amt | 896 | 0.062111 |
| Contacts_Count_12_mon | 629 | 0.073961 |
| Total_Amt_Chng_Q4_Q1 | 396 | 0.039103 |
| Total_Ct_Chng_Q4_Q1 | 394 | 0.038906 |
| Months_on_book | 386 | 0.038116 |
| Months_Inactive_12_mon | 331 | 0.032685 |
| Customer_Age | 2 | 0.000197 |
| Total_Trans_Ct | 2 | 0.000197 |

### Variabel-variabel pada dataset adalah sebagai berikut:
Variabel pada dataset Bank Churner ini terdapat 21 variabel. Daftar variabel beserta tipe data dan deskripsinya dapat dilihat pada Tabel 3.3.

Tabel 3.3 Daftar Variabel pada Dataset
| Variabel | Tipe Data | Deskripsi |
| ------ | ------ | ------ |
| CLIENTNUM | Number | Nomor identifikasi unik untuk setiap nasabah dalam dataset. Ini adalah variabel yang digunakan untuk mengidentifikasi individu secara unik. |
| Attrition_Flag | Object | Variabel ini menunjukkan apakah nasabah churned (pergi) atau masih aktif. Nilai "Existing Customer" menunjukkan nasabah yang masih aktif, sementara nilai "Attrited Customer" menunjukkan nasabah yang telah churned. |
| Customer_Age | Number | Variabel ini menunjukkan usia nasabah dalam tahun. |
| Gender | Object | Variabel ini menunjukkan jenis kelamin nasabah. Nilai "M" menunjukkan laki-laki, sementara nilai "F" menunjukkan perempuan. |
| Dependent_count | Number | Variabel ini menunjukkan jumlah orang yang menjadi tanggungan nasabah, seperti pasangan atau anak-anak. |
| Education_Level | Object | Variabel ini menunjukkan tingkat pendidikan nasabah. Nilai-nilai yang mungkin termasuk "Unknown", "Uneducated", "High School", "College", "Graduate", "Post-Graduate", dan "Doctorate". |
| Marital_Status | Object | Variabel ini menunjukkan status perkawinan nasabah. Nilai-nilai yang mungkin termasuk "Unknown", "Single", "Married", "Divorced", dan "Separated". |
| Income_Category | Object | Variabel ini menunjukkan kategori pendapatan nasabah. Nilai-nilai yang mungkin termasuk "Unknown", "Less than  40K"," 40K -  60K"," 60K -  80K"," 80K -  120K",dan" 120K +". |
| Card_Category | Object | Variabel ini menunjukkan jenis kategori kartu kredit yang dimiliki oleh nasabah. Nilai-nilai yang mungkin termasuk "Blue", "Silver", "Gold", dan "Platinum". |
| Months_on_book | Number | Variabel ini menunjukkan jumlah bulan sejak nasabah pertama kali membuka rekening. |
| Total_Relationship_Count | Number | Variabel ini menunjukkan jumlah produk keuangan yang dimiliki oleh nasabah di bank tersebut. |
| Months_Inactive_12_mon | Number | Variabel ini menunjukkan jumlah bulan ketika nasabah tidak aktif dalam 12 bulan terakhir. |
| Contacts_Count_12_mon | Number | Variabel ini menunjukkan jumlah kali nasabah dihubungi oleh bank dalam 12 bulan terakhir. |
| Credit_Limit | Number | Variabel ini menunjukkan batas kredit yang diberikan kepada nasabah. |
| Total_Revolving_Bal | Number |  Variabel ini menunjukkan saldo total dari kredit yang belum dibayar oleh nasabah. |
| Avg_Open_To_Buy | Number | Variabel ini menunjukkan rata-rata jumlah uang yang tersedia untuk nasabah dalam melakukan pembelian dengan kartu kredit. |
| Total_Amt_Chng_Q4_Q1 | Number | Variabel ini menunjukkan persentase perubahan total jumlah transaksi nasabah dari kuartal keempat ke kuartal pertama. |
| Total_Trans_Amt | Number | Variabel ini menunjukkan jumlah total transaksi yang dilakukan oleh nasabah. |
| Total_Trans_Ct | Number | Variabel ini menunjukkan jumlah total transaksi yang dilakukan oleh nasabah. |
| Total_Ct_Chng_Q4_Q1 | Number | Variabel ini menunjukkan persentase perubahan total jumlah transaksi nasabah dari kuartal keempat ke kuartal pertama. |
| Avg_Utilization_Ratio | Number | Variabel ini menunjukkan rasio penggunaan rata-rata dari total kredit yang tersedia kepada nasabah. |

**EDA**:
1. Apa tingkat pendidikan pelanggan dengan rata-rata total transaksi terbesar?
Rata-rata total transaksi pada setiap tingkat pendidikan pelanggan cenderung sama, hanya terdapat perbedaan hanya sekitar 1 nilai. Data ini dapat dilihat pada Gambar 3.1.

![task1-1.jpg](https://i.ibb.co/TvCQFtQ/download-7.png)
Gambar 3.1 Tingkat Pendidikan Pelanggan dengan Rata-rata Total Transaksi Terbesar

2. Bagaimana karakteristik pelanggan yang telah menjadi "Attrited Customer"?
Karakteristik pelanggan yang telah menjadi "Attrited Customer" ini cukup beragam, tergantung sudut pandang variabel yang digunakan. Pada Tabel 3.4, kita dapat melihat bahwa pelanggan yang telah menjadi "Attrited Customer" cenderung berasal dari pelanggan yang memiliki Card_Category Blue. Pada Tabel 3.6, kita dapat melihat bahwa pelanggan yang telah menjadi "Attrited Customer" cenderung berasal dari pelanggan yang memiliki Income_Category yang memiliki income cenderung rendah. Jika berdasarkan Education_Level, yang dapat dilihat pada Tabel 3.5, cenderung pada dua kelas yaitu Graduate dan High School.

    - Card_Category
    Tabel 3.4 Karakteristik Pelanggan "Attrited Customer" berdasarkan Card_Category
    
    | Card_Category | Total Nasabah dengan Kondisi | Total Nasabah | Persentase Nasabah dengan Kondisi |
    | ------ | ------ |  ------ |  ------ |
    | Blue | 1519 | 9436 | 0.160979 |
    | Silver | 82 | 555 | 0.147748 |
    | Gold | 21 | 116 | 0.181034 |
    | Platinum | 5 | 20 | 0.250000 |
    
    - Education_Level
    Tabel 3.5 Karakteristik Pelanggan "Attrited Customer" berdasarkan Education_Level
    
    | Education_Level | Total Nasabah dengan Kondisi | Total Nasabah | Persentase Nasabah dengan Kondisi |
    | ------ | ------ |  ------ |  ------ |
    | Graduate | 487 | 3128 | 0.155691 |
    | High School | 306 | 2013 | 0.152012 |
    | Uneducated | 237 | 1487 | 0.159381 |
    | College | 154 | 1013 | 0.152024 |
    | Doctorate | 95 | 451 | 0.210643 |
    | Post-Graduate | 92 | 516 | 0.178295 |

    - Income_Category
    Tabel 3.6 Karakteristik Pelanggan "Attrited Customer" berdasarkan Income_Category
    
    | Card_Category | Total Nasabah dengan Kondisi | Total Nasabah | Persentase Nasabah dengan Kondisi |
    | ------ | ------ |  ------ |  ------ |
    | Less than $40K | 612 | 3561 | 0.171862 |
    | $40K - $60K | 271 | 1790 | 0.151397 |
    | $80K - $120K | 242 | 1535 | 0.157655 |
    | $60K - $80K | 189 | 1402 | 0.134807 |
    | $120K + | 126 | 727 | 0.173315 |
    
3. Apakah jumlah tanggungan seorang nasabah dapat mempengaruhi limit dari kartu kredit nasabah tersebut?
Jumlah tanggungan seorang nasabah cenderung mempengaruhi limit dari kartu kredit nasabah. Pada Gambar 3.2 dapat dilihat terdapat kenaikan dari jumlah tanggungannya 0 hingga 4. Namun, pada jumlah tanggunan 5 ini terdapat penurunan yang menjelaskan bahwa perlu dilakukan pengecekan pada variabel lain terkait hal ini.

![bar-5.jpg](https://i.ibb.co/02BgJnz/download-21.png)
Gambar 3.2 Limit dari Kartu Kredit berdasarkan Jumlah Tanggungan Seorang Nasabah

## Data Preparation
Pada dataset terdapat beberapa *missing values* yang dapat dilihat pada Tabel 4.1 dan *outliers* yang dapat dilihat pada Tabel 4.2. Untuk *duplicate data* pada dataset ini tidak ada.

**Handle Missing Values**:
Tabel 4.1 Daftar *Missing Value* pada Dataset
| Missing values | Total | Percent |                    
| ------ | ------ | ------ |
| Marital_Status | 749 | 0.073961 |
| Income_Category | 1112 | 0.109805 |
| Education_Level | 1519 | 0.149995 | 
Missing values diputuskan untuk dibuatkan data kategori baru yaitu "Unknown" untuk mendefinisikan kategorikal lain. Data ini tidak dihapus karena ditakutkan terdapat data penting yang terhapus.

**Handle Duplicate Data**:
*Duplicate data* tidak ada.

**Handle Outliers**:
Tabel 4.2 Daftar *Outliers* pada Dataset
| Column | Jumlah Outliers | Persentase Outliers |
| ------ | ------ |  ------ |
| Credit_Limit | 984 | 0.097166 |
| Avg_Open_To_Buy | 963 | 0.095092 |
| Total_Trans_Amt | 896 | 0.088476 |
| Contacts_Count_12_mon | 629 | 0.062111 |
| Total_Amt_Chng_Q4_Q1 | 396 | 0.039103 |
| Total_Ct_Chng_Q4_Q1 | 394 | 0.038906 |
| Months_on_book | 386 | 0.038116 |
| Months_Inactive_12_mon | 331 | 0.032685 |
| Customer_Age | 2 | 0.000197 |
| Total_Trans_Ct | 2 | 0.000197 |

Metode penanganan outliers:
Dalam kasus dataset perbankan ini, penulis telah memutuskan untuk tidak melakukan tindakan apa pun terhadap outliers yang ditemukan. Hal ini dikarenakan penulis menganggap jumlah outliers yang terdeteksi relatif sedikit dan dianggap wajar dalam konteks industri perbankan.

Pertama, penting untuk memahami bahwa tidak semua outliers harus diubah atau dihapus dari dataset. Outliers dapat muncul secara alami dalam data dan mungkin memiliki arti atau informasi yang bernilai. Dalam industri perbankan, variasi ekstrim dalam beberapa variabel seperti credit limit, total transaksi, atau jumlah kontak pelanggan dalam 12 bulan mungkin terjadi karena situasi atau kondisi khusus yang sah.

Selain itu, mengubah atau menghapus outliers dapat memiliki konsekuensi yang signifikan terhadap analisis dan interpretasi data. Tindakan seperti penghapusan outliers dapat menyebabkan pergeseran distribusi data dan mengubah karakteristik asli dari dataset. Ini dapat mempengaruhi analisis statistik, pemodelan, dan kesimpulan yang diambil dari data.

Dalam konteks dataset perbankan, keputusan untuk tidak mengubah outliers dapat dipertimbangkan jika:
1. Outliers yang terdeteksi masih merupakan data yang valid dan mewakili situasi yang mungkin terjadi dalam industri perbankan.
2. Jumlah outliers yang terdeteksi relatif sedikit dibandingkan dengan ukuran total dataset.

**Encoder Data Categorical**:
Data Categorical diputuskan untuk dilakukan LabelEncoder untuk memfasilitasi pemrosesan data oleh algoritma machine learning yang umumnya hanya dapat memanipulasi bilangan numerik.

## Modeling
Pada proses modeling (classfication), penulis melakukan berbagai tahapan untuk menentukan model terbaik. Berikut ini tahapannya:
1. Melakukan feature selection dengan menggunakan sampling, baik oversampling maupun undersampling, untuk menentukan feature terbaik untuk setiap sampling. Pada tahap ini juga digunakan cross validation.
2. Selanjutnya, melakukan prediksi untuk setiap model yang digunakan dengan sampling dan feature terbaiknya. 
3. Memilih dua model terbaik.
4. Dua model terbaik tersebut dilakukan modeling terakhir dengan diperlakukan tuning hyperparameter. Dua model tersebut adalah XGB dengan Random Oversampling dan Random Forest dengan Random Oversampling.
5. Model terbaik adalah XGB dengan Random Oversampling, dengan alasan memiliki Hasil Evaluasi yang lebih baik. Model XGBClassifier ini menggunakan parameter min_child_weight=1, gamma=1.5, colsample_bytree=0.8, max_depth=5.

**Kelebihan dan kekurangan untuk setiap model**
1. Decision Tree
    Kelebihan:
    - Mudah dipahami dan diinterpretasikan.
    - Mampu menangani data numerik dan kategorikal.
    - Dapat digunakan untuk melakukan feature selection.
    - Tidak memerlukan preprocessing data yang rumit.
    
    Kekurangan:
    - Rentan terhadap overfitting jika tidak diatur dengan baik.
    - Tidak efektif dalam menangani masalah dengan banyak fitur berkorelasi.

2. Random Forest
    Kelebihan:
    - Mengatasi masalah overfitting dengan menggunakan ensemble dari decision tree.
    - Mampu menangani data numerik dan kategorikal.
    - Dapat digunakan untuk melakukan feature selection.
    - Menghasilkan estimasi yang stabil dan akurat.
    
    Kekurangan:
    - Lebih kompleks dibandingkan dengan decision tree tunggal.
    - Sulit untuk diinterpretasikan secara langsung karena terdiri dari banyak decision tree.

3. Gaussian
    Kelebihan:
    - Sederhana dan efisien.
    - Mampu menangani data dengan banyak fitur.
    - Relatif tahan terhadap fitur yang tidak relevan.
    
    Kekurangan:
    - Mengasumsikan independensi fitur yang kuat, yang tidak selalu terpenuhi dalam data nyata.
    - Tidak dapat menangani hubungan kompleks antara fitur.

4. KNN
    Kelebihan:
    - Sederhana dan mudah diimplementasikan.
    - Tidak memerlukan asumsi tentang distribusi data.
    - Dapat digunakan untuk masalah klasifikasi dan regresi.
    
    Kekurangan:
    - Sensitif terhadap pemilihan parameter k (jumlah tetangga terdekat).
    - Membutuhkan perhitungan jarak yang mahal secara komputasi jika dataset besar.
    - Tidak efisien dalam menghadapi data dengan dimensi tinggi.

5. Softmax Regression
    Kelebihan:
    - Menghasilkan probabilitas prediksi untuk setiap kelas.
    - Dapat digunakan untuk klasifikasi multikelas.
    - Memiliki interpretasi yang jelas dan dapat diinterpretasikan sebagai model regresi logistik multikelas.
    
    Kekurangan:
    - Tidak efektif untuk data dengan banyak fitur yang berkorelasi.
    - Tidak tahan terhadap nilai yang ekstrim atau pencilan (outliers).

6. Linear SVC
    Kelebihan:
    - Efektif dalam menghadapi data dengan banyak fitur.
    - Dapat digunakan untuk klasifikasi biner dan multikelas.
    - Memiliki toleransi terhadap outliers.
    
    Kekurangan:
    -Memerlukan pemilihan parameter C (penalty parameter) yang optimal.
    - Tidak efektif jika terdapat banyak data noise atau data yang tumpang tindih.

7. SVC
    Kelebihan:
    - Efektif dalam menangani data dengan banyak fitur.
    - Dapat menghasilkan solusi yang optimal dalam ruang fitur yang tinggi.
    - Tahan terhadap overfitting.
    
    Kekurangan:
    - Memerlukan pemilihan parameter kernel yang tepat.
    - Memerlukan waktu komputasi yang lama untuk dataset yang besar.
    - Kurang efisien dalam menangani data yang tidak seimbang.

8. Percepton
    Kelebihan:
    - Sederhana dan cepat dalam pelatihan.
    - Efektif untuk data yang linier terpisah.
    
    Kekurangan:
    - Tidak mampu menangani data yang tidak linier terpisah.
    - Rentan terhadap overfitting jika dataset tidak linier terpisah.

9. SGD
    Kelebihan:
    - Efisien dalam memproses dataset besar.
    - Dapat digunakan untuk klasifikasi dan regresi.
    - Memungkinkan pembaruan parameter secara iteratif.
    
    Kekurangan:
    - Memerlukan tuning parameter yang tepat.
    - Rentan terhadap konvergensi ke titik terjebak jika learning rate tidak sesuai.

10. XGB
    Kelebihan:
    - Menghasilkan model yang sangat akurat.
    - Efektif dalam menangani data dengan banyak fitur.
    - Menggunakan teknik ensemble boosting untuk mengurangi overfitting.
    
    Kekurangan:
    - Memerlukan waktu komputasi yang lama untuk pelatihan dan prediksi.
    - Rentan terhadap overfitting jika parameter tidak diatur dengan baik.

11. Adaboost
    Kelebihan:
    - Efektif dalam meningkatkan kinerja model yang lemah.
    - Mampu menangani data dengan banyak fitur.
    - Dapat digunakan untuk klasifikasi dan regresi.
    
    Kekurangan:
    - Rentan terhadap noise dan data pencilan.
    - Memerlukan waktu komputasi yang lama untuk pelatihan.

12. Gradient Boosting
    Kelebihan:
    - Menghasilkan model yang sangat akurat.
    - Mampu menangani data dengan banyak fitur.
    - Menggunakan teknik ensemble boosting untuk mengurangi overfitting.
    
    Kekurangan:
    - Memerlukan waktu komputasi yang lama untuk pelatihan dan prediksi.
    - Rentan terhadap overfitting jika parameter tidak diatur dengan baik.

Dari kelebihan dan kekurangan tersebut, model XGBoost dapat dikatakan cocock dengan data yang banyak fitur dan dilakukannya tuning hyperparameter.

Setiap algoritma dijalankan dengan penggunaan parameter default dari model-model tersebut, kecuali dua modek yang dipilih, yaitu XGBoost dan Random Forest. XGBoost menggunakan parameter min_child_weight=1, gamma=1.5, colsample_bytree=0.8, max_depth=5, sedangkan Random Forest menggunakan parameter max_depth=6, max_leaf_nodes=9.

Lebih lanjut terkait penerapan tahap feature selection, langkah-langkah yang dilakukan adalah sebagai berikut:
1. **Sampling**: Pada tahap ini, dilakukan sampling untuk mengatasi ketidakseimbangan kelas pada data. Dua teknik yang digunakan adalah oversampling dan undersampling.
    - Oversampling: Meningkatkan jumlah sampel pada kelas minoritas. Teknik oversampling yang digunakan adalah Random Oversampling, SMOTE, Borderline-SMOTE, dan Borderline Oversampling with SVM
    - Undersampling: Mengurangi jumlah sampel pada kelas mayoritas. Teknik undersampling yang digunakan adalah Random Undersampling dan NearMiss.
2. **Feature Selection**: Setelah melakukan sampling, dilakukan seleksi fitur untuk menentukan subset fitur yang paling informatif. Teknik yang digunakan adalah Recursive Feature Elimination, mengeliminasi fitur secara rekursif berdasarkan kontribusinya terhadap model.
3. **Cross Validation**: Pada tahap ini, dilakukan evaluasi model menggunakan metode cross validation. Data dibagi menjadi beberapa subset (fold), dan model dilatih dan dievaluasi pada setiap fold secara bergantian. Tujuan dari cross validation adalah untuk mendapatkan perkiraan kinerja model yang lebih stabil dan mengurangi risiko overfitting atau underfitting.

Setelah melakukan feature selection dan cross validation, dilakukan prediksi menggunakan berbagai model yang telah dipilih. Dari hasil evaluasi, dipilih dua model terbaik, yaitu XGBoost (Extreme Gradient Boosting) dengan Random Oversampling dan Random Forest dengan Random Oversampling.

Alasan memilih model XGBoost dan Random Forest sebagai dua model terbaik adalah karena keduanya memiliki kemampuan yang baik dalam menangani data dengan banyak fitur dan kecenderungan untuk menghasilkan prediksi yang akurat. XGBoost menggunakan teknik ensemble boosting yang menggabungkan beberapa model lemah menjadi satu model yang kuat, sementara Random Forest menggunakan ensemble dari decision tree untuk mengurangi overfitting dan meningkatkan kinerja prediksi.

Kemudian, dua model terbaik tersebut diperlakukan dengan tuning hyperparameter. Hal ini dilakukan untuk mencari kombinasi parameter yang optimal agar model dapat menghasilkan prediksi yang lebih baik. Setelah melalui proses tuning, model XGBoost dengan Random Oversampling dipilih sebagai model terbaik karena memiliki hasil evaluasi yang lebih baik.

**Proses improvement yang dilakukan**
1. Melakukan sampling dengan tujuan untuk memastikan bahwa dataset yang digunakan untuk melatih model klasifikasi memiliki representasi yang seimbang dari semua kelas target yang ada.
2. Melakukan tuning hyperparameter pada dua model terbaik berdasarkan hasil *classification* dengan *sampling*.

## *Evaluation*
Hasil Evaluasi berdasarkan classification report dapat dilihat pada Tabel 6.1. Sedangkan Confusion Matrix dapat dilihat pada Tabel 6.2.

Tabel 6.1 Hasil Evaluasi

              precision    recall  f1-score   support
           0       0.87      0.92      0.90       223
           1       0.98      0.97      0.98      1194

    accuracy                           0.97      1417
    macro avg      0.93      0.95      0.94      1417
    weighted avg   0.97      0.97      0.97      1417

Tabel 6.2 Confusion Matrix

    prediction/actual        0        1
           0                 205      18
           1                 30       1164


*   Accuracy Average: 0.9661256175017643
*   F1 Macro Average: 0.937497243174099
*   F1 Micro Average: 0.9661256175017643
*   Precision Macro Average: 0.928555999567988
*   Precision Micro Average: 0.9661256175017643
*   Recall Macro Average: 0.9470784415350294
*   Recall Micro Average: 0.9661256175017643

### Penjelasan
Berdasarkan hasil evaluasi *classification report*, model terbaik yang dipilih, yaitu XGBoost dengan Random Oversampling, menghasilkan hasil evaluasi yang sangat baik dengan tingkat akurasi sebesar 0.97 atau 97%. Hal ini berarti model berhasil memprediksi dengan benar sekitar 97% dari total kasus yang diamati.

Berikut adalah penjelasan lebih lanjut tentang metrik evaluasi yang digunakan:
- Precision: Menunjukkan seberapa akurat model dalam mengidentifikasi kelas positif (nasabah yang berpotensi churn). Precision yang tinggi (0.98) menunjukkan bahwa sebagian besar prediksi positif yang dilakukan oleh model benar, dengan sedikit false positive.
- Recall: Menunjukkan seberapa baik model dapat menemukan semua kasus positif yang sebenarnya (nasabah yang berpotensi *churn*). Recall yang tinggi (0.97) menunjukkan bahwa model mampu mengidentifikasi sebagian besar kasus positif yang ada.
- F1-score: Merupakan harmonic mean antara precision dan recall. F1-score yang tinggi (0.98) menunjukkan bahwa model memiliki keseimbangan yang baik antara precision dan recall, dengan kemampuan yang baik dalam mengidentifikasi nasabah yang berpotensi *churn*.

*Confusion matrix* juga memberikan gambaran tentang kinerja model**:
- Terdapat 205 prediksi yang benar untuk kelas 0 (nasabah yang tidak berpotensi *churn*) dan 1164 prediksi yang benar untuk kelas 1 (nasabah yang berpotensi *churn*).
- Terdapat 18 false negative (kasus yang sebenarnya *churn* tetapi diprediksi sebagai tidak *churn*) dan 30 false positive (kasus yang sebenarnya tidak *churn* tetapi diprediksi sebagai *churn*).

Metrik evaluasi yang digunakan untuk memilih model terbaik dapat bervariasi tergantung pada kebutuhan bisnis dan preferensi. Dalam kasus ini, beberapa metrik evaluasi yang digunakan adalah:
- Accuracy: Mengukur seberapa akurat model dalam melakukan prediksi secara keseluruhan. Model dengan akurasi yang tinggi akan memberikan hasil yang baik dalam mengklasifikasikan nasabah yang berpotensi *churn* dan tidak *churn*.
- F1-score: Menggabungkan precision dan recall untuk memberikan gambaran keseimbangan antara kedua metrik tersebut. F1-score yang tinggi menunjukkan bahwa model mampu mengidentifikasi dengan baik nasabah yang berpotensi *churn*.

Dengan hasil evaluasi yang baik, model ini dapat membantu dalam mengidentifikasi nasabah yang berpotensi *churn*. Dengan menggunakan fitur-fitur yang relevan, model dapat memprediksi nasabah yang memiliki kecenderungan untuk berhenti menggunakan layanan perbankan. Dengan demikian, bank dapat mengambil tindakan yang tepat untuk mempertahankan nasabah tersebut, seperti memberikan penawaran khusus atau layanan yang lebih baik, sehingga dapat mengurangi tingkat *churn* dan mempertahankan basis nasabah yang lebih baik.

## *Kesimpulan*
- Dengan hasil ini, kita mendapatkan model untuk memprediksi *churn* nasabah dalam industri perbankan dengan model klasifikasi XGB. Model ini cukup baik dengan akurasi sebesar 96.61%.
- Kita juga mendapatkan bahwa faktor-faktor yang berkontribusi terhadap *churn* nasabah dalam industri perbankan adalah:
    - Gender
    - Dependent_count
    - Education_Level
    - Marital_Status
    - Total_Relationship_Count
    - Months_Inactive_12_mon
    - Contacts_Count_12_mon
    - Credit_Limit
    - Total_Revolving_Bal
    - Total_Amt_Chng_Q4_Q1
    - Total_Trans_Amt
    - Total_Trans_Ct
    - Total_Ct_Chng_Q4_Q1
    - Avg_Utilization_Ratio

## *Saran*
- Saran untuk tindakan lebih lanjut:
    Berdasarkan hasil analisis, lembaga keuangan dapat mengambil beberapa tindakan untuk mengurangi *churn* nasabah dan meningkatkan retensi nasabah. Berikut adalah beberapa saran yang dapat dipertimbangkan:
    1. Memberikan penawaran khusus atau insentif kepada nasabah yang berpotensi *churn* untuk mempertahankan mereka, seperti program loyalitas, penawaran suku bunga yang lebih baik, atau diskon pada biaya layanan.
    2. Meningkatkan kualitas layanan dan pengalaman nasabah secara keseluruhan. Hal ini dapat mencakup peningkatan responsivitas, personalisasi layanan, dan kemudahan akses ke produk dan layanan perbankan.
    3. Mengoptimalkan komunikasi dengan nasabah melalui saluran yang berbeda, seperti pemasaran berbasis email, pemberitahuan SMS, atau platform media sosial. Komunikasi yang efektif dapat membantu membangun hubungan yang kuat dengan nasabah.
- Pengembangan lebih lanjut:
    1. Model ini dapat diintegrasikan ke dalam sistem perbankan yang ada untuk memberikan prediksi *churn* secara *real-time* dan memungkinkan lembaga keuangan mengambil tindakan yang cepat.
    2. Lebih lanjut, penting untuk terus memantau performa model yang ada dan melakukan evaluasi berkala terhadap model ini untuk memastikan tetap relevan dan efektif dalam menghadapi perubahan tren dan kebutuhan nasabah.

**Daftar Pustaka**: 
[1] Naufal, M. F., Subrata, S., Susanto, A. F., Kansil, C. N., & Huda, S. (2023). Penerapan Machine Learning untuk Prediksi Potensi Hilangnya Nasabah Bank. Techno. com, 22(1), 1-11.
[2] Rahman, M., & Kumar, V. (2020, November). Machine learning based customer churn prediction in banking. In 2020 4th international conference on electronics, communication and aerospace technology (ICECA) (pp. 1196-1201). IEEE.

**---Ini adalah bagian akhir laporan---**
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 21:39:43 2023

@author: cevas
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle

df = pd.read_csv('Data Mahasiswa Baru.csv')
df.head()
df.shape
df.cov()

#data preprosesing
def kategorikan_ipk(ipk):
    if 3.5 <= ipk <= 4.0:
        return 1
    elif 3.0 <= ipk < 3.5:
        return 2
    elif 2.5 <= ipk < 3.0:
        return 3
    elif 2.0 <= ipk < 2.5:
        return 4

# Menerapkan fungsi kategorikan_ipk ke kolom IPK
df['RankIPK'] = df['IPK'].apply(kategorikan_ipk)

rank1 = df[df['RankIPK']==1]
rank2 = df[df['RankIPK']==2]
rank3 = df[df['RankIPK']==3]
rank4 = df[df['RankIPK']==4]

# Pilih fitur-fitur 
features_df = df[['RankIPK', 'LamaStudi']]
features_df.shape
#Buat array Numpy utk features
features_np = np.array(features_df.values)

# Buat numpy array X dari dataframe features_df
X = np.array(features_df.values)

#Import library k-Means
from sklearn.cluster import KMeans

# Manual kelas k-Means:
#https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html

#Lakukan clustering (fit) terhadap X dgn jumlah cluster = 3
kmeans_model = KMeans(n_clusters=5, random_state=0).fit(X)

# Eksperimen:
# Ubahlah nilai n_clusters = 4, 5, 6, .... amati hasil-hasil di bawah 

# Simpan hasil clustering berupa nomor klaster tiap objek/rekord di
# varialbel klaster_objek
klaster_objek = kmeans_model.labels_

# Simpan hasil clustering berupa centroid (titik pusat) tiap kelompok
# di variabel centroids
centroids = kmeans_model.cluster_centers_

#import library silhoutte score
from sklearn.metrics import silhouette_score

#clustering kmeans dengan pemeriksaan kualitas hasil cluster menggunakan elbow method dan koefisien silhouette
#clustering kmeans dilakukan dengan menjalankan algoritmanya menggunakan nilai k 2 hingga 15
intertia = []
silhouette_coefficients = []
K = range(2,10)
for k in K:
    kmeans_model = KMeans(n_clusters=k, random_state=0).fit(X)
    intertia.append(kmeans_model.inertia_)
    score = silhouette_score(X, kmeans_model.labels_,  metric='euclidean')
    silhouette_coefficients.append(score)

#visualisasi hasil elbow method    
plt.plot(K, intertia, marker= "o")
plt.xlabel('k')
plt.xticks(np.arange(2, 10))
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.show()

#visualisasi hasil perhitungan koefisien Silhouette   
plt.plot(K, silhouette_coefficients, marker= "o")
plt.xlabel('k')
plt.xticks(np.arange(2, 10))
plt.ylabel("Silhouette Coefficient")
plt.title("AVG Silhouette Coefficient")
plt.show()


# Warna yang akan digunakan untuk mewakili setiap klaster pada scatter plot
colors = ['r', 'g', 'b', 'y', 'm']

# Membuat scatter plot
plt.figure(figsize=(8, 6))

for i in range(len(X)):
    plt.scatter(X[i, 0], 
                X[i, 1], 
                color=colors[klaster_objek[i]], 
                marker='o', 
                s=70) #untuk mengatur ukuran marker

# Menambahkan label sumbu dan judul
plt.xlabel('IPK')
plt.ylabel('LamaStudi')
plt.title('Visualisasi Klaster dengan Scatter Plot')

# Menambahkan legenda
for i, color in enumerate(colors):
    plt.scatter([], [], color=color, label=f'Klaster {i}')
    plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1))

# Menampilkan plot
plt.show()

unique_clusters, cluster_counts = np.unique(klaster_objek, return_counts=True)

# Tampilkan hasil
for cluster, count in zip(unique_clusters, cluster_counts):
    print(f"Cluster {cluster}: {count} anggota")

df_new = df[['IPK', 'LamaStudi']]
df_new['Label'] = klaster_objek
df_new['RankIPK'] = df['RankIPK']
df_new.to_csv('Data Hasil Clustering.csv', index=False)

label0 = df_new[df_new['Label']==0]
label1 = df_new[df_new['Label']==1]
label2 = df_new[df_new['Label']==2]
label3 = df_new[df_new['Label']==3]
label4 = df_new[df_new['Label']==4]

label0.describe()
label1.describe()
label2.describe()
label3.describe()
label4.describe()

print('IPK label0: mean=%.3f stdv=%.3f' % (label0['IPK'].mean(), label0['IPK'].std()))
print('IPK label1: mean=%.3f stdv=%.3f' % (label1['IPK'].mean(), label1['IPK'].std()))
print('IPK label2: mean=%.3f stdv=%.3f' % (label2['IPK'].mean(), label2['IPK'].std()))
print('IPK label3: mean=%.3f stdv=%.3f' % (label3['IPK'].mean(), label3['IPK'].std()))
print('IPK label4: mean=%.3f stdv=%.3f' % (label4['IPK'].mean(), label4['IPK'].std()))


avg_lama_studi0 = label0.groupby('IPK')['LamaStudi'].mean().reset_index()

coeffs = np.polyfit(avg_lama_studi0['IPK'], avg_lama_studi0['LamaStudi'], 1)
plt.scatter(avg_lama_studi0['IPK'], avg_lama_studi0['LamaStudi']) # plotting the clusters
# Menambahkan garis regresi
plt.plot(avg_lama_studi0['IPK'], np.polyval(coeffs, avg_lama_studi0['IPK']), color='red', label='Regresi')
plt.title('Pola Antara IPK dengan LamaStudi Klaster 0')
plt.xlabel("IPK") # X-axis label
plt.ylabel("LamaStudi") # Y-axis label
plt.show() 

dataframeLabel = [label0, label1, label2, label3, label4]
labels = ['label0', 'label1', 'label2', 'label3', 'label4']
boxplot_data = [df['IPK'].values for df in dataframeLabel]

plt.figure(figsize=(8, 6))  # Adjust the figure size as per your preference

# Create the box plots for each DataFrame
plt.boxplot(boxplot_data, labels=labels)
plt.xlabel('Label')
plt.ylabel('IPK')
plt.title('Sebaran IPK untuk tiap Label')
plt.grid(True)  # Add grid lines for better readability
plt.show()

boxplot_data2 = [df['LamaStudi'].values for df in dataframeLabel]

plt.figure(figsize=(8, 6))  # Adjust the figure size as per your preference

# Create the box plots for each DataFrame
plt.boxplot(boxplot_data2, labels=labels)
plt.xlabel('Label')
plt.ylabel('LamaStudi')
plt.title('Sebaran LamaStudi untuk tiap Label')
plt.grid(True)  # Add grid lines for better readability
plt.show()


plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.xlabel('IPK')
plt.ylabel('Density')
plt.hist(label0['RankIPK'], bins=30, density=True, color='blue', alpha=0.7)
plt.title('Histogram')

plt.hist(label0['IPK'], bins=10, edgecolor='black')  # You can adjust the number of bins as per your preference
plt.xlabel('IPK')
plt.ylabel('Frequency')
plt.title('Persebaran IPK Cluster 0')
plt.show()





# Model terbaik clustering (berupa centroids) dapat disimpan dan dimanfaatkan
# untuk mencari kelompok dari objek-objek baru 

#Lakukan clustering (fit) terhadap X dgn jumlah cluster = 3
kmeans_model = KMeans(n_clusters=5, random_state=0).fit(X)

#Baca rekord-rekord bunga Irish yg belum dikeathui cluster-nya
df_baru = pd.read_csv('Data Mahasiswa 2.csv', delimiter = ',')
df_baru['RankIPK'] = df_baru['IPK'].apply(kategorikan_ipk)
#Ambil & print rekord bag atas
print(df_baru.head())

# Pilih fitur-fitur 
features_dfBaru = df_baru[['RankIPK', 'LamaStudi']]
features_dfBaru.shape
#Buat array Numpy utk features
features_npBaru = np.array(features_dfBaru.values)
# Buat numpy array X_new
X_new = np.array(features_dfBaru.values)

#Cari (predict) cluster dari rekord-rekord bunga Iris yg baru
kmeans_model.predict(X_new)
klaster_objek2 = kmeans_model.predict(X_new)

df_baru['Label'] = klaster_objek2
df_baru.to_csv('Data Hasil Clustering Predict.csv', index=False)

#Simpan model clustering (agar dapat digunakan lagi lain kali)
# pickle.dump(model, open(filename, 'wb')) #Saving the model
pickle.dump(kmeans_model, open('kmeans_model', 'wb'))

# Baca model dan gunakan kembali untuk memprediksi cluster objek baru
loaded_model = pickle.load(open('kmeans_model', 'rb'))


##Clustering data tambahan
df_extra = pd.read_csv('Data Mahasiswa Tambahan.csv', delimiter=',')
df10_extra = df_extra.head(10)
df10_extra['RankIPK'] = df10_extra['IPK'].apply(kategorikan_ipk)

fitur_extra = df10_extra[['RankIPK', 'LamaStudi']]
X_extra = np.array(fitur_extra.values)
#Cari (predict) cluster dari rekord-rekord bunga Iris yg baru
loaded_model.predict(X_new)
klaster_extra = loaded_model.predict(X_extra)
df10_extra['Label'] = klaster_extra
df10_extra.to_csv('Data Clustering Extra.csv', index=False)



# Jika jumlah atribut banyak dan ingin dicari fitur yg sesuai untuk clustering
# Salah satu cara: membandingkan variance antar atribut
#
df_baru.cov()


# Pencarian jumlah kelompok terbaik
intertia = []
silhouette_coefficients = []
K = range(2,10)
for k in K:
    kmeans_model = KMeans(n_clusters=k, random_state=0).fit(X)
    intertia.append(kmeans_model.inertia_)
    score = silhouette_score(X, kmeans_model.labels_,  metric='euclidean')
    silhouette_coefficients.append(score)

#visualisasi hasil elbow method    
plt.plot(K, intertia, marker= "o")
plt.xlabel('k')
plt.xticks(np.arange(2, 10))
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.show()

#visualisasi hasil perhitungan koefisien Silhouette   
# Pada hasil plot terlihat bahwa koefisien pada k=3 lebih baik dibanding
# menggunakan seluruh (4) atribut
plt.plot(K, silhouette_coefficients, marker= "o")
plt.xlabel('k')
plt.xticks(np.arange(2, 10))
plt.ylabel("Silhouette Coefficient")
plt.title("AVG Silhouette Coefficient")
plt.show()

















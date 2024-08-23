# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 21:49:36 2023

@author: cevas
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle

df = pd.read_csv('Data Mahasiswa.csv')
df2 = pd.read_csv('Data Mahasiswa Tambahan.csv')
df.head()
df.shape
df.cov()

df_all = pd.concat([df, df2], ignore_index=True)

# Pilih fitur-fitur 
features_df = df[['IPK', 'LamaStudi']]
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


# Model terbaik clustering (berupa centroids) dapat disimpan dan dimanfaatkan
# untuk mencari kelompok dari objek-objek baru 

#Lakukan clustering (fit) terhadap X dgn jumlah cluster = 3
kmeans_model = KMeans(n_clusters=5, random_state=0).fit(X)

#Baca rekord-rekord bunga Irish yg belum dikeathui cluster-nya
df_baru = pd.read_csv('Data Mahasiswa 2.csv', delimiter = ',')

#Ambil & print rekord bag atas
print(df_baru.head())

# Pilih fitur-fitur 
features_dfBaru = df_baru[['IPK', 'LamaStudi']]
features_dfBaru.shape
#Buat array Numpy utk features
features_npBaru = np.array(features_dfBaru.values)
# Buat numpy array X_new
X_new = np.array(features_dfBaru.values)

#Cari (predict) cluster dari rekord-rekord bunga Iris yg baru
kmeans_model.predict(X_new)

#Simpan model clustering (agar dapat digunakan lagi lain kali)
# pickle.dump(model, open(filename, 'wb')) #Saving the model
pickle.dump(kmeans_model, open('kmeans_model', 'wb'))

# Baca model dan gunakan kembali untuk memprediksi cluster objek baru
loaded_model = pickle.load(open('kmeans_model', 'rb'))

#Cari (predict) cluster dari rekord-rekord bunga Iris yg baru
loaded_model.predict(X_new)

# Jika jumlah atribut banyak dan ingin dicari fitur yg sesuai untuk clustering
# Salah satu cara: membandingkan variance antar atribut
#
df_baru.cov()

# Terlihat atribut sepal_width memiliki nilai kovariance kecil dibandingkan 
# yang lain, di sini sepal_width akan diabaikan (tidak digunakan)

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





###########################################

### DBSCAN ###

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle

df = pd.read_csv('Data Mahasiswa.csv')
df.head()
df.shape

# Pilih fitur-fitur 
features_df = df[['IPK', 'LamaStudi']]
features_df.shape
#Buat array Numpy utk features
features_np = np.array(features_df.values)

# Buat numpy array X dari dataframe features_df
X = np.array(features_df.values)

#menentukan epsilon
# CARA 1
from sklearn.neighbors import NearestNeighbors # importing the library
neighb = NearestNeighbors(n_neighbors=2) # creating an object of the NearestNeighbors class
nbrs=neighb.fit(features_df) # fitting the data to the object
distances,indices=nbrs.kneighbors(features_df) # finding the nearest neighbours

# Sort and plot the distances results
distances = np.sort(distances, axis = 0) # sorting the distances
distances = distances[:, 1] # taking the second column of the sorted distances
plt.rcParams['figure.figsize'] = (5,3) # setting the figure size
plt.plot(distances) # plotting the distances
plt.ylabel('Epsilon')
plt.xlabel('Jumlah Data')
plt.show() # showing the plot


from sklearn.cluster import DBSCAN
# cluster the data into five clusters
dbscan = DBSCAN(eps = 0.04, min_samples = 4).fit(features_df) # fitting the model
labels = dbscan.labels_ # getting the labels
labels
labels.max()

plt.scatter(features_df.iloc[:,0], features_df.iloc[:,1], c = labels, cmap= "plasma") # plotting the clusters
plt.xlabel("IPK") # X-axis label
plt.ylabel("LamaStudi") # Y-axis label
plt.show() # showing the plot

#CARA 2
# Instantiate DBSCAN
eps = 0.5  # Maximum distance between two samples to be considered as in the same neighborhood
min_samples = 5  # Minimum number of samples in a neighborhood for a point to be a core point
dbscan = DBSCAN(eps=eps, min_samples=min_samples)

# Fit and predict using DBSCAN
labels = dbscan.fit_predict(X)

# Visualize the results
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.title("DBSCAN Clustering")
plt.xlabel("IPK")
plt.ylabel("LamaStudi")

for i, color in enumerate(colors):
    plt.scatter([], [], color=color, label=f'Klaster {i}')
    plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1))

plt.show()













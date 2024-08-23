# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 12:25:52 2023

@author: cevas
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
import seaborn as sns


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

# Pilih fitur-fitur 
features_df = df[['RankIPK', 'LamaStudi']]
features_df.shape
#Buat array Numpy utk features
features_np = np.array(features_df.values)

# Buat numpy array X dari dataframe features_df
X = np.array(features_df.values)

#bikin model agglomerative single linkage
from sklearn.cluster import AgglomerativeClustering
agglom = AgglomerativeClustering(n_clusters = 5, linkage = 'single')
agglom.fit(X)
agglom2 = AgglomerativeClustering(n_clusters = 5, linkage = 'complete')
agglom2.fit(X)

#dapetin label cluster
labels = agglom.labels_
labels2 = agglom2.labels_

from sklearn.metrics import silhouette_score

#clustering kmeans dengan pemeriksaan kualitas hasil cluster menggunakan elbow method dan koefisien silhouette
#clustering kmeans dilakukan dengan menjalankan algoritmanya menggunakan nilai k 2 hingga 15
intertia = []
silhouette_coefficients = []
K = range(2,10)
for k in K:
    agglom = AgglomerativeClustering(n_clusters=k, random_state=0).fit(X)
    intertia.append(agglom.inertia_)
    score = silhouette_score(X, agglom.labels_,  metric='euclidean')
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
                color=colors[labels2[i]], 
                marker='o', 
                s=70) #untuk mengatur ukuran marker

# Menambahkan label sumbu dan judul
plt.xlabel('IPK')
plt.ylabel('LamaStudi')
plt.title('Visualisasi Klaster dengan Scatter Plot 2')

# Menambahkan legenda
for i, color in enumerate(colors):
    plt.scatter([], [], color=color, label=f'Klaster {i}')
    plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1))

# Menampilkan plot
plt.show()

unique_clusters, cluster_counts = np.unique(labels2, return_counts=True)

df_new = df[['IPK', 'LamaStudi']]
df_new['Label'] = labels2
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


# Tampilkan hasil
for cluster, count in zip(unique_clusters, cluster_counts):
    print(f"Cluster {cluster}: {count} anggota")

#liat tabel distance matrix
from scipy.spatial import distance_matrix
dist_matrix = distance_matrix(X,X)
print(dist_matrix)

#bikin bagan hierarki dan dendrogram
from scipy.cluster import hierarchy
Z = hierarchy.linkage(dist_matrix, 'single')
dendro = hierarchy.dendrogram(Z)


















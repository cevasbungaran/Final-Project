# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 21:07:13 2024

@author: cevas
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle

df = pd.read_excel('Data Lulusan 2018-2023 Bersih.xlsx')
df.head()
df.shape
df.cov()

# Pilih fitur-fitur 
features_df = df[['IPK', 'SEMESTER TEMPUH']]
features_df.shape
#Buat array Numpy utk features
features_np = np.array(features_df.values)

# Buat numpy array X dari dataframe features_df
X = np.array(features_df.values)

#Import library k-Means
from sklearn.cluster import KMeans

#import library silhoutte score
from sklearn.metrics import silhouette_score

#clustering kmeans dengan pemeriksaan kualitas hasil cluster menggunakan elbow method dan koefisien silhouette
#clustering kmeans dilakukan dengan menjalankan algoritmanya menggunakan nilai k 2 hingga 15
inertia = []
silhouette_coefficients = []
K = range(2,10)
for k in K:
    kmeans_model = KMeans(n_clusters=k, random_state=0).fit(X)
    inertia.append(kmeans_model.inertia_)
    score = silhouette_score(X, kmeans_model.labels_,  metric='euclidean')
    silhouette_coefficients.append(score)
    print(k, score)
    
#visualisasi hasil elbow method    
plt.plot(K, inertia, marker= "o")
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

kmeans_model = KMeans(n_clusters=7, random_state=0).fit(X)

# Menyimpan label hasil clustering
klaster_objek = kmeans_model.labels_

#
centroids = kmeans_model.cluster_centers_

#Simpan model clustering (agar dapat digunakan lagi lain kali)
pickle.dump(kmeans_model, open('kmeans_model', 'wb'))

# Baca model dan gunakan kembali untuk memprediksi cluster objek baru
loaded_model = pickle.load(open('kmeans_model', 'rb'))


# EKSPLORASI
df_new = df[['THN AKD LULUS', 'PROGRAM STUDI', 'IPK', 'SEMESTER TEMPUH', 'JALUR', 'Kota asal SMA', 'Provinsi asal SMA']]
df_new['Cluster'] = klaster_objek

# Membuat variabel untuk masing-masing cluster
Cluster_0 = df_new[df_new['Cluster']==0]
Cluster_1 = df_new[df_new['Cluster']==1]
Cluster_2 = df_new[df_new['Cluster']==2]
Cluster_3 = df_new[df_new['Cluster']==3]
Cluster_4 = df_new[df_new['Cluster']==4]
Cluster_5 = df_new[df_new['Cluster']==5]
Cluster_6 = df_new[df_new['Cluster']==6]

# Boxplot IPK
dataframeLabel = [Cluster_0, Cluster_1, Cluster_2, Cluster_3, Cluster_4, Cluster_5, Cluster_6]
labels = ['Cluster_0', 'Cluster_1', 'Cluster_2', 'Cluster_3', 'Cluster_4', 'Cluster_5', 'Cluster_6']
boxplot_data = [df['IPK'].values for df in dataframeLabel]

plt.figure(figsize=(8, 6))  # Adjust the figure size as per your preference

# Create the box plots for each DataFrame
plt.boxplot(boxplot_data, labels=labels)
plt.xlabel('Cluster')
plt.ylabel('IPK')
plt.title('Sebaran IPK Untuk Tiap Cluster')
plt.grid(True)  # Add grid lines for better readability
plt.show()

# Boxplot lama studi
dataframeLabel = [Cluster_0, Cluster_1, Cluster_2, Cluster_3, Cluster_4, Cluster_5, Cluster_6]
labels = ['Cluster_0', 'Cluster_1', 'Cluster_2', 'Cluster_3', 'Cluster_4', 'Cluster_5', 'Cluster_6']
boxplot_data = [df['SEMESTER TEMPUH'].values for df in dataframeLabel]

plt.figure(figsize=(8, 6))  # Adjust the figure size as per your preference

# Create the box plots for each DataFrame
plt.boxplot(boxplot_data, labels=labels)
plt.xlabel('Cluster')
plt.ylabel('SEMESTER TEMPUH')
plt.title('Sebaran Lama Studi Untuk Tiap Cluster')
plt.grid(True)  # Add grid lines for better readability
plt.show()

def hitung_statistik_lamaStudi(df, kategori, kolom_lama_studi='SEMESTER TEMPUH'):
    result_df = pd.DataFrame()
    
    # Hitung rata-rata Lama Studi berdasarkan kategori
    rata_rata = df.groupby(kategori)[kolom_lama_studi].mean()
    
    # Hitung modus Lama Studi berdasarkan kategori
    modus = df.groupby(kategori)[kolom_lama_studi].apply(lambda x: x.mode().iloc[0] if not x.mode().empty else None)
    
    # Hitung nilai minimum Lama Studi berdasarkan kategori
    nilai_min = df.groupby(kategori)[kolom_lama_studi].min()
    
    # Hitung nilai median Lama Studi berdasarkan kategori
    nilai_median = df.groupby(kategori)[kolom_lama_studi].median()
    
    # Hitung nilai maksimum Lama Studi berdasarkan kategori
    nilai_max = df.groupby(kategori)[kolom_lama_studi].max()
    
    # Mengisi DataFrame hasil perhitungan
    result_df['Rata-rata Lama Studi'] = rata_rata
    result_df['Modus Lama Studi'] = modus
    result_df['Nilai Minimum Lama Studi'] = nilai_min
    result_df['Nilai Median Lama Studi'] = nilai_median
    result_df['Nilai Maksimum Lama Studi'] = nilai_max
    result_df['Total Lulusan'] = df[kategori].value_counts()
    
    sort_result_df = result_df.sort_values(by='Rata-rata Lama Studi', ascending=True)
    return sort_result_df

def hitung_statistik_ipk(df, kategori, kolom_ipk='IPK'):
    result_df = pd.DataFrame()
    
    # Hitung rata-rata Lama Studi berdasarkan kategori
    rata_rata = df.groupby(kategori)[kolom_ipk].mean()
    
    # Hitung modus Lama Studi berdasarkan kategori
    modus = df.groupby(kategori)[kolom_ipk].apply(lambda x: x.mode().iloc[0] if not x.mode().empty else None)
    
    # Hitung nilai minimum Lama Studi berdasarkan kategori
    nilai_min = df.groupby(kategori)[kolom_ipk].min()
    
    # Hitung nilai median Lama Studi berdasarkan kategori
    nilai_median = df.groupby(kategori)[kolom_ipk].median()
    
    # Hitung nilai maksimum Lama Studi berdasarkan kategori
    nilai_max = df.groupby(kategori)[kolom_ipk].max()
    
    # Mengisi DataFrame hasil perhitungan
    result_df['Rata-rata IPK'] = rata_rata
    result_df['Modus IPK'] = modus
    result_df['Nilai Minimum IPK'] = nilai_min
    result_df['Nilai Median IPK'] = nilai_median
    result_df['Nilai Maksimum IPK'] = nilai_max
    result_df['Total Lulusan'] = df[kategori].value_counts()
    
    sort_result_df = result_df.sort_values(by='Rata-rata IPK', ascending=False)
    return sort_result_df

def frekuensi_kolom(df, cluster_column, target_column):
    
    # Group by the cluster column and target column, then count the occurrences
    frequency_df = df.groupby([cluster_column, target_column]).size().reset_index(name='count')
    
    # Pivot the table to get a better view
    frequency_pivot = frequency_df.pivot(index=cluster_column, columns=target_column, values='count').fillna(0)
    
    return frequency_pivot

frekuensi_jalur_masuk = frekuensi_kolom(df_new, 'Cluster', 'JALUR')
frekuensi_provinsi = frekuensi_kolom(df_new, 'Cluster', 'Provinsi asal SMA')
    

# Melihat statistik IPK dan lama studi tiap jalur masuk
# ipk
statistik_klaster_ipk = hitung_statistik_ipk(df_new, 'Cluster')
# lama studi
statistik_klaster_lamaStudi = hitung_statistik_lamaStudi(df_new, 'Cluster')

# Melihat total jalur masuk untuk tiap klaster
jalur_masuk_klaster0 = Cluster_0['JALUR'].value_counts()
jalur_masuk_klaster1 = Cluster_1['JALUR'].value_counts()
jalur_masuk_klaster2 = Cluster_2['JALUR'].value_counts()
jalur_masuk_klaster3 = Cluster_3['JALUR'].value_counts()
jalur_masuk_klaster4 = Cluster_4['JALUR'].value_counts()

# Melihat total provinsi asal sma untuk tiap klaster
provinsi_klaster0 = Cluster_0['Provinsi asal SMA'].value_counts()
provinsi_klaster1 = Cluster_1['Provinsi asal SMA'].value_counts()
provinsi_klaster2 = Cluster_2['Provinsi asal SMA'].value_counts()
provinsi_klaster3 = Cluster_3['Provinsi asal SMA'].value_counts()
provinsi_klaster4 = Cluster_4['Provinsi asal SMA'].value_counts()

# Melihat total tahun akademik kelulusan untuk tiap klaster
tahun_klaster0 = Cluster_0['THN AKD LULUS'].value_counts()
tahun_klaster1 = Cluster_1['THN AKD LULUS'].value_counts()
tahun_klaster2 = Cluster_2['THN AKD LULUS'].value_counts()
tahun_klaster3 = Cluster_3['THN AKD LULUS'].value_counts()
tahun_klaster4 = Cluster_4['THN AKD LULUS'].value_counts()









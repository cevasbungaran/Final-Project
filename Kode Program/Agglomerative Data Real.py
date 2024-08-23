# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 12:35:13 2024

@author: cevas
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
import seaborn as sns


df = pd.read_excel('Data Lulusan 2018-2023 Bersih.xlsx')
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
features_df = df[['IPK', 'SEMESTER TEMPUH']]
features_df.shape
#Buat array Numpy utk features
features_np = np.array(features_df.values)

# Buat numpy array X dari dataframe features_df
X = np.array(features_df.values)

#bikin model agglomerative single linkage
from sklearn.cluster import AgglomerativeClustering


from sklearn.metrics import silhouette_score
#clustering kmeans dengan pemeriksaan kualitas hasil cluster menggunakan elbow method dan koefisien silhouette
#clustering kmeans dilakukan dengan menjalankan algoritmanya menggunakan nilai k 2 hingga 15
intertia = []
silhouette_coefficients = []
K = range(2,10)
for k in K:
    agglom = AgglomerativeClustering(n_clusters=k, linkage='complete').fit(X)
    score = silhouette_score(X, agglom.labels_,  metric='euclidean')
    print(k, score)
    
    
    
    
    
    




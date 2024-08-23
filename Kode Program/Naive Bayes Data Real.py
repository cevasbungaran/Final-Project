# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 18:56:24 2023

@author: cevas
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
from sklearn.metrics import mean_squared_error

df = pd.read_excel('Data Lulusan 2018-2023 Maks Smt 14.xlsx')
df.info()
###### PEMBUATAN MODEL PREDIKSI IPK ######
#data preprosesing
def kategori_ipk(ipk):
    if 3.5 <= ipk <= 4.0:
        return 1
    elif 3.0 <= ipk < 3.5:
        return 2
    elif 2.5 <= ipk < 3.0:
        return 3
    elif 2.0 <= ipk < 2.5:
        return 4
    
# Menerapkan fungsi kategorikan_ipk ke kolom IPK
df['RankIPK'] = df['IPK'].apply(kategori_ipk)

#Buat label kelas
df_labels_rank = df['RankIPK']

#Buat array Numpy utk kelas/label # numpy array 
label_np_ipk = np.array(df_labels_rank.values)

#Cek dimensi array
label_np_ipk.shape

# Import LabelEncoder
from sklearn import preprocessing
#creating labelEncoder
le = preprocessing.LabelEncoder()

label_en_ipk = le.fit_transform(label_np_ipk)

# Pemilihan fitur
lamaStudi = df['SEMESTER TEMPUH']
jalur = df['JALUR']
prodi = df['PROGRAM STUDI']

# Mengubah label string dan fitur string ke numerik
jalur_en = le.fit_transform(jalur)
prodi_en = le.fit_transform(prodi)

# Penyatuan Fitur
features_df = pd.DataFrame()
features_df['PROGRAM STUDI'] = prodi_en
features_df['SEMESTER TEMPUH'] = lamaStudi
features_df['JALUR'] = jalur_en

# Menampilkan arti dari nilai numerik pada kolom Program Studi
prodi_en_unik = features_df['PROGRAM STUDI'].unique()
prodi_df_unik = df['PROGRAM STUDI'].unique()
prodi_en_df = pd.DataFrame()
prodi_en_df['Prodi String'] = prodi_df_unik
prodi_en_df['Prodi Numerik'] = prodi_en_unik
prodi_en_df = prodi_en_df.sort_values(by='Prodi Numerik', ascending=True)
print(prodi_en_df)

# Menampilkan arti dari nilai numerik pada kolom Jalur
jalur_en_unik = features_df['JALUR'].unique()
jalur_df_unik = df['JALUR'].unique()
jalur_en_df = pd.DataFrame()
jalur_en_df['Jalur String'] = jalur_df_unik
jalur_en_df['Jalur Numerik'] = jalur_en_unik
jalur_en_df = jalur_en_df.sort_values(by='Jalur Numerik', ascending=True)
print(jalur_en_df)

#Buat array Numpy utk features
features_np = np.array(features_df.values)

#Cek dimensi
label_np_ipk.shape
features_np.shape

# kemitraan = 0, pmdk = 1, usm1 = 2, usm2 = 3, usm3 = 4

# Import train_test_split function
from sklearn.model_selection import train_test_split

# Split dataset into training set and test set: 70% training and 30% test
X_train, X_test, y_train, y_test = train_test_split(features_np, label_np_ipk, test_size=0.3) 

## NAIVE BAYES ##
from sklearn.naive_bayes import GaussianNB

NBC_model = GaussianNB() #model rank ipk

#Train the model using the training sets
NBC_model.fit(X_train, y_train)

#Predict the response for test dataset
Y_pred = NBC_model.predict(X_test)

# Evaluasi model menggunakan Mean Squared Error
mse = mean_squared_error(y_test, Y_pred)
print('Nilai MSE Model Prediksi IPK:',mse)

# Pelatihan data menggunakan K-Fold Cross Validation
from sklearn.model_selection import cross_val_score, KFold
best_k = None
best_mse_ipk = float('inf')
best_model_ipk = None
fold = 5
while fold <= 10:
    kfold = KFold(n_splits=fold, shuffle=True)
    scores = cross_val_score(NBC_model, features_np, label_np_ipk, cv=kfold, scoring='neg_mean_squared_error')
    mse_scores = -scores.mean()  # Ambil nilai negatif kembali
    print(f'Mean MSE Prediksi IPK: {mse_scores}', 'k=',fold)
    
    if mse_scores < best_mse_ipk:
        best_mse_ipk = mse_scores
        best_k = fold

        # Latih ulang model dengan seluruh data dan simpan sebagai model terbaik
        best_model_ipk = GaussianNB()
        best_model_ipk.fit(features_np, label_np_ipk)
    
    fold+=1

print(f'Best k: {best_k} with MSE: {best_mse_ipk}')

# Menyimpan model nbc IPK dengan nilai k terbaik
if best_model_ipk is not None:
    pkl_nbc_ipk = "nbc_modelIPK.pkl" 
    with open(pkl_nbc_ipk, 'wb') as file:
        pickle.dump(best_model_ipk, file)



###### PEMBUATAN MODEL PREDIKSI LAMA STUDI ######
#Buat label kelas
df_labels2 = df[['SEMESTER TEMPUH']]

#Buat array Numpy utk kelas/label
label_np2 = np.array(df_labels2.values) # numpy array 

#Cek dimensi array
label_np2.shape
print(label_np2)

#Ubah matriks 1 kolom ke 1 baris (spy dpt jadi parameter le.fit_transform(.))
df_label_np2= label_np2.ravel()
df_label_np2.shape  # Cek dimensinya

# Import LabelEncoder
from sklearn import preprocessing
#creating labelEncoder
le = preprocessing.LabelEncoder()

# Pemilihan fitur
jalur2 = df['JALUR']
prodi2 = df['PROGRAM STUDI']
ipk = df['IPK']

# Mengubah label string dan fitur string ke numerik
jalur2_en = le.fit_transform(jalur2)
prodi2_en = le.fit_transform(prodi2)

# Penyatuan Fitur
features_df2 = pd.DataFrame()
features_df2['PROGRAM STUDI'] = prodi2_en
features_df2['IPK'] = ipk
features_df2['JALUR'] = jalur2_en

# Menampilkan arti dari nilai numerik pada kolom Program Studi
prodi_en_unik2 = features_df2['PROGRAM STUDI'].unique()
prodi_df_unik2 = df['PROGRAM STUDI'].unique()
prodi_en_df2 = pd.DataFrame()
prodi_en_df2['Prodi String'] = prodi_df_unik2
prodi_en_df2['Prodi Numerik'] = prodi_en_unik2
prodi_en_df2 = prodi_en_df2.sort_values(by='Prodi Numerik', ascending=True)
print(prodi_en_df2)

# Menampilkan arti dari nilai numerik pada kolom Jalur
jalur_en_unik2 = features_df2['JALUR'].unique()
jalur_df_unik2 = df['JALUR'].unique()
jalur_en_df2= pd.DataFrame()
jalur_en_df2['Jalur String'] = jalur_df_unik2
jalur_en_df2['Jalur Numerik'] = jalur_en_unik2
jalur_en_df2 = jalur_en_df2.sort_values(by='Jalur Numerik', ascending=True)
print(jalur_en_df2)

#Buat array Numpy utk features
features_np2 = np.array(features_df2.values)

#Cek dimensi
features_np2.shape


# Import train_test_split function
from sklearn.model_selection import train_test_split

# Split dataset into training set and test set: 70% training and 30% test
X_train2, X_test2, y_train2, y_test2 = train_test_split(features_np2, df_label_np2, test_size=0.3) 

## NAIVE BAYES ##
from sklearn.naive_bayes import GaussianNB

NBC_model2 = GaussianNB() #model rank ipk

#Train the model using the training sets
NBC_model2.fit(X_train2, y_train2)

#Predict the response for test dataset
Y_pred2 = NBC_model2.predict(X_test2)

# Evaluasi model menggunakan Mean Squared Error
mse2 = mean_squared_error(y_test2, Y_pred2)
print('Nilai MSE Model Prediksi Lama Studi:',mse2)

# Pelatihan data menggunakan K-Fold Cross Validation
from sklearn.model_selection import cross_val_score, KFold
best_k = None
best_mse_lama_studi = float('inf')
best_model_lama_studi = None
fold = 5
while fold <= 10:
    kfold = KFold(n_splits=fold, shuffle=True)
    scores = cross_val_score(NBC_model2, features_np2, df_label_np2, cv=kfold, scoring='neg_mean_squared_error')
    mse_scores = -scores.mean()  # Ambil nilai negatif kembali
    print(f'Mean MSE Prediksi Lama Studi: {mse_scores}', 'k=',fold)
    
    if mse_scores < best_mse_lama_studi:
        best_mse_lama_studi = mse_scores
        best_k = fold

        # Latih ulang model dengan seluruh data dan simpan sebagai model terbaik
        best_model_lama_studi = GaussianNB()
        best_model_lama_studi.fit(features_np2, df_label_np2)
    
    fold+=1

print(f'Best k: {best_k} with MSE: {best_mse_lama_studi}')

# Menyimpan model nbc lama studi dengan nilai k terbaik
if best_model_lama_studi is not None:
    pkl_nbc_lamaStudi = "nbc_model_lamaStudi.pkl" 
    with open(pkl_nbc_lamaStudi, 'wb') as file:
        pickle.dump(best_model_lama_studi, file)


# Load Model NBC IPK
pkl_filename2 = "nbc_modelIPK.pkl"  
with open(pkl_filename2, 'rb') as file:  
    loaded_model_nbc = pickle.load(file)

# Load Model NBC Lama Studi
pkl_filename2 = "nbc_model_lamaStudi.pkl"  
with open(pkl_filename2, 'rb') as file:  
    loaded_model_nbc2 = pickle.load(file)







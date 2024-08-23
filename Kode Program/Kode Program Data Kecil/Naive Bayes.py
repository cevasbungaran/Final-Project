# -*- coding: utf-8 -*-
"""
Created on Sun Sep 17 17:09:41 2023

@author: cevas
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
from sklearn.metrics import mean_squared_error

df = pd.read_csv('Data Mahasiswa Baru.csv')

###### KLASIFIKASI ######

#Buat label kelas
df_labels = df[['IPK']]
df_labels2 = df[['LamaStudi']]   # hasil: 1 kolom 
#Buat array Numpy utk kelas/label
label_np = np.array(df_labels.values) # numpy array 
label_np2 = np.array(df_labels2.values)

#Cek dimensi array
label_np.shape
label_np2.shape
print(label_np)
print(label_np2)

#Ubah matriks 1 kolom ke 1 baris (spy dpt jadi parameter le.fit_transform(.))
df_label_np= label_np.ravel()
df_label_np.shape  # Cek dimensinya
df_label_np2= label_np2.ravel()
df_label_np2.shape 

# Import LabelEncoder
from sklearn import preprocessing
#creating labelEncoder
le = preprocessing.LabelEncoder()

# Pemilihan fitur prediksi IPK
lamaStudi = df['LamaStudi']
jalur = df['JalurMasuk']

# Pemilihan fitur prediksi LamaStudi
ipk = df['IPK']
# jalur masuk sama seperti di atas

# Mengubah label string dan fitur string ke numerik
jalur_en = le.fit_transform(jalur)
ipk_en = le.fit_transform(ipk)

# Mengubah label IPK menggunakan label encoder
label_en = le.fit_transform(df_label_np) 

# Penyatuan Fitur prediksi IPK
features_df = pd.DataFrame()
features_df['LamaStudi'] = lamaStudi
features_df['JalurMasuk'] = jalur_en

# Penyatuan Fitur prediksi lama studi
features_df2 = pd.DataFrame()
features_df2['IPK'] = ipk_en
features_df2['JalurMasuk'] = jalur_en

# Buat array Numpy utk features
features_np = np.array(features_df.values)
features_np2 = np.array(features_df2.values)

#Cek dimensi
features_np.shape
features_np2.shape

# kemitraan = 0, pmdk = 1, usm1 = 2, usm2 = 3, usm3 = 4

# Import train_test_split function
from sklearn.model_selection import train_test_split

# Split dataset into training set and test set: 70% training and 30% test
X_train, X_test, y_train, y_test = train_test_split(features_np, label_en, test_size=0.3) 
X_train2, X_test2, y_train2, y_test2 = train_test_split(features_np2, df_label_np2, test_size=0.3) 


## NAIVE BAYES ##
from sklearn.naive_bayes import GaussianNB

NBC_model = GaussianNB() #model rank ipk
NBC_model2 = GaussianNB() #model lama studi

#Train the model using the training sets
NBC_model.fit(X_train, y_train)
NBC_model2.fit(X_train2, y_train2)

#Predict the response for test dataset
Y_pred = NBC_model.predict(X_test)
Y_pred2 = NBC_model2.predict(X_test2)

# Evaluasi model predksi IPK menggunakan Mean Squared Error
mse = mean_squared_error(y_test, Y_pred)
print('Nilai MSE Model Prediksi IPK:',mse)

# Pelatihan data menggunakan K-Fold Cross Validation
from sklearn.model_selection import cross_val_score, KFold
fold = 5
while fold <= 10:
    kfold = KFold(n_splits=fold, shuffle=True)
    scores = cross_val_score(NBC_model, features_np, label_en, cv=kfold, scoring='neg_mean_squared_error')
    mse_scores = -scores  # Ambil nilai negatif kembali
    print(f'Mean MSE Peringkat IPK: {mse_scores.mean()}', 'k=',fold)
    fold+=1

rmse = mean_squared_error(y_test, Y_pred, squared=False)
print('Nilai RMSE Model Prediksi IPK:',rmse)

mse2 = mean_squared_error(y_test2, Y_pred2)
print('Nilai MSE Model Prediksi LamaStudi:',mse2)

# Pelatihan data menggunakan K-Fold Cross Validation
from sklearn.model_selection import cross_val_score, KFold
fold = 5
while fold <= 10:
    kfold = KFold(n_splits=fold, shuffle=True)
    scores = cross_val_score(NBC_model2, features_np2, df_label_np2, cv=kfold, scoring='neg_mean_squared_error')
    mse_scores = -scores  # Ambil nilai negatif kembali
    print(f'Mean MSE Peringkat Lama Studi: {mse_scores.mean()}', 'k=',fold)
    fold+=1

#Simpan model dgn nama: Model_kNN_Irish.pkl
pkl_filename = "nbc_modelIPK.pkl"  
with open(pkl_filename, 'wb') as file:  
    pickle.dump(NBC_model, file)

pkl_filename2 = "nbc_model_lamaStudi.pkl"  
with open(pkl_filename2, 'wb') as file:  
    pickle.dump(NBC_model2, file)


# Prediksi IPK
# Load model yang sudah disimpan
import pickle 
pkl_filename = "nbc_modelIPK.pkl"  
with open(pkl_filename, 'rb') as file:  
    loaded_model_nbc = pickle.load(file)

#load data extra
df_extra = pd.read_csv('Data Extra.csv', delimiter = ',')

# Pemilihan fitur
lamaStudi_ekstra = df_extra['LamaStudi']
jalur_ekstra = df_extra['JalurMasuk']

# Mengubah label string dan fitur string ke numerik
jalur_en_ekstra = le.fit_transform(jalur_ekstra)

# Penyatuan Fitur
features_df_ekstra = pd.DataFrame()
features_df_ekstra['LamaStudi'] = lamaStudi_ekstra
features_df_ekstra['JalurMasuk'] = jalur_en_ekstra

#Buat array Numpy utk features
features_np_ekstra = np.array(features_df_ekstra.values)

# Buat variabel baru untuk fitur data baru
X_extra = features_np_ekstra

#Lakukan prediksi (mencari nilai mpg)
Y_pred_extra = loaded_model_nbc.predict(X_extra)
print(Y_pred_extra)

# Membuat dataframe baru untuk menyimpan hasil prediksi
Y_pred_df_extra = pd.DataFrame(Y_pred_extra)

# Membuat kolom baru untuk memasukan hasil prediksi ke df_extra
df_extra['Prediksi IPK'] = Y_pred_df_extra
#df_extra = df_extra.drop(['Prediksi IPK'], axis=1)

# Menghitung error
mse_prediksi_ipk = mean_squared_error(df_extra['IPK'], Y_pred_extra)
print('Nilai MSE Prediksi IPK:',mse_prediksi_ipk)


# Prediksi LamaStudi
pkl_filename2 = "nbc_model_lamaStudi.pkl"  
with open(pkl_filename2, 'rb') as file:  
    loaded_model_nbc2 = pickle.load(file)

# Pemilihan fitur
jalur2_extra = df_extra['JalurMasuk']
ipk_extra = df_extra['IPK']

# Mengubah label string dan fitur string ke numerik
jalur2_en_extra = le.fit_transform(jalur2_extra)

# Penyatuan Fitur
features_df2_extra = pd.DataFrame()
features_df2_extra['IPK'] = ipk_extra
features_df2_extra['JalurMasuk'] = jalur2_en_extra 
    
#Buat array Numpy utk features
features_np2_ekstra = np.array(features_df2_extra.values)
    
X_extra2 = features_np2_ekstra
    
Y_pred_extra2 = loaded_model_nbc2.predict(X_extra2)
print(Y_pred_extra2)
    
Y_pred_df_extra2 = pd.DataFrame(Y_pred_extra2)
    
df_extra['Prediksi LamaStudi'] = Y_pred_df_extra2
#df_extra = df_extra.drop(['Prediksi LamaStudi'], axis=1)    

mse_prediksi_lamaStudi = mean_squared_error(df_extra['LamaStudi'], Y_pred_extra2)
print('Nilai MSE Prediksi Lama Studi:',mse_prediksi_lamaStudi)










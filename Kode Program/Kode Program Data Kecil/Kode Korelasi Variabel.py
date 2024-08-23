# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 16:41:10 2023

@author: cevas
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy import cov
import seaborn as sns
from scipy.stats import chi2_contingency

df = pd.read_csv('Data Mahasiswa Baru.csv')

df.describe()
df.info()

df['IPK'].mean()
df['IPK'].std()

def kategorikan_ipk(ipk):
    if 3.5 <= ipk <= 4.0:
        return 1
    elif 3.0 <= ipk < 3.5:
        return 2
    elif 2.5 <= ipk < 3.0:
        return 3
    elif 2.0 <= ipk < 2.5:
        return 4

def kategori_jalurMasuk(JalurMasuk):
    if JalurMasuk == 'Kemitraan':
        return 1
    elif JalurMasuk == 'PMDK':
        return 2
    elif JalurMasuk == 'USM 1':
        return 3
    elif JalurMasuk == 'USM 2':
        return 4
    elif JalurMasuk == 'USM 3':
        return 5

def kategorikan_studi(lamaStudi):
    if 8 <= lamaStudi <= 10:
        return 1
    elif 11 <= lamaStudi <= 13:
        return 2
    elif lamaStudi == 14:
        return 3

# Menerapkan function  ke kolom IPK dan JalurMasuk
df['RankIPK'] = df['IPK'].apply(kategorikan_ipk)
df['RankJalurMasuk'] = df['JalurMasuk'].apply(kategori_jalurMasuk)
df['RankLamaStudi'] = df['LamaStudi'].apply(kategorikan_studi)

data1 = df['IPK']
data2 = df['LamaStudi']
data3 = df['JalurMasuk']

print('data1: mean=%.3f stdv=%.3f' % (data1.mean(), data1.std()))
print('data2: mean=%.3f stdv=%.3f' % (data2.mean(), data2.std()))
#print('data3: mean=%.3f stdv=%.3f' % (data3.mean(), data3.std()))

data1_rank = df['RankIPK']
data2_rank = df['RankLamaStudi']
data3_rank = df['RankJalurMasuk']

print('data1_rank: mean=%.3f stdv=%.3f' % (data1_rank.mean(), data1_rank.std()))
print('data2_rank: mean=%.3f stdv=%.3f' % (data2_rank.mean(), data2_rank.std()))
print('data3_rank: mean=%.3f stdv=%.3f' % (data3_rank.mean(), data3_rank.std()))

plt.scatter(data3, data1)
plt.show()

covariance = cov(data1, data2)
print(covariance)

pmdk = df[df['JalurMasuk'] == 'PMDK']
kemitraan = df[df['JalurMasuk'] == 'Kemitraan']
usm1 = df[df['JalurMasuk'] == 'USM 1']
usm2 = df[df['JalurMasuk'] == 'USM 2']
usm3 = df[df['JalurMasuk'] == 'USM 3']


dataframe = [kemitraan, pmdk, usm1, usm2, usm3]
labels = ['Kemitraan', 'PMDK', 'USM 1', 'USM 2', 'USM 3']
boxplot_data = [df['IPK'].values for df in dataframe]

plt.figure(figsize=(8, 6))  # Adjust the figure size as per your preference

# Create the box plots for each DataFrame
plt.boxplot(boxplot_data, labels=labels)
plt.xlabel('DataFrames')
plt.ylabel('IPK')
plt.title('Sebaran IPK untuk tiap Jalur Masuk')
plt.grid(True)  # Add grid lines for better readability
plt.show()

boxplot_data2 = [df['LamaStudi'].values for df in dataframe]

plt.figure(figsize=(8, 6))  # Adjust the figure size as per your preference

# Create the box plots for each DataFrame
plt.boxplot(boxplot_data2, labels=labels)
plt.xlabel('DataFrames')
plt.ylabel('LamaStudi')
plt.title('Sebaran Lama Studi untuk tiap Jalur Masuk')
plt.grid(True)  # Add grid lines for better readability
plt.show()

#membuat scatter plot
avg_lama_studi = df.groupby('IPK')['LamaStudi'].mean().reset_index()
avg_lama_studi_pmdk = pmdk.groupby('IPK')['LamaStudi'].mean().reset_index()
avg_lama_studi_kemitraan = kemitraan.groupby('IPK')['LamaStudi'].mean().reset_index()
avg_lama_studi_usm1 = usm1.groupby('IPK')['LamaStudi'].mean().reset_index()
avg_lama_studi_usm2 = usm2.groupby('IPK')['LamaStudi'].mean().reset_index()
avg_lama_studi_usm3 = usm3.groupby('IPK')['LamaStudi'].mean().reset_index()

#scatter plot data besar
plt.scatter(avg_lama_studi['IPK'], avg_lama_studi['LamaStudi']) # plotting the clusters
plt.title('Pola Antara IPK dengan LamaStudi')
plt.xlabel("IPK") # X-axis label
plt.ylabel("LamaStudi") # Y-axis label
plt.show() 

#scatter plot pmdk
plt.scatter(avg_lama_studi_pmdk['IPK'], avg_lama_studi_pmdk['LamaStudi']) # plotting the clusters
plt.title('Pola Antara IPK dengan LamaStudi Jalur PMDK')
plt.xlabel("IPK") # X-axis label
plt.ylabel("LamaStudi") # Y-axis label
plt.show() 

#scatter plot kemitraan
# Menghitung koefisien regresi linier
plt.scatter(avg_lama_studi_kemitraan['IPK'], avg_lama_studi_kemitraan['LamaStudi'])
plt.title('Pola Antara IPK dengan LamaStudi Jalur Kemitraan')
plt.xlabel("IPK") # X-axis label
plt.ylabel("LamaStudi") # Y-axis label
plt.show() 

#scatter plot usm 1
# Menghitung koefisien regresi linier
plt.scatter(avg_lama_studi_usm1['IPK'], avg_lama_studi_usm1['LamaStudi']) # plotting the clusters
plt.title('Pola Antara IPK dengan LamaStudi Jalur USM 1')
plt.xlabel("IPK") # X-axis label
plt.ylabel("LamaStudi") # Y-axis label
plt.show() 

#scatter plot usm 2
# Menghitung koefisien regresi linier
plt.scatter(avg_lama_studi_usm2['IPK'], avg_lama_studi_usm2['LamaStudi']) # plotting the clusters
plt.title('Pola Antara IPK dengan LamaStudi Jalur USM 2')
plt.xlabel("IPK") # X-axis label
plt.ylabel("LamaStudi") # Y-axis label
plt.show() 

#scatter plot usm 3
# Menghitung koefisien regresi linier
plt.scatter(avg_lama_studi_usm3['IPK'], avg_lama_studi_usm3['LamaStudi']) # plotting the clusters
plt.title('Pola Antara IPK dengan LamaStudi Jalur USM 3')
plt.xlabel("IPK") # X-axis label
plt.ylabel("LamaStudi") # Y-axis label
plt.show() 



#buat data
data1 = df['IPK']
data2 = df['LamaStudi']

#uji pearson
from scipy.stats import pearsonr

correlationLamaStudi_ipkP, p_value = pearsonr(data1, data2)
print("Korelasi Pearson Lama Studi dan IPK:", correlationLamaStudi_ipkP)
print("Nilai p:", p_value)


# Uji chi square
from scipy.stats import chi2_contingency
from scipy.stats import chi2 as chi2_distribution

def chi_square(df, atribut1, atribut2, alpha):
    # Membuat tabel kontigensi
    contigency_table = pd.crosstab(df[atribut1], df[atribut2])
    # Print tabel kontigensi
    print(contigency_table)
    # Menghitung nilai chi square, p value, dof, dan expected frequencies
    stat, p, dof, expected = chi2_contingency(contigency_table)
    # Menentukan nilai probabilitas
    prob = alpha
    # Menghitung critical value
    critical = chi2_distribution.ppf(1-prob, dof)
    # Print nilai probabilitas, critical value, dan nilai chi square
    print('probability=%.3f, critical=%.3f, stat=%.3f' % (prob, critical, stat))
    print('dof=',dof)
    # Menentukan tolak H0 atau terima H0
    if abs(stat) >= critical:
     print('Dependent (reject H0)')
    else:
     print('Independent (fail to reject H0)')

## Uji chi square dengan alpha=0.5
chiSquare_lamaStudi = chi_square(df, 'JalurMasuk', 'LamaStudi', 0.05) #idependent
chiSquare_ipk = chi_square(df, 'JalurMasuk', 'IPK', 0.05) #independent













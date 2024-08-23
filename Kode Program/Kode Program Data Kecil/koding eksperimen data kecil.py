# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 22:08:48 2023

@author: cevas
"""

#import tools pandas
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle

#membaca dataset dengan format csv
df = pd.read_csv('Data Mahasiswa Baru.csv')

#EKSPLORASI
df.info()
df.dtypes
df.describe(include='all')
df['JalurMasuk'].describe()

df.mean()
df.std()

df.mode()
df.median()
df.max()
df.min()


df.head()
df.tail()

# Menghitung frekuensi masing-masing JalurMasuk
jalur_masuk_counts = df['JalurMasuk'].value_counts()

# Membuat bar plot
plt.figure(figsize=(8, 6))
ax = jalur_masuk_counts.plot(kind='bar', color='skyblue')
plt.title('Frekuensi Jalur Masuk')
plt.xlabel('Jalur Masuk')
plt.ylabel('Frekuensi')
plt.xticks(rotation=45)
plt.tight_layout()

for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() + p.get_width() / 2, p.get_height() + 1.0), ha='center')

plt.tight_layout()
plt.show()


###############################################################
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
df.info()

pmdk = df[df['JalurMasuk'] == 'PMDK']
kemitraan = df[df['JalurMasuk'] == 'Kemitraan']
usm1 = df[df['JalurMasuk'] == 'USM 1']
usm2 = df[df['JalurMasuk'] == 'USM 2']
usm3 = df[df['JalurMasuk'] == 'USM 3']

pmdk.describe()
kemitraan.describe()
usm1.describe()
usm2.describe()
usm3.describe()

#rata-rata dan stdv ipk
print('IPK PMDK: mean=%.3f stdv=%.3f' % (pmdk['IPK'].mean(), pmdk['IPK'].std()))
print('IPK Kemitraan: mean=%.3f stdv=%.3f' % (kemitraan['IPK'].mean(), kemitraan['IPK'].std()))
print('IPK usm1: mean=%.3f stdv=%.3f' % (usm1['IPK'].mean(), usm1['IPK'].std()))
print('IPK usm2: mean=%.3f stdv=%.3f' % (usm2['IPK'].mean(), usm2['IPK'].std()))
print('IPK usm3: mean=%.3f stdv=%.3f' % (usm3['IPK'].mean(), usm3['IPK'].std()))
#rata-rata dan stdv lama studi
print('LamaStudi PMDK: mean=%.3f stdv=%.3f' % (pmdk['LamaStudi'].mean(), pmdk['LamaStudi'].std()))
print('LamaStudi Kemitraan: mean=%.3f stdv=%.3f' % (kemitraan['LamaStudi'].mean(), kemitraan['LamaStudi'].std()))
print('LamaStudi usm1: mean=%.3f stdv=%.3f' % (usm1['LamaStudi'].mean(), usm1['LamaStudi'].std()))
print('LamaStudi usm2: mean=%.3f stdv=%.3f' % (usm2['LamaStudi'].mean(), usm2['LamaStudi'].std()))
print('LamaStudi usm3: mean=%.3f stdv=%.3f' % (usm3['LamaStudi'].mean(), usm3['LamaStudi'].std()))
# #rata-rata kategori dan stdv ipk
# print('IPK PMDK: mean=%.3f stdv=%.3f' % (pmdk['Kategori IPK'].mean(), pmdk['Kategori IPK'].std()))
# print('IPK Kemitraan: mean=%.3f stdv=%.3f' % (kemitraan['Kategori IPK'].mean(), kemitraan['Kategori IPK'].std()))
# print('IPK usm1: mean=%.3f stdv=%.3f' % (usm1['Kategori IPK'].mean(), usm1['Kategori IPK'].std()))
# print('IPK usm2: mean=%.3f stdv=%.3f' % (usm2['Kategori IPK'].mean(), usm2['Kategori IPK'].std()))
# print('IPK usm3: mean=%.3f stdv=%.3f' % (usm3['Kategori IPK'].mean(), usm3['Kategori IPK'].std()))



#histogram ipk tiap jalur masuk
plt.hist(pmdk['IPK'], bins=10, edgecolor='black')  # You can adjust the number of bins as per your preference
plt.xlabel('IPK')
plt.ylabel('Frequency')
plt.title('Persebaran IPK jalur PMDK')
plt.show()

plt.hist(kemitraan['IPK'], bins=10, edgecolor='black')  # You can adjust the number of bins as per your preference
plt.xlabel('IPK')
plt.ylabel('Frequency')
plt.title('Persebaran IPK jalur Kemitraan')
plt.show()

plt.hist(usm1['IPK'], bins=10, edgecolor='black')  # You can adjust the number of bins as per your preference
plt.xlabel('IPK')
plt.ylabel('Frequency')
plt.title('Persebaran IPK jalur USM 1')
plt.show()

plt.hist(usm2['IPK'], bins=10, edgecolor='black')  # You can adjust the number of bins as per your preference
plt.xlabel('IPK')
plt.ylabel('Frequency')
plt.title('Persebaran IPK jalur USM 2')
plt.show()

plt.hist(usm3['IPK'], bins=10, edgecolor='black')  # You can adjust the number of bins as per your preference
plt.xlabel('IPK')
plt.ylabel('Frequency')
plt.title('Persebaran IPK jalur USM 3')
plt.show()

#histogram lama studi tiap jalur masuk
plt.hist(pmdk['LamaStudi'], bins=7, edgecolor='black')  # You can adjust the number of bins as per your preference
plt.xlabel('Lama Studi')
plt.ylabel('Frequency')
plt.title('Persebaran Lama Studi jalur PMDK')
plt.show()

plt.hist(kemitraan['LamaStudi'], bins=7, edgecolor='black')  # You can adjust the number of bins as per your preference
plt.xlabel('Lama Studi')
plt.ylabel('Frequency')
plt.title('Persebaran Lama Studi jalur Kemitraan')
plt.show()

plt.hist(usm1['LamaStudi'], bins=7, edgecolor='black')  # You can adjust the number of bins as per your preference
plt.xlabel('Lama Studi')
plt.ylabel('Frequency')
plt.title('Persebaran Lama Studi jalur USM 1')
plt.show()

plt.hist(usm2['LamaStudi'], bins=7, edgecolor='black')  # You can adjust the number of bins as per your preference
plt.xlabel('Lama Studi')
plt.ylabel('Frequency')
plt.title('Persebaran Lama Studi jalur USM 2')
plt.show()

plt.hist(usm3['LamaStudi'], bins=7, edgecolor='black')  # You can adjust the number of bins as per your preference
plt.xlabel('Lama Studi')
plt.ylabel('Frequency')
plt.title('Persebaran Lama Studi jalur USM 3')
plt.show()

# #histogram katogori ipk tiap jalur masuk
# plt.bar(pmdk['JalurMasuk'], pmdk['Kategori IPK'], edgecolor='black')  # You can adjust the number of bins as per your preference
# plt.xlabel('Kategori IPK')
# plt.ylabel('Frequency')
# plt.title('Persebaran Kategori IPK tiap Jalur Masuk')
# plt.show()

# plt.hist(kemitraan['Kategori IPK'], bins=10, edgecolor='black')  # You can adjust the number of bins as per your preference
# plt.xlabel('Kategori IPK')
# plt.ylabel('Frequency')
# plt.title('Persebaran Kategori IPK jalur Kemitraan')
# plt.show()

# plt.hist(usm1['Kategori IPK'], bins=10, edgecolor='black')  # You can adjust the number of bins as per your preference
# plt.xlabel('Kategori IPK')
# plt.ylabel('Frequency')
# plt.title('Persebaran Kategori IPK jalur USM 1')
# plt.show()

# plt.hist(usm2['Kategori IPK'], bins=10, edgecolor='black')  # You can adjust the number of bins as per your preference
# plt.xlabel('Kategori IPK')
# plt.ylabel('Frequency')
# plt.title('Persebaran Kategori IPK jalur USM 2')
# plt.show()

# plt.hist(usm3['Kategori IPK'], bins=10, edgecolor='black')  # You can adjust the number of bins as per your preference
# plt.xlabel('Kategori IPK')
# plt.ylabel('Frequency')
# plt.title('Persebaran Kategori IPK jalur USM 3')
# plt.show()

#boxplot ipk tiap jalur masuk
dataframe = [kemitraan, pmdk, usm1, usm2, usm3]
labels = ['Kemitraan', 'PMDK', 'USM 1', 'USM 2', 'USM 3']
boxplot_data = [df['IPK'].values for df in dataframe]
#boxplot_data2 = [df['Kategori IPK'].values for df in dataframe]

plt.figure(figsize=(8, 6))  # Adjust the figure size as per your preference

# Create the box plots for each DataFrame
plt.boxplot(boxplot_data, labels=labels)
plt.xlabel('DataFrames')
plt.ylabel('IPK')
plt.title('Sebaran IPK untuk tiap Jalur Masuk')
plt.grid(True)  # Add grid lines for better readability
plt.show()

#boxplot lama studi untuk tiap jalur masuk
dataframe = [kemitraan, pmdk, usm1, usm2, usm3]
labels = ['Kemitraan', 'PMDK', 'USM 1', 'USM 2', 'USM 3']
boxplot_data = [df['LamaStudi'].values for df in dataframe]

plt.figure(figsize=(8, 6))  # Adjust the figure size as per your preference

# Create the box plots for each DataFrame
plt.boxplot(boxplot_data, labels=labels)
plt.xlabel('DataFrames')
plt.ylabel('Lama Studi')
plt.title('Sebaran Lama Studi untuk tiap Jalur Masuk')
plt.grid(True)  # Add grid lines for better readability
plt.show()



















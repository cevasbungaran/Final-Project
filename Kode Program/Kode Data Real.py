# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 19:32:10 2023

@author: cevas
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#import data
df2 = pd.read_excel('Data Lulusan 2018-2023 (2).xlsx')

#drop nilai na pada kolom
df2.dropna(axis=1, how='all', inplace=True)

#drop kolom yang tidak dibutuhkan
df2.drop(df2[df2['JALUR'] == 'USM'].index, inplace=True)
df2.drop(df2[df2['JALUR'] == 'UMB'].index, inplace=True)
df2.drop(df2[df2['PROGRAM STUDI'] == 'Diploma Tiga Manajemen Perusahaan'].index, inplace=True)
df2 = df2.drop(['SEM AKD LULUS', 'SKS IPK'], axis=1)

#lowercase
df2 = df2.applymap(lambda x: x.lower() if isinstance(x, str) else x)

# Drop kolom yang memiliki provinsi dan kota dari luar indonesia
df2 = df2.drop(df2[df2['Provinsi asal SMA'] == 'provinsi amerika'].index, axis=0)
df2 = df2.drop(df2[df2['Provinsi asal SMA'] == 'provinsi china'].index, axis=0)
df2 = df2.drop(df2[df2['Provinsi asal SMA'] == 'provinsi brunei darussalam'].index, axis=0)
df2 = df2.drop(df2[df2['Provinsi asal SMA'] == 'provinsi china'].index, axis=0)
df2 = df2.drop(df2[df2['Provinsi asal SMA'] == 'provinsi colombia'].index, axis=0)
df2 = df2.drop(df2[df2['Provinsi asal SMA'] == 'provinsi japan'].index, axis=0)
df2 = df2.drop(df2[df2['Provinsi asal SMA'] == 'provinsi laos'].index, axis=0)
df2 = df2.drop(df2[df2['Provinsi asal SMA'] == 'provinsi nigeria'].index, axis=0)
df2 = df2.drop(df2[df2['Provinsi asal SMA'] == 'provinsi singapur'].index, axis=0)
df2 = df2.drop(df2[df2['Provinsi asal SMA'] == 'provinsi timor-leste'].index, axis=0)
df2 = df2.drop(df2[df2['Provinsi asal SMA'] == 'provinsi republic of korea'].index, axis=0)

#melihat rata2 dan standar deviasi
df2.mean()
df2.std()

#melihat pusat sebaran data
def central_tendency(df, kolom):
    result_df = pd.DataFrame(columns=['Rata-rata', 'Modus', 'Nilai Median'])
    
    rata2 = df[kolom].mean()
    median = df[kolom].median()
    modus = df[kolom].mode()
    
    # Menempatkan nilai-nilai central tendency ke dalam DataFrame
    result_df.loc[0] = [rata2, modus.values[0], median]
    
    return result_df
    
df2_central_ipk = central_tendency(df2, 'IPK')
df2_central_lama_studi = central_tendency(df2, 'SEMESTER TEMPUH')

df2.median()
df2.max()
df2.min()
df2['IPK'].mean()

#melihat tipe data di tiap kolom dan info lainnya
df2.info()
df2.dtypes
df2.describe()

# Mengecek apakah terdapat nilai "NA" pada kolom 'Lama_Studi'
is_na = df2['SEMESTER TEMPUH'].isna().any()  # atau df['Lama_Studi'].isnull().any()

if is_na:
    print("\nTerdapat nilai 'NA' pada kolom 'SEMESTER TEMPUH'.")
else:
    print("\nTidak terdapat nilai 'NA' pada kolom 'SEMESTER TEMPUH'.")
    
df2 = df2.dropna(subset=['SEMESTER TEMPUH'])

# Mengubah tipe data kolom 'SEMESTER TEMPUH' dari float menjadi integer
df2['SEMESTER TEMPUH'] = df2['SEMESTER TEMPUH'].astype(int)

#menyamakan nama jalur masuk seleksi khusus
df2['JALUR'] = df2['JALUR'].replace('seleksi_khusus', 'seleksi khusus')
df2['JALUR'] = df2['JALUR'].replace('jalur khusus', 'seleksi khusus')
df2['JALUR'] = df2['JALUR'].replace('khusus', 'seleksi khusus')

# Menyimpan DataFrame ke dalam file Excel
#df2.to_excel('Data Lulusan 2018-2023 Bersih.xlsx', index=False)

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

# Melihat statistik IPK dan lama studi tiap jalur masuk
# ipk
statistik_jalur_ipk = hitung_statistik_ipk(df2, 'JALUR')
statistik_thn_masuk_ipk = hitung_statistik_ipk(df2, 'THN AKD LULUS')
# lama studi
statistik_jalur_lamaStudi = hitung_statistik_lamaStudi(df2, 'JALUR')
statistik_thn_masuk_lamaStudi = hitung_statistik_lamaStudi(df2, 'THN AKD LULUS')

#mengelompokkan dataset berdasarkan jalur masuk
pmdk = df2[df2['JALUR']=='pmdk']
seleksi_khusus = df2[df2['JALUR']=='seleksi khusus']
usm1 = df2[df2['JALUR']=='usm1']
usm2 = df2[df2['JALUR']=='usm2']
usm3 = df2[df2['JALUR']=='usm3']

#boxplot ipk tiap jalur masuk
dataframe = [seleksi_khusus, pmdk, usm1, usm2, usm3]
labels = ['seleksi_khusus', 'pmdk', 'usm1', 'usm2', 'usm3']
#data boxplot
boxplot_data = [df2['IPK'].values for df in dataframe]
#boxplot_data2 = [df['Kategori IPK'].values for df in dataframe]

plt.figure(figsize=(8, 6))  # Adjust the figure size as per your preference

# Create the box plots for each DataFrame
plt.boxplot(boxplot_data, labels=labels)
plt.xlabel('Jalur Masuk')
plt.ylabel('IPK')
plt.title('Sebaran IPK Untuk Tiap Jalur Masuk')
plt.grid(True)  # Add grid lines for better readability
plt.show()

# Mendeteksi Outliers untuk IPK
Q1_ipk= df2['IPK'].quantile(0.25)
Q3_ipk = df2['IPK'].quantile(0.75)
IQR = Q3_ipk - Q1_ipk

# Menghitung Batas Atas dan Batas Bawah
upper_bound = Q3_ipk + 1.5 * IQR
lower_bound = Q1_ipk - 1.5 * IQR

# Mengidentifikasi outlier
outliers_ipk = df2[(df2['IPK'] > upper_bound) | (df2['IPK'] < lower_bound)]
print("Outliers:")
print(outliers_ipk)


#boxplot semester tempuh tiap jalur masuk
dataframe = [seleksi_khusus, pmdk, usm1, usm2, usm3]
labels = ['seleksi_khusus', 'pmdk', 'usm1', 'usm2', 'usm3']
#data boxplot
boxplot_data2 = [df2['SEMESTER TEMPUH'].values for df in dataframe]
#boxplot_data2 = [df['Kategori IPK'].values for df in dataframe]

plt.figure(figsize=(8, 6))  # Adjust the figure size as per your preference

# Create the box plots for each DataFrame
plt.boxplot(boxplot_data2, labels=labels)
plt.xlabel('Jalur Masuk')
plt.ylabel('Semester Tempuh')
plt.title('Sebaran Semester Tempuh Untuk Tiap Jalur Masuk')
plt.grid(True)  # Add grid lines for better readability
plt.show()


####################
import seaborn as sns
plt.figure(figsize=(10, 6))
sns.boxplot(x='JALUR', y='IPK', data=df2)
plt.title('Sebaran IPK berdasarkan Jalur Masuk')
plt.xlabel('Jalur Masuk')
plt.ylabel('IPK')
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(x='JALUR', y='SEMESTER TEMPUH', data=df2)
plt.title('Sebaran Semester Tempuh berdasarkan Jalur Masuk')
plt.xlabel('Semester Tempuh')
plt.ylabel('IPK')
plt.show()
####################

# Mendeteksi Outliers untuk Lama Studi
Q1_lamaStudi= df2['SEMESTER TEMPUH'].quantile(0.25)
Q3_lamaStudi = df2['SEMESTER TEMPUH'].quantile(0.75)
IQR = Q3_lamaStudi - Q1_lamaStudi

# Menghitung Batas Atas dan Batas Bawah
upper_bound = Q3_lamaStudi + 1.5 * IQR
lower_bound = Q1_lamaStudi - 1.5 * IQR

# Mengidentifikasi outlier
outliers_lamaStudi = df2[(df2['SEMESTER TEMPUH'] > upper_bound) | (df2['SEMESTER TEMPUH'] < lower_bound)]
print("Outliers:")
print(outliers_lamaStudi)

outlier_lamaStudi_jalur_count = outliers_lamaStudi['JALUR'].value_counts()
print(outlier_lamaStudi_jalur_count)

# Membuat dataframe dengan lama studi maksimal semester 14
df2_maks14 = df2[df2['SEMESTER TEMPUH'] <= 14]

# Menyimpan dataframe dengan lama studi maksimal semester 14 ke file excel
# df2_maks14.to_excel('Data Lulusan 2018-2023 Maks Smt 14.xlsx', index=False)

# Mengelompokkan dataset kembali berdasarkan jalur masuk menggunakan df_no_outlier
pmdk = df2[df2['JALUR']=='pmdk']
seleksi_khusus = df2[df2['JALUR']=='seleksi khusus']
usm1 = df2[df2['JALUR']=='usm1']
usm2 = df2[df2['JALUR']=='usm2']
usm3 = df2[df2['JALUR']=='usm3']

# Statistik tiap jalur masuk
pmdk.describe()
seleksi_khusus.describe()
usm1.describe()
usm2.describe()
usm3.describe()

#mencari index kota bandung yang ada di kolom provinsi
index_bandung = df2.index[df2['Provinsi asal SMA'].str.contains('kota bandung', case=False)].tolist()
# Menampilkan indeks yang ditemukan
print("Indeks yang memiliki kata 'Bandung' pada kolom Provinsi:", index_bandung)

#mencari index kota jakarta barat yang ada di kolom provinsi
index_jkt_barat = df2.index[df2['Provinsi asal SMA'].str.contains('kota jakarta barat', case=False)].tolist()
# Menampilkan indeks yang ditemukan
print("Indeks 'kota jakarta barat':", index_jkt_barat)

#mencari index kota jakarta selatan yang ada di kolom provinsi
index_jkt_sel = df2.index[df2['Provinsi asal SMA'].str.contains('kota jakarta selatan', case=False)].tolist()
# Menampilkan indeks yang ditemukan
print("Indeks 'kota jakarta selatan':", index_jkt_sel)

#mencari index kota semarang yang ada di kolom provinsi
index_semarang= df2.index[df2['Provinsi asal SMA'].str.contains('kota semarang', case=False)].tolist()
# Menampilkan indeks yang ditemukan
print("Indeks 'kota semarang':", index_semarang)

#mencari index kota jakarta selatan yang ada di kolom provinsi
index_tangsel = df2.index[df2['Provinsi asal SMA'].str.contains('kota tangerang selatan', case=False)].tolist()
# Menampilkan indeks yang ditemukan
print("Indeks 'kota tangerang selatan':", index_tangsel)

#mencari index kota semarang yang ada di kolom provinsi
index_tasikmalaya= df2.index[df2['Provinsi asal SMA'].str.contains('kota tasikmalaya', case=False)].tolist()
# Menampilkan indeks yang ditemukan
print("Indeks 'kota tasikmalaya':", index_tasikmalaya)

#mencari index kab bandung yang ada di kolom provinsi
index_kabbdg= df2.index[df2['Provinsi asal SMA'].str.contains('kabupaten bandung', case=False)].tolist()
# Menampilkan indeks yang ditemukan
print("Indeks 'kabupaten bandung':", index_kabbdg)

#mencari index kab bandung yang ada di kolom provinsi
index_kabcianjur= df2.index[df2['Provinsi asal SMA'].str.contains('kabupaten cianjur', case=False)].tolist()
# Menampilkan indeks yang ditemukan
print("Indeks 'kabupaten cianjur':", index_kabcianjur)

#mencari index kab bandung yang ada di kolom provinsi
index_kabtang= df2.index[df2['Provinsi asal SMA'].str.contains('kabupaten tangerang', case=False)].tolist()
# Menampilkan indeks yang ditemukan
print("Indeks 'kabupaten tangerang':", index_kabtang)

#mencari index papua (irian jaya) yang ada di kolom provinsi
index_irianJaya= df2.index[df2['Provinsi asal SMA'].str.contains('papua (irian jaya)', case=False)].tolist()
# Menampilkan indeks yang ditemukan
print("Indeks 'papua irian jaya':", index_irianJaya)

#mencari index nias barat yang ada di kolom kota
index_niasBarat= df2.index[df2['Kota asal SMA'].str.contains('nias barat', case=False)].tolist()
# Menampilkan indeks yang ditemukan
print("Indeks 'nias barat':", index_niasBarat)

#mencari index nias barat yang ada di kolom kota
index_kepRiau= df2.index[df2['Kota asal SMA'].str.contains('kepulauan riau', case=False)].tolist()
# Menampilkan indeks yang ditemukan
print("Indeks 'kepulauan riau':", index_kepRiau)

#mencari index kepulauan  riau yang ada di kolom provinsi
index_kepulauan__riau= df2.index[df2['Provinsi asal SMA'].str.contains('kepulauan  riau', case=False)].tolist()
# Menampilkan indeks yang ditemukan
print("Indeks 'kepulauan  riau':", index_kepulauan__riau)


#histogram ipk tiap jalur masuk
# histogram data besar
plt.hist(df2['IPK'], bins=20, edgecolor='black')  # You can adjust the number of bins as per your preference
plt.xlabel('IPK')
plt.ylabel('Frequency')
plt.title('Persebaran IPK Seluruh Data')
plt.show()

#seleksi khusus
plt.hist(seleksi_khusus['IPK'], bins=10, edgecolor='black')  # You can adjust the number of bins as per your preference
plt.xlabel('IPK')
plt.ylabel('Frequency')
plt.title('Persebaran IPK Jalur Seleksi Khusus')
plt.show()

#pmdk
plt.hist(pmdk['IPK'], bins=10, edgecolor='black')  # You can adjust the number of bins as per your preference
plt.xlabel('IPK')
plt.ylabel('Frequency')
plt.title('Persebaran IPK Jalur PMDK')
plt.show()

#usm 1
plt.hist(usm1['IPK'], bins=10, edgecolor='black')  # You can adjust the number of bins as per your preference
plt.ylim(0, 300)
plt.xlabel('IPK')
plt.ylabel('Frequency')
plt.title('Persebaran IPK Jalur USM 1')
plt.show()

#usm 2
plt.hist(usm2['IPK'], bins=10, edgecolor='black')  # You can adjust the number of bins as per your preference
plt.ylim(0, 300)
plt.xlabel('IPK')
plt.ylabel('Frequency')
plt.title('Persebaran IPK Jalur USM 2')
plt.show()

#usm 3
plt.hist(usm3['IPK'], bins=10, edgecolor='black')  # You can adjust the number of bins as per your preference
plt.ylim(0, 300)
plt.xlabel('IPK')
plt.ylabel('Frequency')
plt.title('Persebaran IPK Jalur USM 3')
plt.show()

print('IPK PMDK: mean=%.3f stdv=%.3f' % (pmdk['IPK'].mean(), pmdk['IPK'].std()))
print('IPK Seleksi Khusus: mean=%.3f stdv=%.3f' % (seleksi_khusus['IPK'].mean(), seleksi_khusus['IPK'].std()))
print('IPK usm1: mean=%.3f stdv=%.3f' % (usm1['IPK'].mean(), usm1['IPK'].std()))
print('IPK usm2: mean=%.3f stdv=%.3f' % (usm2['IPK'].mean(), usm2['IPK'].std()))
print('IPK usm3: mean=%.3f stdv=%.3f' % (usm3['IPK'].mean(), usm3['IPK'].std()))


#barplot semester tempuh tiap jalur masuk
#seleksi khusus
# Menghitung frekuensi lama studi
frekuensi_lama_studi_seleksiKhusus = seleksi_khusus['SEMESTER TEMPUH'].value_counts().sort_index()
# Membuat barplot
plt.figure(figsize=(8, 6)) 
bars = frekuensi_lama_studi_seleksiKhusus.plot(kind='bar', color='skyblue')
# Menetapkan batas sumbu y
#bars.axes.set_ylim(0, 300)
plt.title('Frekuensi Lama Studi Mahasiswa Jalur Seleksi Khusus')
plt.xlabel('Lama Studi')
plt.xticks(rotation=0)
plt.ylabel('Frekuensi')
for i, value in enumerate(frekuensi_lama_studi_seleksiKhusus):
    plt.text(i, value + 1, str(value), ha='center', va='bottom')
plt.show()

#pmdk
# Menghitung frekuensi lama studi
frekuensi_lama_studi_pmdk = pmdk['SEMESTER TEMPUH'].value_counts().sort_index()
# Membuat barplot
plt.figure(figsize=(8, 6)) 
bars = frekuensi_lama_studi_pmdk.plot(kind='bar', color='skyblue')
# Menetapkan batas sumbu y
#bars.axes.set_ylim(0, 300)
plt.title('Frekuensi Lama Studi Mahasiswa Jalur PMDK')
plt.xlabel('Lama Studi')
plt.xticks(rotation=0)
plt.ylabel('Frekuensi')
for i, value in enumerate(frekuensi_lama_studi_pmdk):
    plt.text(i, value + 1, str(value), ha='center', va='bottom')
plt.show()

#usm 1
# Menghitung frekuensi lama studi
frekuensi_lama_studi_usm1 = usm1['SEMESTER TEMPUH'].value_counts().sort_index()
# Membuat barplot
plt.figure(figsize=(8, 6)) 
bars = frekuensi_lama_studi_usm1.plot(kind='bar', color='skyblue')
# Menetapkan batas sumbu y
#bars.axes.set_ylim(0, 300)
plt.title('Frekuensi Lama Studi Mahasiswa Jalur USM 1')
plt.xlabel('Lama Studi')
plt.xticks(rotation=0)
plt.ylim(0, 450)
plt.ylabel('Frekuensi')
for i, value in enumerate(frekuensi_lama_studi_usm1):
    plt.text(i, value + 1, str(value), ha='center', va='bottom')
plt.show()

#usm 2
# Menghitung frekuensi lama studi
frekuensi_lama_studi_usm2 = usm2['SEMESTER TEMPUH'].value_counts().sort_index()
# Membuat barplot
plt.figure(figsize=(8, 6)) 
bars = frekuensi_lama_studi_usm2.plot(kind='bar', color='skyblue')
# Menetapkan batas sumbu y
#bars.axes.set_ylim(0, 300)
plt.title('Frekuensi Lama Studi Mahasiswa Jalur USM 2')
plt.xlabel('Lama Studi')
plt.xticks(rotation=0)
plt.ylim(0, 450)
plt.ylabel('Frekuensi')
for i, value in enumerate(frekuensi_lama_studi_usm2):
    plt.text(i, value + 1, str(value), ha='center', va='bottom')
plt.show()

#usm 3
# Menghitung frekuensi lama studi
frekuensi_lama_studi_usm3 = usm3['SEMESTER TEMPUH'].value_counts().sort_index()
# Membuat barplot
plt.figure(figsize=(8, 6)) 
bars = frekuensi_lama_studi_usm3.plot(kind='bar', color='skyblue')
# Menetapkan batas sumbu y
#bars.axes.set_ylim(0, 300)
plt.title('Frekuensi Lama Studi Mahasiswa Jalur USM 3')
plt.xlabel('Lama Studi')
plt.ylim(0, 450)
plt.xticks(rotation=0)
plt.ylabel('Frekuensi')
for i, value in enumerate(frekuensi_lama_studi_usm3):
    plt.text(i, value + 1, str(value), ha='center', va='bottom')
plt.show()


##EKSPLORASI PRODI
d3_manajemen_perusahaan = df2[df2['PROGRAM STUDI']=='diploma tiga manajemen perusahaan']
adbis = df2[df2['PROGRAM STUDI']=='sarjana administrasi bisnis']
adpub = df2[df2['PROGRAM STUDI']=='sarjana administrasi publik']
akuntansi = df2[df2['PROGRAM STUDI']=='sarjana akuntansi']
arsitektur = df2[df2['PROGRAM STUDI']=='sarjana arsitektur']
ekbang = df2[df2['PROGRAM STUDI']=='sarjana ekonomi pembangunan']
filsafat = df2[df2['PROGRAM STUDI']=='sarjana filsafat']
fisika = df2[df2['PROGRAM STUDI']=='sarjana fisika']
hi = df2[df2['PROGRAM STUDI']=='sarjana hubungan internasional']
hukum = df2[df2['PROGRAM STUDI']=='sarjana hukum']
informatika = df2[df2['PROGRAM STUDI']=='sarjana informatika']
manajemen = df2[df2['PROGRAM STUDI']=='sarjana manajemen']
mate = df2[df2['PROGRAM STUDI']=='sarjana matematika']
elektro = df2[df2['PROGRAM STUDI']=='sarjana teknik elektro']
industri = df2[df2['PROGRAM STUDI']=='sarjana teknik industri']
tekim = df2[df2['PROGRAM STUDI']=='sarjana teknik kimia']
sipil = df2[df2['PROGRAM STUDI']=='sarjana teknik sipil']

#adbis
#boxplot ipk tiap jalur masuk
dataframe = [seleksi_khusus, pmdk, usm1, usm2, usm3]
labels = ['seleksi_khusus', 'pmdk', 'usm1', 'usm2', 'usm3']
#data boxplot
boxplot_data = [adbis['IPK'].values for df in dataframe]
#boxplot_data2 = [df['Kategori IPK'].values for df in dataframe]

plt.figure(figsize=(8, 6))  # Adjust the figure size as per your preference

# Create the box plots for each DataFrame
plt.boxplot(boxplot_data, labels=labels)
plt.xlabel('Jalur Masuk')
plt.ylabel('IPK')
plt.title('Sebaran IPK Untuk Tiap Jalur Masuk Prodi Adbis')
plt.grid(True)  # Add grid lines for better readability
plt.show()

# Statistik Prodi IPK dan Lama Studi
statistik_prodi_ipk = hitung_statistik_ipk(df2, 'PROGRAM STUDI')
statistik_prodi_lamaStudi = hitung_statistik_lamaStudi(df2, 'PROGRAM STUDI')

prodi = df2['PROGRAM STUDI'].value_counts()

#barplot program studi
# Menghitung frekuensi masing-masing JalurMasuk
lulusan_prodi_counts = df2['PROGRAM STUDI'].value_counts()

# Membuat bar plot
plt.figure(figsize=(10, 10))
ay = lulusan_prodi_counts.plot(kind='bar', color='skyblue')
plt.title('Frekuensi Lulusan Berdasarkan Program Studi')
plt.xlabel('Program Studi')
plt.ylabel('Frekuensi')
plt.xticks(rotation=75)
plt.tight_layout()

for p in ay.patches:
    ay.annotate(str(p.get_height()), (p.get_x() + p.get_width() / 2, p.get_height() + 1.0), ha='center')

plt.tight_layout()
plt.show()

#jumlah tiap program studi
prodi = df2['PROGRAM STUDI'].value_counts()
prodi.describe()


##EKSPLORASI PROVINSI DAN KOTA

#kota
hasil_kota_lamaStudi = hitung_statistik_lamaStudi(df2, 'Kota asal SMA')
hasil_kota_ipk = hitung_statistik_ipk(df2, 'Kota asal SMA')
kota = df2['Kota asal SMA'].value_counts()



hasil_kota_lamaStudi.to_excel('Hasil Kota Lama Studi.xlsx')
hasil_kota_ipk.to_excel('Hasil Kota IPK.xlsx')

jumlah_kota = kota.index.str.contains("kota", case=False).sum()
jumlah_kabupaten = kota.index.str.contains("kabupaten", case=False).sum()
# Menampilkan hasil
print(f"Jumlah kota: {jumlah_kota}")
print(f"Jumlah kabupaten: {jumlah_kabupaten}")


#provinsi
hasil_provinsi_lamaStudi = hitung_statistik_lamaStudi(df2, 'Provinsi asal SMA')
hasil_provinsi_ipk = hitung_statistik_ipk(df2, 'Provinsi asal SMA')

#mengetahui jumlah provinsi
provinsi = df2['Provinsi asal SMA'].value_counts()

# Membuat bar plot provinsi
plt.figure(figsize=(10, 10))
ay = provinsi.plot(kind='bar', color='skyblue')
plt.title('Frekuensi Lulusan Berdasarkan Provinsi Asal SMA')
plt.xlabel('Provinsi Asal SMA')
plt.ylabel('Frekuensi')
plt.xticks(rotation=75)
plt.tight_layout()

for p in ay.patches:
    ay.annotate(str(p.get_height()), (p.get_x() + p.get_width() / 2, p.get_height() + 1.0), ha='center')

plt.tight_layout()
plt.show()



# Membuat scatter plot
avg_lama_studi = df2.groupby('IPK')['SEMESTER TEMPUH'].mean().reset_index()
avg_lama_studi_pmdk = pmdk.groupby('IPK')['SEMESTER TEMPUH'].mean().reset_index()
avg_lama_studi_seleksiKhusus = seleksi_khusus.groupby('IPK')['SEMESTER TEMPUH'].mean().reset_index()
avg_lama_studi_usm1 = usm1.groupby('IPK')['SEMESTER TEMPUH'].mean().reset_index()
avg_lama_studi_usm2 = usm2.groupby('IPK')['SEMESTER TEMPUH'].mean().reset_index()
avg_lama_studi_usm3 = usm3.groupby('IPK')['SEMESTER TEMPUH'].mean().reset_index()

#scatter plot data besar
plt.scatter(avg_lama_studi['IPK'], avg_lama_studi['SEMESTER TEMPUH']) # plotting the clusters
plt.title('Pola Antara IPK dengan Semester Tempuh')
plt.xlabel("IPK") # X-axis label
plt.ylabel("Semester Tempuh") # Y-axis label
plt.show() 

#scatter plot pmdk
plt.scatter(avg_lama_studi_pmdk['IPK'], avg_lama_studi_pmdk['SEMESTER TEMPUH']) # plotting the clusters
plt.title('Pola Antara IPK dengan Semester Tempuh Jalur PMDK')
plt.xlabel("IPK") # X-axis label
plt.ylabel("Semester Tempuh") # Y-axis label
plt.show() 

#scatter plot kemitraan
plt.scatter(avg_lama_studi_seleksiKhusus['IPK'], avg_lama_studi_seleksiKhusus['SEMESTER TEMPUH']) # plotting the clusters
plt.title('Pola Antara IPK dengan LamaStudi Jalur Seleksi Khusus')
plt.xlabel("IPK") # X-axis label
plt.ylabel("Semester Tempuh") # Y-axis label
plt.show() 

#scatter plot usm 1
plt.scatter(avg_lama_studi_usm1['IPK'], avg_lama_studi_usm1['SEMESTER TEMPUH']) # plotting the clusters
plt.title('Pola Antara IPK dengan Semester Tempuh Jalur USM 1')
plt.xlabel("IPK") # X-axis label
plt.ylabel("Semester Tempuh") # Y-axis label
plt.show() 

#scatter plot usm 2
plt.scatter(avg_lama_studi_usm2['IPK'], avg_lama_studi_usm2['SEMESTER TEMPUH']) # plotting the clusters
plt.title('Pola Antara IPK dengan Semester Tempuh Jalur USM 2')
plt.xlabel("IPK") # X-axis label
plt.ylabel("Semester Tempuh") # Y-axis label
plt.show() 

#scatter plot usm 3
plt.scatter(avg_lama_studi_usm3['IPK'], avg_lama_studi_usm3['SEMESTER TEMPUH']) # plotting the clusters
plt.title('Pola Antara IPK dengan Semester Tempuh Jalur USM 3')
plt.xlabel("IPK") # X-axis label
plt.ylabel("Semester Tempuh") # Y-axis label
plt.show() 


## KORELASI ANTAR VARIABEL
# Membuat df baru untuk kolom IPK dan SEMESTER TEMPUH
df2_ipk = df2['IPK']
df2_lama_studi = df2['SEMESTER TEMPUH']

# Uji Korelasi pearson
korelasi_pearson = df2_ipk.corr(df2_lama_studi, method='pearson')
print(f"Korelasi antara 'IPK' dan 'Lama_Studi': {korelasi_pearson}")


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
    print('Korelasi antara ', atribut1, atribut2)
    if abs(stat) >= critical:
     print('Dependent (reject H0)')
    else:
     print('Independent (fail to reject H0)')
     

## Uji chi square dengan alpha=0.5
# provinsi
chiSquare_provinsi_lamaStudi_05 = chi_square(df2, 'Provinsi asal SMA', 'SEMESTER TEMPUH', 0.5) #dependent
chiSquare_provinsi_ipk_05 = chi_square(df2, 'Provinsi asal SMA', 'IPK', 0.5) #independent
# prodi
chiSquare_prodi_lamaStudi_05 = chi_square(df2, 'PROGRAM STUDI', 'SEMESTER TEMPUH', 0.5) #dependent
chiSquare_prodi_ipk_05 = chi_square(df2, 'PROGRAM STUDI', 'IPK', 0.5) #dependent
# jalur masuk
chiSquare_jalurMasuk_lamaStudi_05 = chi_square(df2, 'JALUR', 'SEMESTER TEMPUH', 0.5) #dependent
chiSquare_jalurMasuk_ipk_05 = chi_square(df2, 'JALUR', 'IPK', 0.5) #dependent

## Uji chi square dengan alpha=0.01
# provinsi
chiSquare_provinsi_lamaStudi_001 = chi_square(df2, 'Provinsi asal SMA', 'SEMESTER TEMPUH', 0.01) #dependent
chiSquare_provinsi_ipk_001 = chi_square(df2, 'Provinsi asal SMA', 'IPK', 0.01) #independent
# prodi
chiSquare_prodi_lamaStudi_001 = chi_square(df2, 'PROGRAM STUDI', 'SEMESTER TEMPUH', 0.01) #dependent
chiSquare_prodi_ipk_001 = chi_square(df2, 'PROGRAM STUDI', 'IPK', 0.01) #dependent
# jalur masuk
chiSquare_jalurMasuk_lamaStudi_001 = chi_square(df2, 'JALUR', 'SEMESTER TEMPUH', 0.01) #dependent
chiSquare_jalurMasuk_ipk_001 = chi_square(df2, 'JALUR', 'IPK', 0.01) #dependent

## Uji chi square dengan alpha=0.025
# provinsi
chiSquare_provinsi_lamaStudi_0025 = chi_square(df2, 'Provinsi asal SMA', 'SEMESTER TEMPUH', 0.025) #dependent
chiSquare_provinsi_ipk_0025 = chi_square(df2, 'Provinsi asal SMA', 'IPK', 0.025) #independent
# prodi
chiSquare_prodi_lamaStudi_0025 = chi_square(df2, 'PROGRAM STUDI', 'SEMESTER TEMPUH', 0.025) #dependent
chiSquare_prodi_ipk_0025 = chi_square(df2, 'PROGRAM STUDI', 'IPK', 0.025) #dependent
# jalur masuk
chiSquare_jalurMasuk_lamaStudi_0025 = chi_square(df2, 'JALUR', 'SEMESTER TEMPUH', 0.025) #dependent
chiSquare_jalurMasuk_ipk_0025 = chi_square(df2, 'JALUR', 'IPK', 0.025) #dependent

## Uji chi square dengan alpha=0.1
# provinsi
chiSquare_provinsi_lamaStudi_01 = chi_square(df2, 'Provinsi asal SMA', 'SEMESTER TEMPUH', 0.1) #dependent
chiSquare_provinsi_ipk_01 = chi_square(df2, 'Provinsi asal SMA', 'IPK', 0.1) #independent
# prodi
chiSquare_prodi_lamaStudi_01 = chi_square(df2, 'PROGRAM STUDI', 'SEMESTER TEMPUH', 0.1) #dependent
chiSquare_prodi_ipk_01 = chi_square(df2, 'PROGRAM STUDI', 'IPK', 0.1) #dependent
# jalur masuk
chiSquare_jalurMasuk_lamaStudi_01 = chi_square(df2, 'JALUR', 'SEMESTER TEMPUH', 0.1) #dependent
chiSquare_jalurMasuk_ipk_01= chi_square(df2, 'JALUR', 'IPK', 0.1) #dependent




## EKSPLORASI LAMA STUDI
# Menghitung frekuensi lama studi
frekuensi_lama_studi = df2['SEMESTER TEMPUH'].value_counts().sort_index()
# Membuat barplot
plt.figure(figsize=(8, 6)) 
bars = frekuensi_lama_studi.plot(kind='bar', color='skyblue')
# Menetapkan batas sumbu y
#bars.axes.set_ylim(0, 300)
plt.title('Frekuensi Lama Studi Mahasiswa')
plt.xlabel('Lama Studi')
plt.xticks(rotation=0)
plt.ylabel('Frekuensi')
for i, value in enumerate(frekuensi_lama_studi):
    plt.text(i, value + 1, str(value), ha='center', va='bottom')
plt.show()

df_lain = pd.read_excel('Data Lulusan 2018-2023 Maks Smt 14.xlsx')
frekuensi_lama_studi = df_lain['SEMESTER TEMPUH'].value_counts().sort_index()
# Membuat barplot
plt.figure(figsize=(8, 6)) 
bars = frekuensi_lama_studi.plot(kind='bar', color='skyblue')
# Menetapkan batas sumbu y
#bars.axes.set_ylim(0, 300)
plt.title('Frekuensi Lama Studi Mahasiswa')
plt.xlabel('Lama Studi')
plt.xticks(rotation=0)
plt.ylabel('Frekuensi')
for i, value in enumerate(frekuensi_lama_studi):
    plt.text(i, value + 1, str(value), ha='center', va='bottom')
plt.show()


















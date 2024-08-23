# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 20:35:29 2023

@author: cevas
"""

import random
from faker import Faker
import pandas as pd

fake = Faker()

def generate_ipk(min_value=2.0, max_value=4.0):
    return round(random.uniform(min_value, max_value), 2)

def generate_jalurMsk():
    jalur = ['PMDK', 'Kemitraan', 'USM 1', 'USM 2', 'USM 3']
    jlr = random.choice(jalur)
    return jlr
# Contoh penggunaan

ipk = generate_ipk()
print(ipk)

jalurMasuk = generate_jalurMsk()
print(jalurMasuk)


dataMhs = []
for i in range(1000):
    nama = fake.name()
    IPK = generate_ipk()
    lamaStudi = random.randint(8, 14)
    jalurMasuk = generate_jalurMsk()
    dataMhs.append([nama, IPK, lamaStudi, jalurMasuk])
df_dataMhs = pd.DataFrame(dataMhs, columns=['Nama', 'IPK', 'LamaStudi', 'JalurMasuk'])
df_dataMhs.to_csv('Data Mahasiswa Tambahan.csv', index=False)
df_dataMhs.info()


dataMhs2 = []
for i in range(1000):
    nama = fake.name()
    IPK = generate_ipk()
    lamaStudi = random.randint(8, 14)
    dataMhs2.append([nama, IPK, lamaStudi])
df_dataMhs2 = pd.DataFrame(dataMhs2, columns=['Nama', 'IPK', 'LamaStudi'])
df_dataMhs2.to_csv('Data Mahasiswa 2.csv', index=False)
df_dataMhs2.info()

########################################
#DATA BARU
def generate_ipk1(min_value=2.0, max_value=2.5):
    return round(random.uniform(min_value, max_value), 2)

def generate_ipk2(min_value=2.5, max_value=3.0):
    return round(random.uniform(min_value, max_value), 2)

def generate_ipk3(min_value=3.0, max_value=3.5):
    return round(random.uniform(min_value, max_value), 2)

def generate_ipk4(min_value=3.5, max_value=4.0):
    return round(random.uniform(min_value, max_value), 2)

ipk1 = generate_ipk1()
print(ipk1)

data = []
for i in range(250):
    nama = fake.name()
    ipk1 = generate_ipk1()
    lamaStudi = random.randint(8,14)
    jalurMasuk = generate_jalurMsk()
    data.append((nama, ipk1, lamaStudi, jalurMasuk))
for i in range(250):
    nama = fake.name()
    ipk1 = generate_ipk2()
    lamaStudi = random.randint(8,14)
    jalurMasuk = generate_jalurMsk()
    data.append((nama, ipk1, lamaStudi, jalurMasuk))
for i in range(250):
    nama = fake.name()
    ipk1 = generate_ipk3()
    lamaStudi = random.randint(8,14)
    jalurMasuk = generate_jalurMsk()
    data.append((nama, ipk1, lamaStudi, jalurMasuk))
for i in range(250):
    nama = fake.name()
    ipk1 = generate_ipk4()
    lamaStudi = random.randint(8,14)
    jalurMasuk = generate_jalurMsk()
    data.append((nama, ipk1, lamaStudi, jalurMasuk))
df_dataMhs4 = pd.DataFrame(data, columns=['Nama', 'IPK', 'LamaStudi', 'JalurMasuk'])
df_dataMhs4.to_csv('Data Mahasiswa Baru.csv', index=False)
df_dataMhs4.info()


def generate_jalurMsk():
    jalur = ['PMDK', 'Kemitraan', 'USM 1', 'USM 2', 'USM 3']
    jlr = random.choice(jalur)
    return jlr


def generate_data(n):
    names = fake.name()
    jalur_masuk = ["PMDK", "Kemitraan", "USM 1", "USM 2", "USM 3"]
    
    data = []
    per_group = n // len(names)
    
    for _ in range(n):
        nama = fake.name()
        
        # Generate balanced IPK values
        group_index = _ // per_group
        if group_index == 0:
            ipk = round(random.uniform(2.00, 2.50), 2)
        elif group_index == 1:
            ipk = round(random.uniform(2.50, 3.00), 2)
        elif group_index == 2:
            ipk = round(random.uniform(3.00, 3.50), 2)
        else:
            ipk = round(random.uniform(3.50, 4.00), 2)
        
        # Generate balanced LamaStudi values
        if group_index == 0:
            lama_studi = random.randint(8, 9)
        elif group_index == 1:
            lama_studi = random.randint(10, 11)
        elif group_index == 2:
            lama_studi = random.randint(12, 13, 14)
        # else:
        #     lama_studi = random.randint(13, 14)
        
        jalur = random.choice(jalur_masuk)
        data.append((nama, ipk, lama_studi, jalur))
    
    return data

def save_to_csv(data):
    with open("data_mahasiswa.csv", "w") as file:
        file.write("Nama,IPK,LamaStudi,JalurMasuk\n")
        for item in data:
            file.write(f"{item[0]},{item[1]},{item[2]},{item[3]}\n")

data = generate_data(1000)
df_dataMhs3 = pd.DataFrame(data, columns=['Nama', 'IPK', 'LamaStudi', 'JalurMasuk'])
df_dataMhs2.to_csv('Data Mahasiswa 2.csv', index=False)
df_dataMhs2.info()

save_to_csv(data)












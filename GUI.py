# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 16:36:11 2023

@author: cevas
"""
import os
import sys
import tkinter as tk
import pandas as pd
import pickle
from tkinter import ttk
from tkinter import ttk, messagebox
import numpy as np
from ttkthemes import ThemedStyle

def get_model_path(filename):
    return os.path.join(os.path.dirname(__file__), filename)
    
class GUIApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Aplikasi Prediksi")
        self.root.config(bg="skyblue")
        
        # Load model prediksi IPK Dtree
        try:
            model_ipk_dt_path = get_model_path('DT_model_ipk.pkl')
            with open(model_ipk_dt_path, 'rb') as file_ipk_dt:
                self.model_ipk_dt = pickle.load(file_ipk_dt)
        except FileNotFoundError:
            messagebox.showerror("Error", "File DT_model_ipk.pkl tidak ditemukan.")
        
        # Load model prediksi IPK NBC
        try:
            model_ipk_nbc_path = get_model_path('nbc_modelIPK.pkl')
            with open(model_ipk_nbc_path, 'rb') as file_ipk_nbc:
                self.model_ipk_nbc = pickle.load(file_ipk_nbc)
        except FileNotFoundError:
            messagebox.showerror("Error", "File nbc_modelIPK.pkl tidak ditemukan.")
        
        # Load model prediksi Lama Studi NBC
        try:
            model_lama_studi_nbc_path = get_model_path('nbc_model_lamaStudi.pkl')
            with open(model_lama_studi_nbc_path, 'rb') as file_lama_studi_nbc:
                self.model_lama_studi_nbc = pickle.load(file_lama_studi_nbc)
        except FileNotFoundError:
            messagebox.showerror("Error", "File nbc_model_lamaStudi.pkl tidak ditemukan.")
        
        # Load model prediksi Lama Studi Dtree
        try:
            model_lama_studi_dt_path = get_model_path('DT_model_lamaStudi.pkl')
            with open(model_lama_studi_dt_path, 'rb') as file_lama_studi_dt:
                self.model_lama_studi_dt = pickle.load(file_lama_studi_dt)
        except FileNotFoundError:
            messagebox.showerror("Error", "File DT_model_lamaStudi.pkl tidak ditemukan.")

        
        # pake notebook tkinter
        self.notebook = ttk.Notebook(root, width=1000, height=900)
        self.notebook.pack(padx=30, pady=30, expand=True)
        style = ThemedStyle(self.notebook)
        style.set_theme("breeze")
        
        # Tab Landing Page
        self.landing_page = ttk.Frame(self.notebook)
        self.notebook.add(self.landing_page, text="Landing Tab")
        self.create_scrollable_landing_page()
        
        # Tab Prediksi IPK
        self.page_ipk = ttk.Frame(self.notebook)
        self.notebook.add(self.page_ipk, text="Prediksi IPK")
        
        # Element GUI untuk prediksi IPK
        self.label_text_ipk = ttk.Label(self.page_ipk, text=
                                 "PREDIKSI IPK")
        self.label_text_ipk.config(font=('arial', 14, 'bold'))
        self.label_text_ipk.pack(pady=10)
        
        # Pilih Algoritma
        self.label_algoritma_ipk = ttk.Label(self.page_ipk, text="Pilih Algoritma:")
        self.label_algoritma_ipk.config(font=('arial', 12))
        self.label_algoritma_ipk.pack(pady=10)
        self.entry_algoritma_ipk = ttk.Combobox(self.page_ipk, values=[
            'Naive Bayes', 'Decision Tree'],
            state='readonly', font=('arial', 11))
        self.entry_algoritma_ipk.config(font=('arial', 12))
        self.entry_algoritma_ipk.pack(pady=5)
        
        # Lama Studi
        self.label_lama_studi_ipk = ttk.Label(self.page_ipk, text="Lama Studi:")
        self.label_lama_studi_ipk.config(font=('arial', 12))
        self.label_lama_studi_ipk.pack(pady=10)
        self.entry_lama_studi_ipk = ttk.Combobox(self.page_ipk, values=[
            '7', '8', '9', '10', '11', '12', '13', '14'],
            state='readonly')
        self.entry_lama_studi_ipk.config(font=('arial', 12))
        self.entry_lama_studi_ipk.pack(pady=5)
        
        # Program Studi
        self.label_program_studi_ipk = ttk.Label(self.page_ipk, text="Program Studi:")
        self.label_program_studi_ipk.config(font=('arial', 12))
        self.label_program_studi_ipk.pack()
        self.entry_program_studi_ipk = ttk.Combobox(self.page_ipk, values=[
            'Sarjana Administrasi Bisnis',
            'Sarjana Administrasi Publik',
            'Sarjana Akuntansi',
            'Sarjana Arsitektur',
            'Sarjana Ekonomi Pembangunan',
            'Sarjana Filsafat',
            'Sarjana Fisika',
            'Sarjana Hubungan Internasional',
            'Sarjana Hukum',
            'Sarjana Informatika',
            'Sarjana Manajemen',
            'Sarjana Matematika',
            'Sarjana Teknik Elektro',
            'Sarjana Teknik Industri',
            'Sarjana Teknik Kimia',
            'Sarjana Teknik Sipil'
            ], 
            width=30,
            state='readonly')
        self.entry_program_studi_ipk.config(font=('arial', 12))
        self.entry_program_studi_ipk.pack(pady=6)
        
        # Jalur Masuk
        self.label_jalur_masuk_ipk = ttk.Label(self.page_ipk, text="Jalur Masuk:")
        self.label_jalur_masuk_ipk.config(font=('arial', 12))
        self.label_jalur_masuk_ipk.pack()
        self.entry_jalur_masuk_ipk = ttk.Combobox(self.page_ipk, values=[
            'PMDK',
            'Seleksi Khusus',
            'USM1',
            'USM2',
            'USM3'
            ],
            state='readonly')
        self.entry_jalur_masuk_ipk.config(font=('arial', 12))
        self.entry_jalur_masuk_ipk.pack(pady=5)
        
        # Button Prediksi IPK
        self.button_prediksi_ipk = ttk.Button(self.page_ipk, text="Prediksi IPK", command=self.prediksi_ipk)
        self.button_prediksi_ipk.pack(pady=10)
        
        # Hasil Prediksi IPK
        self.label_hasil_ipk = ttk.Label(self.page_ipk, text="Hasil Prediksi IPK:")
        self.label_hasil_ipk.config(font=('arial', 12))
        self.label_hasil_ipk.pack()
        self.output_string_ipk = tk.StringVar()
        self.output_text_ipk = ttk.Label(self.page_ipk, textvariable=self.output_string_ipk)
        self.output_text_ipk.config(font=('arial', 13, 'bold'))
        self.output_text_ipk.pack()
        
        # Hasil MSE Prediksi IPK
        self.output_mse_ipk = tk.StringVar()
        self.output_mse_ipk = ttk.Label(self.page_ipk, textvariable=self.output_mse_ipk)
        self.output_mse_ipk.config(font=('arial', 13, 'bold'))
        self.output_mse_ipk.pack()



        # Tab Prediksi Lama Studi
        self.page_lama_studi = ttk.Frame(self.notebook)
        self.notebook.add(self.page_lama_studi, text="Prediksi Lama Studi")

        # Element GUI untuk prediksi Lama Studi
        self.label_text_lamaStudi = ttk.Label(self.page_lama_studi, text=
                                 "PREDIKSI LAMA STUDI")
        self.label_text_lamaStudi.config(font=('arial', 14, 'bold'))
        self.label_text_lamaStudi.pack(pady=10)
        
        # Pilih Algoritma
        self.label_algoritma_lamaStudi = ttk.Label(self.page_lama_studi, text="Pilih Algoritma:")
        self.label_algoritma_lamaStudi.config(font=('arial', 12))
        self.label_algoritma_lamaStudi.pack(pady=10)
        self.entry_algoritma_lamaStudi = ttk.Combobox(self.page_lama_studi, values=[
            'Naive Bayes', 'Decision Tree'],
            state='readonly')
        self.entry_algoritma_lamaStudi.config(font=('arial', 12))
        self.entry_algoritma_lamaStudi.pack(pady=5)
        
        # Program Studi
        self.label_program_studi_lama_studi = ttk.Label(self.page_lama_studi, text="Program Studi:")
        self.label_program_studi_lama_studi.config(font=('arial', 12))
        self.label_program_studi_lama_studi.pack()
        self.entry_program_studi_lama_studi = ttk.Combobox(self.page_lama_studi, values=[
            'Sarjana Administrasi Bisnis',
            'Sarjana Administrasi Publik',
            'Sarjana Akuntansi',
            'Sarjana Arsitektur',
            'Sarjana Ekonomi Pembangunan',
            'Sarjana Filsafat',
            'Sarjana Fisika',
            'Sarjana Hubungan Internasional',
            'Sarjana Hukum',
            'Sarjana Informatika',
            'Sarjana Manajemen',
            'Sarjana Matematika',
            'Sarjana Teknik Elektro',
            'Sarjana Teknik Industri',
            'Sarjana Teknik Kimia',
            'Sarjana Teknik Sipil'
            ], 
            width=30,
            state='readonly')
        self.entry_program_studi_lama_studi.config(font=('arial', 12))
        self.entry_program_studi_lama_studi.pack(pady=5)
        
        # IPK
        self.label_ipk_lama_studi = ttk.Label(self.page_lama_studi, text="IPK:")
        self.label_ipk_lama_studi.config(font=('arial', 12))
        self.label_ipk_lama_studi.pack(pady=10)
        self.entry_ipk_lama_studi = ttk.Entry(self.page_lama_studi)
        self.entry_ipk_lama_studi.config(font=('arial', 12))
        self.entry_ipk_lama_studi.pack(pady=5)
        
        self.label_keterangan_ipk = ttk.Label(self.page_lama_studi, text=
                                 "Masukkan nilai IPK dalam skala 2.00-4.00:")
        self.label_keterangan_ipk.config(font=('arial', 11))
        self.label_keterangan_ipk.pack(pady=5)
        
        # Jalur Masuk
        self.label_jalur_masuk_lama_studi = ttk.Label(self.page_lama_studi, text="Jalur Masuk:")
        self.label_jalur_masuk_lama_studi.config(font=('arial', 12))
        self.label_jalur_masuk_lama_studi.pack()
        self.entry_jalur_masuk_lama_studi = ttk.Combobox(self.page_lama_studi, values=[
            'PMDK',
            'Seleksi Khusus',
            'USM1',
            'USM2',
            'USM3'
            ],
            state='readonly')
        self.entry_jalur_masuk_lama_studi.config(font=('arial', 12))
        self.entry_jalur_masuk_lama_studi.pack(pady=5)
        
        # Button Prediksi Lama Studi
        self.button_prediksi_lama_studi = ttk.Button(self.page_lama_studi, text="Prediksi Lama Studi", command=self.prediksi_lama_studi)
        self.button_prediksi_lama_studi.pack(pady=10)
        
        # Hasil Prediksi Lama Studi
        self.label_hasil_lama_studi = ttk.Label(self.page_lama_studi, text="Hasil Prediksi Lama Studi:")
        self.label_hasil_lama_studi.config(font=('arial', 12))
        self.label_hasil_lama_studi.pack()
        self.output_string_lama_studi = tk.StringVar()
        self.output_text_lama_studi = ttk.Label(self.page_lama_studi, textvariable=self.output_string_lama_studi)
        self.output_text_lama_studi.config(font=('arial', 13, 'bold'))
        self.output_text_lama_studi.pack()
        
        # Hasil MSE Lama Studi
        self.output_mse_lama_studi = tk.StringVar()
        self.output_mse_lama_studi = ttk.Label(self.page_lama_studi, textvariable=self.output_mse_lama_studi)
        self.output_mse_lama_studi.config(font=('arial', 13, 'bold'))
        self.output_mse_lama_studi.pack()
        
        # Button Reset Input IPK
        self.button_reset_ipk = ttk.Button(self.page_ipk, text="Reset", command=self.ipk_reset_inputs)
        self.button_reset_ipk.pack(pady=10)
        
        # Button Reset Input Lama Studi
        self.button_reset_lama_studi = ttk.Button(self.page_lama_studi, text="Reset", command=self.lama_studi_reset_input)
        self.button_reset_lama_studi.pack(pady=10)
    
    def create_scrollable_landing_page(self):
        # Membuat Canvas
        canvas = tk.Canvas(self.landing_page)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Membuat Scrollbar
        scrollbar = ttk.Scrollbar(self.landing_page, orient="vertical", command=canvas.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Menghubungkan Scrollbar dengan Canvas
        canvas.configure(yscrollcommand=scrollbar.set)

        # Membuat Frame dalam Canvas untuk menampung semua widget
        scrollable_frame = ttk.Frame(canvas)
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="n")
        
        # Fungsi untuk memperbarui lebar frame
        def configure_frame(event):
            canvas_width = event.width
            scrollable_frame.config(width=canvas_width)

        # Mengikat event konfigurasi dengan fungsi
        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", configure_frame)
        
        # Menambahkan konten ke dalam scrollable_frame
        self.label_text_judul = ttk.Label(scrollable_frame, text="Aplikasi GUI Prediksi IPK dan Lama Studi")
        self.label_text_judul.config(font=('arial', 14, 'bold'))
        self.label_text_judul.pack(pady=25)

        self.label_text_dataset = ttk.Label(scrollable_frame, text="Deskripsi Dataset")
        self.label_text_dataset.config(font=('arial', 13, 'bold'))
        self.label_text_dataset.pack(pady=10)

        self.label_text_deskripsi_dataset = ttk.Label(
            scrollable_frame, 
            text="Dataset yang digunakan sudah dibersihkan terlebih dahulu dan sudah dilakukan ekstraksi fitur. \nTerdiri dari 6114 baris dan 4 kolom. Kolom yang digunakan adalah: \n1. Program Studi \n2. IPK \n3. Semester Tempuh \n4. Jalur Masuk"
        )
        self.label_text_deskripsi_dataset.config(font=('arial', 12))
        self.label_text_deskripsi_dataset.pack(pady=10)
        
        #Rata-rata IPK dan Lama Studi
        self.label_rata2_ipk_lama_studi = ttk.Label(scrollable_frame, text="Rata-rata IPK dan Lama Studi")
        self.label_rata2_ipk_lama_studi.config(font=('arial', 13, 'bold'))
        self.label_rata2_ipk_lama_studi.pack(pady=10)
        self.label_text_rata2 = ttk.Label(scrollable_frame, text="Rata-rata dari IPK dan Lama Studi untuk seluruh data adalah:")
        self.label_text_rata2.config(font=('arial', 12))
        self.label_text_rata2.pack(pady=10)
        
        self.rata2_ipk_lama_studi(scrollable_frame)
        
        #Rata-rata IPK dan Lama Studi Berdasarkan Jalur Masuk
        self.label_rata2_ipk_lama_studi_jalur_masuk = ttk.Label(scrollable_frame, text="Rata-rata IPK dan Lama Studi Berdasarkan Jalur Masuk")
        self.label_rata2_ipk_lama_studi_jalur_masuk.config(font=('arial', 13, 'bold'))
        self.label_rata2_ipk_lama_studi_jalur_masuk.pack(pady=10)
        self.label_text_rata2_ipk_lama_studi_jalur_masuk = ttk.Label(scrollable_frame, text="Rata-rata dari IPK dan Lama Studi berdasarkan Jalur Masuk adalah:")
        self.label_text_rata2_ipk_lama_studi_jalur_masuk.config(font=('arial', 12))
        self.label_text_rata2_ipk_lama_studi_jalur_masuk.pack(pady=10)

        self.rata2_ipk_lama_studi_jalur_masuk(scrollable_frame)
        
        #Rata-rata IPK dan Lama Studi Berdasarkan Program Studi
        self.label_rata2_ipk_lama_studi_prodi = ttk.Label(scrollable_frame, text="Rata-rata IPK dan Lama Studi Berdasarkan Program Studi")
        self.label_rata2_ipk_lama_studi_prodi.config(font=('arial', 13, 'bold'))
        self.label_rata2_ipk_lama_studi_prodi.pack(pady=10)
        self.label_text_rata2_ipk_lama_studi_prodi = ttk.Label(scrollable_frame, text="Rata-rata dari IPK dan Lama Studi berdasarkan Program Studi adalah:")
        self.label_text_rata2_ipk_lama_studi_prodi.config(font=('arial', 12))
        self.label_text_rata2_ipk_lama_studi_prodi.pack(pady=10)

        self.rata2_ipk_lama_studi_prodi(scrollable_frame)

        # Mengatur ulang scrollregion setelah semua widget ditambahkan
        scrollable_frame.update_idletasks()
        canvas.configure(scrollregion=canvas.bbox("all"))
        
    
    def rata2_ipk_lama_studi(self, parent):
        # Contoh data rata-rata IPK
        data = [
            ("IPK", 3.136),
            ("Semester Tempuh", 8.89)
        ]

        # Membuat Treeview untuk tabel
        columns = ("Kolom", "Rata-rata")
        tree = ttk.Treeview(parent, columns=columns, show='headings', height=1)
        tree.heading("Kolom", text="Kolom")
        tree.heading("Rata-rata", text="Rata-rata")

        # Menambahkan data ke tabel
        for item in data:
            tree.insert('', 'end', values=item)

        # Menempatkan Treeview pada frame
        tree.pack(pady=10)
    
    def rata2_ipk_lama_studi_jalur_masuk(self, parent):
        # Contoh data rata-rata IPK
        data = [
            ("Seleksi Khusus", 3.378, 7.901, 273),
            ("PMDK", 3.247, 8.426, 2994),
            ("USM 3", 3.023, 9.5 , 745),
            ("USM 1", 2.997, 9.315, 1263),
            ("USM 2", 2.98, 9.681, 869)
        ]

        # Membuat Treeview untuk tabel
        columns = ("Jalur Masuk", "Rata-rata IPK", "Rata-rata Lama Studi", "Total Lulusan")
        tree = ttk.Treeview(parent, columns=columns, show='headings', height=4)
        tree.heading("Jalur Masuk", text="Jalur Masuk")
        tree.heading("Rata-rata IPK", text="Rata-rata IPK")
        tree.heading("Rata-rata Lama Studi", text="Rata-rata Lama Studi")
        tree.heading("Total Lulusan", text="Total Lulusan")
        
        # Menambahkan data ke tabel
        for item in data:
            tree.insert('', 'end', values=item)

        # Menempatkan Treeview pada frame
        tree.pack(pady=10)
    
    def rata2_ipk_lama_studi_prodi(self, parent):
        # Contoh data rata-rata IPK
        data = [
            ("Filsafat", 3.532, 8.176, 34),
            ("Hubungan Internasional", 3.445, 8.200, 624),
            ("Teknik Industri", 3.441, 7.9 , 111),
            ("Matematika", 3.262 , 8.312, 147),
            ("Teknik Sipil", 3.226, 8.550, 1013),
            ("Teknik Kimia", 3.183, 9.003, 327),
            ("Akuntansi", 3.179, 8.522, 637),
            ("Teknik Elektro", 3.117, 10.361, 47),
            ("Ekonomi Pembangunan", 3.107, 9.355, 163),
            ("Arsitektur", 3.081, 9.009, 667),
            ("Manajemen", 3.06, 8.991, 590),
            ("Fisika", 3.031, 9.666, 18),
            ("Administrasi Publik", 3.029, 9.850, 261),
            ("Informatika", 3.002, 9.965, 201),
            ("Administrasi Bisnis", 2.994, 8.901, 571),
            ("Hukum", 2.909, 9.461, 733)
        ]

        # Membuat Treeview untuk tabel
        columns = ("Program Studi", "Rata-rata IPK", "Rata-rata Lama Studi", "Total Lulusan")
        tree = ttk.Treeview(parent, columns=columns, show='headings', height=15)
        tree.heading("Program Studi", text="Program Studi")
        tree.heading("Rata-rata IPK", text="Rata-rata IPK")
        tree.heading("Rata-rata Lama Studi", text="Rata-rata Lama Studi")
        tree.heading("Total Lulusan", text="Total Lulusan")
        
        # Menambahkan data ke tabel
        for item in data:
            tree.insert('', 'end', values=item)

        # Menempatkan Treeview pada frame
        tree.pack(pady=10)
    
    def prediksi_ipk(self):
        # Mengambil nilai dari dropdown button dan menjadikan array
        lama_studi = np.array(self.entry_lama_studi_ipk.get())
        program_studi = np.array(self.entry_program_studi_ipk.get())
        jalur_masuk = np.array(self.entry_jalur_masuk_ipk.get())
        algoritma = np.array(self.entry_algoritma_ipk.get())
        
        # Exception semua input harus terisi
        try:
            if not lama_studi or not program_studi or not jalur_masuk or not algoritma:
                raise ValueError("Semua input harus terisi")
            
        except ValueError as ve:
            messagebox.showerror("Error", str(ve))
            return
        
        except Exception as e:
            messagebox.showerror("Error", "Terjadi kesalahan: " + str(e))
            return
        
        print(algoritma)
        print(lama_studi)
        print(program_studi)
        print(jalur_masuk)

        # Mengubah dimensi tiap fitur menjadi 1 dimensi
        lama_studi = lama_studi.ravel()
        jalur_masuk = jalur_masuk.ravel()
        program_studi = program_studi.ravel()
        
        # Mengubah nama tiap program studi menjadi numerik sesuai label encoder
        if program_studi == 'Sarjana Administrasi Bisnis':
             program_studi_en = 0
        elif program_studi == 'Sarjana Administrasi Publik':
             program_studi_en = 1
        elif program_studi =='Sarjana Akuntansi':
             program_studi_en = 2
        elif program_studi ==   'Sarjana Arsitektur':
             program_studi_en = 3
        elif program_studi == 'Sarjana Ekonomi Pembangunan':
             program_studi_en = 4
        elif program_studi =='Sarjana Filsafat':
            program_studi_en = 5
        elif program_studi == 'Sarjana Fisika':
            program_studi_en = 6
        elif program_studi == 'Sarjana Hubungan Internasional':
            program_studi_en = 7
        elif program_studi ==    'Sarjana Hukum':
            program_studi_en = 8
        elif program_studi ==   'Sarjana Informatika':
            program_studi_en = 9
        elif program_studi == 'Sarjana Manajemen':
            program_studi_en = 10
        elif program_studi == 'Sarjana Matematika':
            program_studi_en = 11
        elif program_studi ==    'Sarjana Teknik Elektro':
            program_studi_en = 12
        elif program_studi ==    'Sarjana Teknik Industri':
            program_studi_en = 13
        elif program_studi ==   'Sarjana Teknik Kimia':
            program_studi_en = 14
        elif program_studi ==    'Sarjana Teknik Sipil':
            program_studi_en = 15
        
        # Mengubah nama jalur masuk menjadi numerik sesuai label encoder
        if jalur_masuk == 'PMDK':
            jalur_masuk_en = 0
        elif jalur_masuk == 'Seleksi Khusus':
            jalur_masuk_en = 1
        elif jalur_masuk ==    'USM1':
            jalur_masuk_en = 2
        elif jalur_masuk ==    'USM2':
            jalur_masuk_en = 3
        elif jalur_masuk ==    'USM3':
            jalur_masuk_en = 4
        
        # Menyatukan fitur
        feature_df = pd.DataFrame({
            'program_studi': [program_studi_en],
            'lama_studi': lama_studi,
            'jalur_masuk': [jalur_masuk_en]
        })
        # feature_df = pd.DataFrame()
        # feature_df['program_studi'] = program_studi_en
        # feature_df['lama_studi'] = lama_studi
        # feature_df['jalur_masuk'] = jalur_masuk_en
        
        print(program_studi_en)
        print(feature_df)
        print(feature_df.shape)
        
        # Melakukan prediksi peringkat ipk
        if algoritma == 'Naive Bayes':
            hasil_prediksi_ipk = self.model_ipk_nbc.predict(feature_df)
        elif algoritma == 'Decision Tree':
            hasil_prediksi_ipk = self.model_ipk_dt.predict(feature_df)
            
        print(hasil_prediksi_ipk)
        print(hasil_prediksi_ipk.dtype)
        # Menampilkan hasil prediksi peringkat ipk
        if hasil_prediksi_ipk[0]==1:
            self.output_string_ipk.set("Prediksi IPK: 3.50 - 4.00")
        elif hasil_prediksi_ipk[0]==2:
            self.output_string_ipk.set("Prediksi IPK: 3.00 - 3.49")
        elif hasil_prediksi_ipk[0]==3:
            self.output_string_ipk.set("Prediksi IPK: 2.50 - 2.99")
        elif hasil_prediksi_ipk[0]==4:
            self.output_string_ipk.set("Prediksi IPK: 2.00 - 2.49")
        
        

    def prediksi_lama_studi(self):
        # Mengambil input dari dropdown button dan menjadikan array
        ipk = np.array(self.entry_ipk_lama_studi.get())
        program_studi2 = np.array(self.entry_program_studi_lama_studi.get())
        jalur_masuk2 = np.array(self.entry_jalur_masuk_lama_studi.get())
        #provinsi_sma = np.array(self.entry_provinsi_sma_lama_studi.get())
        algoritma = np.array(self.entry_algoritma_lamaStudi.get())
        print('tipe data input ipk = ', type(self.entry_ipk_lama_studi.get()))
       
        # Exception setiap input harus terisi
        try:
            if not ipk or not program_studi2 or not jalur_masuk2 or not algoritma:
                raise ValueError("Semua input harus terisi")
            
        except ValueError as ve:
            messagebox.showerror("Error", str(ve))
            return
        
        except Exception as e:
            messagebox.showerror("Error", "Terjadi kesalahan: " + str(e))
            return
        
        # Exception untuk input IPK
        decimal = str(ipk).split('.')
        float_ipk = float(ipk)
        try:
            if(len(decimal)==2 and len(decimal[1]) == 2):
                if(float_ipk < 2.00 or float_ipk > 4.00):
                    raise ValueError("IPK harus berada dalam rentang 2.00 - 4.00")
            else:
                raise ValueError("IPK harus memiliki 2 angka di belakang koma")
            
        except ValueError as ve:
            messagebox.showerror("Error", str(ve))
            return
        
        except Exception as e:
            messagebox.showerror("Error", "Terjadi kesalahan: " + str(e))
            return
        
        print(algoritma)
        print(ipk)
        print(jalur_masuk2)
        
        # Mengubah dimensi tiap fitur menjadi 1 dimensi
        ipk = ipk.ravel()
        program_studi2 = program_studi2.ravel()
        jalur_masuk2 = jalur_masuk2.ravel()
        
        # Mengubah nama untuk tiap program studi menjadi numerik
        if program_studi2 == 'Sarjana Administrasi Bisnis':
             program_studi_en2 = 0
        elif program_studi2 == 'Sarjana Administrasi Publik':
             program_studi_en2 = 1
        elif program_studi2 =='Sarjana Akuntansi':
             program_studi_en2 = 2
        elif program_studi2 ==   'Sarjana Arsitektur':
             program_studi_en2 = 3
        elif program_studi2 == 'Sarjana Ekonomi Pembangunan':
             program_studi_en2 = 4
        elif program_studi2 =='Sarjana Filsafat':
            program_studi_en2 = 5
        elif program_studi2 == 'Sarjana Fisika':
            program_studi_en2 = 6
        elif program_studi2 == 'Sarjana Hubungan Internasional':
            program_studi_en2 = 7
        elif program_studi2 ==    'Sarjana Hukum':
            program_studi_en2 = 8
        elif program_studi2 ==   'Sarjana Informatika':
            program_studi_en2 = 9
        elif program_studi2 == 'Sarjana Manajemen':
            program_studi_en2 = 10
        elif program_studi2 == 'Sarjana Matematika':
            program_studi_en2 = 11
        elif program_studi2 ==    'Sarjana Teknik Elektro':
            program_studi_en2 = 12
        elif program_studi2 ==    'Sarjana Teknik Industri':
            program_studi_en2 = 13
        elif program_studi2 ==   'Sarjana Teknik Kimia':
            program_studi_en2 = 14
        elif program_studi2 ==    'Sarjana Teknik Sipil':
            program_studi_en2 = 15
        
        # Mengubah nama tiap jalur masuk menjadi numerik
        if jalur_masuk2 == 'Seleksi Khusus':
            jalur_masuk_en2 = 1
        elif jalur_masuk2 == 'PMDK':
            jalur_masuk_en2 = 0
        elif jalur_masuk2 ==    'USM1':
            jalur_masuk_en2 = 2
        elif jalur_masuk2 ==    'USM2':
            jalur_masuk_en2 = 3
        elif jalur_masuk2 ==    'USM3':
            jalur_masuk_en2 = 4
        
        # Menyatukan fitur
        feature_df2 = pd.DataFrame({
            'PROGRAM STUDI': program_studi_en2,
            'IPK' : ipk,
            'jalur_masuk': jalur_masuk_en2
        })
        
        print(feature_df2)
        # print('ipk', type(feature_df2['IPK']))
        # print('prodi', type(feature_df2['PROGRAM STUDI']))
        # print('jalur masuk', type(feature_df2['jalur_masuk']))
        
        feature_np = np.array(feature_df2)
        print(feature_np)
        
        # Melakukan prediksi lama studi
        if algoritma == 'Naive Bayes':
            hasil_prediksi_lama_studi = self.model_lama_studi_nbc.predict(feature_df2)
        elif algoritma == 'Decision Tree':
            hasil_prediksi_lama_studi = self.model_lama_studi_dt.predict(feature_df2)
        
        print(hasil_prediksi_lama_studi)
        # Menampilkan hasil prediksi lama studi
        self.output_string_lama_studi.set(f"Prediksi Lama Studi: {hasil_prediksi_lama_studi[0]} semester")

    def ipk_reset_inputs(self):
        self.entry_algoritma_ipk.set('')
        self.entry_jalur_masuk_ipk.set('')      
        self.entry_program_studi_ipk.set('')  
        self.entry_lama_studi_ipk.set('')
        self.output_string_ipk.set('')
        
    def lama_studi_reset_input(self):
        self.entry_algoritma_lamaStudi.set('')
        self.entry_jalur_masuk_lama_studi.set('')
        self.entry_ipk_lama_studi.delete(0, tk.END)
        self.entry_program_studi_lama_studi.set('')
        self.output_string_lama_studi.set('')

if __name__ == "__main__":
    root = tk.Tk()
    app = GUIApp(root)
    root.mainloop()




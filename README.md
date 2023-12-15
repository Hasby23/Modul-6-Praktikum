# Modul 6 Praktikum

## Rock, Paper, Scissors Image Classification
Proyek Flask ini adalah aplikasi web sederhana untuk klasifikasi gambar Gunting, Batu, Kertas menggunakan model pembelajaran mesin. Pengguna dapat mengunggah gambar tangan mereka yang sedang membuat salah satu gestur, dan model akan mengklasifikasikannya sebagai Gunting, Batu, Kertas.

## Machine Learning Model
Model yang digunakan dibuat dengan menggunakan model pre train MobileNetV2 yang dilatih menggunakan dataset citra Gunting, Batu, Kertas

## Dataset
![paper](https://github.com/Hasby23/Modul-6-Praktikum/assets/71579603/19484e39-6e6b-4aff-b694-857c1c74a54d)
![scissors](https://github.com/Hasby23/Modul-6-Praktikum/assets/71579603/61adbaf2-88c4-4f08-a2fb-94f5a937d007)
![rock](https://github.com/Hasby23/Modul-6-Praktikum/assets/71579603/b49d0e6e-84d7-44cd-8c90-5f331af9beef)
![scissors](https://github.com/Hasby23/Modul-6-Praktikum/assets/71579603/549715ab-4e21-4af7-9ce5-431550c246c9)

## Accuracy and Loss
![Screenshot (565)](https://github.com/Hasby23/Modul-6-Praktikum/assets/71579603/24d8f712-e4ae-4e3a-bec9-375602eb2ac1)
![Screenshot (566)](https://github.com/Hasby23/Modul-6-Praktikum/assets/71579603/4a0ad774-2553-416b-b560-d8f934691d2f)

## Installation
1. git clone https://github.com/Hasby23/Modul-6-Praktikum.git
2. cd Modul-6-Praktikum
3. pip install -r requirements.txt

## Usage
1. Run file app.py
2. Buka browser dan navigasi ke http://localhost:2000/
3. Upload gambar tangan membuat gestur gunting, batu, atau kertas

## Project Structure
- 'app.py'          : File utama aplikasi flask
- 'static/models/'  : Direktori yang berisikan model machine learning
- 'static/uploads/' : direktori untuk menyimpan citra yang di upload
- 'templates/'      : Templates HTML untuk webpages
- 'requirements.txt': List library yang digunakan

## Local Development
- Prediksi rock
![Screenshot (564)](https://github.com/Hasby23/Modul-6-Praktikum/assets/71579603/bdcbf10a-93fa-4c29-ac94-ee1192150ac7)

- Prediksi paper
![Screenshot (562)](https://github.com/Hasby23/Modul-6-Praktikum/assets/71579603/eb8cd576-ca6e-4824-9610-26ed3c216073)

- Prediksi scissors
![Screenshot (563)](https://github.com/Hasby23/Modul-6-Praktikum/assets/71579603/cc898b23-29d2-4b3d-9209-61ab6581af04)

## Author
- [@Hasby23](https://www.github.com/Hasby23)



from flask import Flask, render_template, request, send_from_directory
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import time
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './static/uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
model = load_model('./static/models/rps.h5')

@app.route('/', methods=['GET', 'POST'])
def index():
    """
    Fungsi untuk menampilkan halaman utama pada aplikasi Flask.
    
    Jika request method adalah 'POST', fungsi ini memproses gambar yang diupload,
    melakukan klasifikasi gambar menggunakan model yang sudah dilatih, dan manampilkan hasilnya di halaman.

    Parameters:
    - None

    Return:
    - render_template: Template HTML untuk halaman utama, dengan konten yang diperbarui berdasarkan hasil klasifikasi gambar.
      Template termasuk label prediksi, durasi waktu eksekusi, persentase akurasi dan
      gambar yang diunggah
    """

    if request.method == 'POST':
        if request.files:
            image = request.files['image']
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
            image.save(img_path)

            img = np.array(Image.open(image).resize((128, 128)).convert('RGB'))
            img = img / 255

            img = img[np.newaxis, :]

            start = time.time()

            pred = model.predict(img)

            # Prediction Time
            runtimes = round(time.time()-start,4)

            max_index = np.argmax(pred[0])
            max_probability = pred[0][max_index]

            # Percentage
            max_percentage = round(max_probability * 100, 2)            

            if max_index == 0:
                result = "Paper"
            elif max_index == 1:
                result = "Rock"
            else:
                result = "Scissors"

            return render_template('/index.html', label=result,
                            run_time=runtimes, img=img, 
                            accuracy=max_percentage, uploaded_image=image.filename)

    return render_template('/index.html')


@app.route('/display/<filename>')
def send_uploaded_image(filename=''):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=2000)

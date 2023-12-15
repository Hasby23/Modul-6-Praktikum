from flask import Flask, render_template, request, send_from_directory
from PIL import Image
import tflite_runtime.interpreter as tflite
import numpy as np
import time
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './static/uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
interpreter = tflite.Interpreter(model_path="static/models/rps.tflite")
interpreter.allocate_tensors()

@app.after_request
def add_header(r):
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if request.files:
            image = request.files['image']
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
            image.save(img_path)

            img = np.array(Image.open(image).resize((128, 128)).convert('RGB'))
            img = img / 255

            img = img[np.newaxis, :]

            # Prediction Time
            start = time.time()

            # Accuracy
            # pred = model.predict(img)

            img = img.astype(np.float32)
            input_tensor_index = interpreter.get_input_details()[0]['index']
            interpreter.set_tensor(input_tensor_index, img)

            # Run inference
            interpreter.invoke()

            # Get the output tensor
            output_tensor_index = interpreter.get_output_details()[0]['index']
            pred = interpreter.get_tensor(output_tensor_index)

            max_index = np.argmax(pred[0])
            print(max_index)
            max_probability = pred[0][max_index]
            max_percentage = round(max_probability * 100, 2)

            
            runtimes = round(time.time()-start,4)

            # Predicted Label
            if max_index == 0:
                result = "Paper"
            elif max_index == 1:
                result = "Rock"
            else:
                result = "Scissors"

            # max_percentage = prediction persentage
            # result = label
            # runtimes = the times in second
            # image
            return render_template('/index.html', label=result,
                            run_time=runtimes, img=img, 
                            accuracy=max_percentage, uploaded_image=image.filename)

    return render_template('/index.html')


@app.route('/display/<filename>')
def send_uploaded_image(filename=''):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=False)

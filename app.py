from flask import Flask, jsonify, request, render_template
from werkzeug.utils import secure_filename
import tensorflow as tf
import numpy as np
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'D:\WebPython\static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load model yang telah dilatih sebelumnya
model = tf.keras.models.load_model('D:\WebPython\my_model.h5')

# Tentukan list kelas yang akan diprediksi
# CLASSES = ['Apel Segar','Apel Busuk']

# Tentukan maksimum ukuran file yang dapat diupload
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

# Fungsi untuk melakukan prediksi pada gambar
def predict(image_path):
    image = tf.keras.preprocessing.image.load_img(image_path, target_size=(150, 150))
    image_array = tf.keras.preprocessing.image.img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)
    images = np.vstack([image_array])
    class_name = model.predict(images, batch_size = 16)
    return class_name

# routes
@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("index.html")

# Definisikan endpoint untuk melakukan prediksi pada gambar
@app.route('/predict', methods=['POST'])
def predict_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return render_template("index.html", unknown = "File Belum Ter-Upload")
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        predicted_class = predict(file_path)
        img_path = "static/" + filename
        segar = predicted_class[0][0] * 100
        busuk = predicted_class[0][1] * 100
        segarint = int(segar)
        busukint = int(busuk)
        bg = "static\cg3.png"

        if predicted_class[0, 0] == 1:
            return render_template("index.html", prediction = "Masih Segar", img_path = img_path, ratio0 = segarint, ratio1 = busukint, bg = bg)
        elif predicted_class[0, 1] == 1:
            return render_template("index.html", prediction = "Sudah Busuk", img_path = img_path, ratio0 = segarint, ratio1 = busukint, bg = bg)
        else:
            return render_template("index.html", unknown = "Citra Tidak Diketahui")
    else:
        """ return jsonify({'error': 'Invalid file type'}) """
        return render_template("index.html", unknown = "File Tidak Diketahui")
    

# Fungsi untuk memeriksa apakah jenis file yang diupload diperbolehkan
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in {'jpg', 'jpeg', 'png'}

if __name__ == '__main__':
    app.run()

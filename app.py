from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
import os
from PIL import Image
from io import BytesIO

app = Flask(__name__, static_url_path='/static')

# Cargar el modelo previamente entrenado
loaded_model = load_model('TC3007_DeepLearning\emotions_30epochs.h5')

def predict_image(file_path):
    class_labels = ['Angry', 'Happy', 'Sad']
    # Cargar y preprocesar la imagen
    img = image.load_img(file_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Hacer la predicción
    prediction = loaded_model.predict(img_array)

    # Interpretar los resultados
    predicted_class_index = np.argmax(prediction)
    predicted_class = class_labels[predicted_class_index]  # Asegúrate de tener tus etiquetas de clase

    return predicted_class

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Manejar la carga de la imagen
        uploaded_file = request.files['file']
        if uploaded_file.filename != '':
            # Guardar la imagen cargada en un archivo temporal
            filename = secure_filename(uploaded_file.filename)
            uploaded_file.save(filename)

            # Obtener la predicción
            prediction = predict_image(filename)

            return render_template('index.html', prediction=prediction, image_path=filename)

    return render_template('index.html', prediction=None, image_path=None)

if __name__ == '__main__':
    app.run(debug=True)

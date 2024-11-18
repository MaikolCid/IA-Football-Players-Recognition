from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from werkzeug.utils import secure_filename
import os
import logging
import pickle
from PIL import Image
import numpy as np
from mtcnn.mtcnn import MTCNN
import tensorflow as tf

# Configuración de la aplicación
app = Flask(__name__)

# Clave secreta para flash
app.secret_key = 'tu_clave_secreta'

# Configurar el registro
logging.basicConfig(level=logging.DEBUG)

# Carpeta donde se guardarán las imágenes subidas
UPLOAD_FOLDER = 'analizar'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Carpeta donde se guardarán las imágenes recortadas
CROPPED_FOLDER = 'recortadas'
if not os.path.exists(CROPPED_FOLDER):
    os.makedirs(CROPPED_FOLDER)

# Extensiones permitidas
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Tamaño de las imágenes para FaceNet
IMG_SIZE = 160

# Ruta al modelo FaceNet en formato .pb
FACENET_MODEL_PATH = 'facenet/20180402-114759.pb'

# Cargar el modelo FaceNet
def load_facenet_model(pb_path):
    with tf.io.gfile.GFile(pb_path, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.compat.v1.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="")
    return graph

# Cargar el modelo KNN y LabelEncoder
with open('modelo_knn_jugadores.pkl', 'rb') as f:
    knn = pickle.load(f)
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Cargar el modelo FaceNet
graph = load_facenet_model(FACENET_MODEL_PATH)
input_tensor = graph.get_tensor_by_name("input:0")
embedding_tensor = graph.get_tensor_by_name("embeddings:0")
phase_train_tensor = graph.get_tensor_by_name("phase_train:0")
session = tf.compat.v1.Session(graph=graph)

# Inicializar el detector de rostros una vez
detector = MTCNN()

# Función para verificar si el archivo tiene una extensión permitida
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Ruta principal que muestra el formulario para subir imágenes
@app.route('/')
def index():
    return render_template('index.html')

# Función para obtener el embedding usando FaceNet
def get_embedding(image):
    image = np.expand_dims(image, axis=0)
    feed_dict = {input_tensor: image, phase_train_tensor: False}
    embedding = session.run(embedding_tensor, feed_dict=feed_dict)
    return embedding[0]

# Función para analizar y predecir el jugador en una imagen
def analizar_imagen(ruta_imagen):
    img = Image.open(ruta_imagen).convert('RGB')
    img_array = np.array(img)

    # Detectar rostro
    results = detector.detect_faces(img_array)

    if len(results) > 0:
        # Tomar el primer rostro detectado
        x, y, width, height = results[0]['box']
        x, y = abs(x), abs(y)
        rostro_array = img_array[y:y+height, x:x+width]

        # Redimensionar el rostro para FaceNet
        rostro_img = Image.fromarray(rostro_array).resize((IMG_SIZE, IMG_SIZE))
        rostro_array = np.array(rostro_img)
        rostro_array = (rostro_array - 127.5) / 128.0  # Normalización para FaceNet

        # Obtener el embedding
        embedding = get_embedding(rostro_array)

        # Predecir la clase usando el clasificador KNN
        prediction = knn.predict([embedding])
        jugador_predicho = label_encoder.inverse_transform(prediction)[0]
        
        return jugador_predicho
    else:
        return "No se detectó rostro en la imagen"

# Ruta que maneja la subida de imágenes y el análisis
@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            flash('No se ha enviado ningún archivo')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No se ha seleccionado ningún archivo')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Analizar la imagen
            jugador_predicho = analizar_imagen(filepath)
            return render_template('resultado.html', jugador=jugador_predicho)
        else:
            flash('Tipo de archivo no permitido')
            return redirect(request.url)
    except Exception as e:
        logging.exception("Error en upload_file")
        flash('Ha ocurrido un error al procesar la imagen')
        return redirect(request.url)

# SOLO PARA ENTRENAMIENTO
@app.route('/subir', methods=['GET', 'POST'])
def subir_imagen():
    if request.method == 'POST':
        try:
            jugador_seleccionado = request.form['jugador']
            if 'file' not in request.files:
                flash('No se ha enviado ningún archivo')
                return redirect(request.url)
            file = request.files['file']
            if file.filename == '':
                flash('No se ha seleccionado ningún archivo')
                return redirect(request.url)
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                # Guardar temporalmente la imagen subida
                temp_folder = 'temp_uploads'
                if not os.path.exists(temp_folder):
                    os.makedirs(temp_folder)
                temp_filepath = os.path.join(temp_folder, filename)
                file.save(temp_filepath)

                # Procesar la imagen (detección y recorte de rostro)
                imagen_procesada = procesar_imagen_para_entrenamiento(temp_filepath)

                # Guardar la imagen en la carpeta correspondiente
                jugador_folder = os.path.join('jugadores_preprocesados', jugador_seleccionado)
                if not os.path.exists(jugador_folder):
                    os.makedirs(jugador_folder)
                destino = os.path.join(jugador_folder, filename)
                imagen_procesada.save(destino)

                # Eliminar la imagen temporal
                os.remove(temp_filepath)

                flash('Imagen subida y asignada correctamente al jugador {}'.format(jugador_seleccionado))
                return redirect(url_for('subir_imagen'))
            else:
                flash('Tipo de archivo no permitido')
                return redirect(request.url)
        except Exception as e:
            print(f"Error en subir_imagen: {e}")
            flash('Ha ocurrido un error al procesar la imagen')
            return redirect(request.url)
    else:
        jugadores = sorted(os.listdir('jugadores_preprocesados'))
        jugador_seleccionado = request.args.get('jugador')  # Obtener jugador seleccionado
        jugador_imagenes = []

        # Cargar imágenes solo si hay un jugador seleccionado
        if jugador_seleccionado:
            jugador_folder = os.path.join('jugadores_preprocesados', jugador_seleccionado)
            if os.path.isdir(jugador_folder):
                jugador_imagenes = sorted(
                    os.listdir(jugador_folder),
                    key=lambda x: os.path.getmtime(os.path.join(jugador_folder, x)),
                    reverse=True
                )

        return render_template(
            'subir_imagen.html', 
            jugadores=jugadores, 
            jugador_seleccionado=jugador_seleccionado, 
            jugador_imagenes=jugador_imagenes
        )

def procesar_imagen_para_entrenamiento(ruta_imagen):
    # Cargar la imagen
    img = Image.open(ruta_imagen)
    img = img.convert('RGB')
    img_array = np.array(img)

    # Detectar rostros
    results = detector.detect_faces(img_array)

    if len(results) > 0:
        # Suponiendo que solo hay un rostro por imagen (el primero detectado)
        x, y, width, height = results[0]['box']
        x, y = abs(x), abs(y)  # Asegurarse de que los valores sean positivos
        rostro_array = img_array[y:y+height, x:x+width]

        # Redimensionar el rostro con el filtro LANCZOS
        rostro_img = Image.fromarray(rostro_array)
        rostro_img = rostro_img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)

        return rostro_img
    else:
        raise ValueError("No se detectó rostro en la imagen")

if __name__ == '__main__':
    app.run(debug=True)


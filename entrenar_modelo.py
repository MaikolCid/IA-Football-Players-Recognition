import tensorflow as tf
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os
import pickle

# Ruta al archivo .pb
MODEL_PATH = 'facenet/20180402-114759.pb'

# Función para cargar el modelo .pb de FaceNet
def load_facenet_pb_model(pb_path):
    with tf.io.gfile.GFile(pb_path, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
        
    with tf.compat.v1.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="")
    return graph

# Cargar el modelo en un grafo
graph = load_facenet_pb_model(MODEL_PATH)

# Obtener las referencias a los tensores de entrada y salida
input_tensor = graph.get_tensor_by_name("input:0")
embedding_tensor = graph.get_tensor_by_name("embeddings:0")
phase_train_tensor = graph.get_tensor_by_name("phase_train:0")

# Sesión de TensorFlow para ejecutar el modelo
session = tf.compat.v1.Session(graph=graph)

# Función para obtener el embedding de una imagen usando FaceNet
def get_embedding(image_path):
    # Cargar y preprocesar la imagen
    image = load_img(image_path, target_size=(160, 160))  # Tamaño de FaceNet
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = (image - 127.5) / 128.0  # Normalización
    
    # Ejecutar el modelo para obtener el embedding
    feed_dict = {input_tensor: image, phase_train_tensor: False}
    embedding = session.run(embedding_tensor, feed_dict=feed_dict)
    return embedding[0]

# Directorio con las imágenes de los jugadores
DATA_DIR = 'jugadores_preprocesados'

# Generar embeddings y etiquetas
embeddings = []
labels = []

# Crear los embeddings para cada imagen en las carpetas de los jugadores
for player_name in os.listdir(DATA_DIR):
    player_dir = os.path.join(DATA_DIR, player_name)
    if os.path.isdir(player_dir):
        for image_name in os.listdir(player_dir):
            image_path = os.path.join(player_dir, image_name)
            embedding = get_embedding(image_path)
            embeddings.append(embedding)
            labels.append(player_name)

# Convertir etiquetas en formato numérico
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels)

# Guardar el codificador de etiquetas
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

# Entrenar un clasificador KNN sobre los embeddings
knn = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
knn.fit(embeddings, y)

# Guardar el clasificador entrenado
with open('modelo_knn_jugadores.pkl', 'wb') as f:
    pickle.dump(knn, f)

print("Modelo KNN para reconocimiento facial de jugadores entrenado y guardado.")

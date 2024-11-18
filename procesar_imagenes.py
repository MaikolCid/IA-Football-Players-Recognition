import cv2
import os
from PIL import Image

# Ruta al clasificador Haar Cascade
CASCADE_PATH = 'haarcascade_frontalface_default.xml'

# Directorio de datos originales
DATA_DIR = 'jugadores2'

# Directorio para guardar las imágenes preprocesadas
OUTPUT_DIR = 'jugadores_preprocesados'

# Tamaño deseado para las imágenes de salida
IMG_SIZE = 224

# Cargar el clasificador de rostros
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

def procesar_imagenes():
    # Crear el directorio de salida si no existe
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Recorrer cada carpeta (clase/jugador) en el directorio de datos
    for clase in os.listdir(DATA_DIR):
        clase_path = os.path.join(DATA_DIR, clase)
        if os.path.isdir(clase_path):
            print(f"Procesando imágenes de: {clase}")
            # Crear la carpeta correspondiente en el directorio de salida
            clase_output_path = os.path.join(OUTPUT_DIR, clase)
            if not os.path.exists(clase_output_path):
                os.makedirs(clase_output_path)
            # Recorrer cada imagen en la carpeta del jugador
            for imagen_nombre in os.listdir(clase_path):
                imagen_path = os.path.join(clase_path, imagen_nombre)
                try:
                    # Leer la imagen
                    img = cv2.imread(imagen_path)
                    # Convertir a escala de grises
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    # Detectar rostros en la imagen
                    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
                    if len(faces) == 0:
                        print(f"No se detectó rostro en la imagen: {imagen_nombre}")
                        continue
                    # Suponiendo que solo hay un rostro por imagen (el primero detectado)
                    (x, y, w, h) = faces[0]
                    # Recortar la imagen para obtener solo la cara
                    rostro = img[y:y+h, x:x+w]
                    # Redimensionar la imagen
                    rostro = cv2.resize(rostro, (IMG_SIZE, IMG_SIZE))
                    # Guardar la imagen preprocesada
                    imagen_output_path = os.path.join(clase_output_path, imagen_nombre)
                    cv2.imwrite(imagen_output_path, rostro)
                except Exception as e:
                    print(f"Error procesando la imagen {imagen_nombre}: {e}")

if __name__ == '__main__':
    procesar_imagenes()
    print("Preprocesamiento completado.")

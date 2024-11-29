import matplotlib.pyplot as plt
import random
import os
from skimage import io
from skimage.segmentation import slic
from skimage.color import label2rgb
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score


images_directory = 'Patologia'
imagenes = []
etiquetas = []


# Cargar imágenes de entrenamiento y etiquetarlas según su carpeta de origen
clases = ['Tejido', 'Tumor']
for clase in clases:
   subdirectory = os.path.join(images_directory, clase)
   for cont, file_name in enumerate(sorted(os.listdir(subdirectory))):
       if cont >= 12:
           break
       file_path = os.path.join(subdirectory, file_name)
       if os.path.isfile(file_path):
           imagen = io.imread(file_path)
           imagenes.append(imagen)
           etiquetas.append(clase)


# Parámetros para la segmentación SLIC
n_segments = 200  # Número de superpíxeles a crear
ventanas = []
ventanas_etiquetas = []


# Extraer características de los superpíxeles
def extraer_caracteristicas_superpixeles(imagen, etiquetas):
   caracteristicas = []
   for label in np.unique(etiquetas):
       mask = (etiquetas == label)
       if np.sum(mask) > 0:
           region = imagen[mask]
           mean_color = np.mean(region, axis=0)
           caracteristicas.append(mean_color)
   return caracteristicas


# Procesar cada imagen
for cont, imagen in enumerate(imagenes):
   # Aplicar SLIC para segmentar la imagen
   etiquetas_superpixeles = slic(imagen, n_segments=n_segments, compactness=10, start_label=1)
   # Extraer características de los superpíxeles
   caracteristicas = extraer_caracteristicas_superpixeles(imagen, etiquetas_superpixeles)
   ventanas.extend(caracteristicas)
   ventanas_etiquetas.extend([etiquetas[cont]] * len(caracteristicas))


# Entrenar el modelo KNN
X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(
   ventanas, ventanas_etiquetas, test_size=0.01, random_state=42
)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_entrenamiento, y_entrenamiento)
print("\nEvaluación del modelo KNN en entrenamiento:")
print(classification_report(y_entrenamiento, knn.predict(X_entrenamiento)))


# Cargar y procesar imágenes de prueba
prueba_directory = os.path.join(images_directory, 'Prueba')
for file_name in sorted(os.listdir(prueba_directory)):
   file_path = os.path.join(prueba_directory, file_name)
   if os.path.isfile(file_path):
       imagen_prueba = io.imread(file_path)


       # Aplicar SLIC a la imagen de prueba
       etiquetas_superpixeles = slic(imagen_prueba, n_segments=n_segments, compactness=10, start_label=1)


       resultado = np.zeros(imagen_prueba.shape[:2], dtype=np.uint8)


       # Clasificar cada superpíxel
       for label in np.unique(etiquetas_superpixeles):
           mask = (etiquetas_superpixeles == label)
           if np.sum(mask) > 0:
               region = imagen_prueba[mask]
               mean_color = np.mean(region, axis=0)
               caracteristicas = mean_color.reshape(1, -1)
               prediccion = knn.predict(caracteristicas)[0]


               # Asignar colores según la predicción
               color = 255 if prediccion == 'Tejido' else 0
               resultado[mask] = color


       # Mostrar la imagen clasificada en blanco y negro
       plt.imshow(resultado, cmap='gray')
       plt.title(f'Clasificación de {file_name}')
       plt.axis('off')
       plt.show()

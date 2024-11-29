import matplotlib.pyplot as plt
import random
import os
from skimage import io
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.svm import SVC


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


# Parámetros para las ventanas y las configuraciones de GLCM y LBP
ventanita_entrenamiento = 256
ventanita_prueba = 128
distancias = [1, 2, 3,4,5,6,7]
angulos = [0, np.pi / 4, np.pi / 6, np.pi / 3, np.pi / 2, 2 * np.pi / 3, 5 * np.pi / 6, 3 * np.pi / 4]
ventanas = []
ventanas_etiquetas = []


# Dividir imágenes de entrenamiento en ventanas de 256x256 y extraer características
def extraer_caracteristicas_glcm(ventana, distances, angles):
   if len(ventana.shape) == 3:
       ventana_gris = np.dot(ventana[..., :3], [0.2989, 0.5870, 0.1140])
   else:
       ventana_gris = ventana


   caracteristicas = []
   for dist in distances:
       for angle in angles:
           glcm = graycomatrix(ventana_gris.astype(np.uint8), distances=[dist], angles=[angle],
                               levels=256, symmetric=True, normed=True)
           contraste = graycoprops(glcm, 'contrast')[0, 0]
           homogeneidad = graycoprops(glcm, 'homogeneity')[0, 0]
           energia = graycoprops(glcm, 'energy')[0, 0]
           entropia = -np.sum(glcm * np.log2(glcm + (glcm == 0)))
           dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
           correlation = graycoprops(glcm, 'correlation')[0, 0]
           asm = graycoprops(glcm, 'ASM')[0, 0]
           caracteristicas.extend([contraste, homogeneidad, energia, entropia, dissimilarity, correlation, asm])


   ventana_gris_uint8 = (ventana_gris * 255).astype(np.uint8) if ventana_gris.dtype != np.uint8 else ventana_gris
   lbp = local_binary_pattern(ventana_gris_uint8, P=8, R=1, method='uniform')
   lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, lbp.max() + 1), density=True)
   caracteristicas.extend(lbp_hist)


   return caracteristicas


# Extraer ventanas y características para entrenamiento
for cont, imagen in enumerate(imagenes):
   alto, ancho = imagen.shape[:2]
   for i in range(0, alto, ventanita_entrenamiento):
       for j in range(0, ancho, ventanita_entrenamiento):
           ventana = imagen[i:i + ventanita_entrenamiento, j:j + ventanita_entrenamiento]
           if ventana.shape[0] == ventanita_entrenamiento and ventana.shape[1] == ventanita_entrenamiento:
               ventanas.append(extraer_caracteristicas_glcm(ventana, distancias, angulos))
               ventanas_etiquetas.append(etiquetas[cont])


# Entrenar el modelo KNN
X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(
   ventanas, ventanas_etiquetas, test_size=0.01, random_state=42
)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_entrenamiento, y_entrenamiento)
print("\nEvaluación del modelo KNN en entrenamiento:")
print(classification_report(y_entrenamiento, knn.predict(X_entrenamiento)))


# Cargar y procesar imágenes de prueba, dividir en ventanas de 128x128
prueba_directory = os.path.join(images_directory, 'Prueba')
for file_name in sorted(os.listdir(prueba_directory)):
   file_path = os.path.join(prueba_directory, file_name)
   if os.path.isfile(file_path):
       imagen_prueba = io.imread(file_path)
       alto, ancho = imagen_prueba.shape[:2]
       resultado = np.zeros((alto, ancho), dtype=np.uint8)


       # Clasificar cada ventana de prueba
       for i in range(0, alto, ventanita_prueba):
           for j in range(0, ancho, ventanita_prueba):
               ventana = imagen_prueba[i:i + ventanita_prueba, j:j + ventanita_prueba]
               if ventana.shape[0] == ventanita_prueba and ventana.shape[1] == ventanita_prueba:
                   caracteristicas = extraer_caracteristicas_glcm(ventana, distancias, angulos)
                   prediccion = knn.predict([caracteristicas])[0]


                   # Asignar colores según la predicción
                   color = 255 if prediccion == 'Tejido' else 0
                   resultado[i:i + ventanita_prueba, j:j + ventanita_prueba] = color


       # Mostrar la imagen clasificada en blanco y negro
       plt.imshow(resultado, cmap='gray')
       plt.title(f'Clasificación de {file_name}')
       plt.axis('off')
       plt.show()

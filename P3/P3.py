import matplotlib.pyplot as plt
import random
import os
from skimage import io
from skimage.feature import graycomatrix, graycoprops
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score


images_directory = 'Brodatz - copia'
imagenes = []
etiquetas = []


# Cargamos 4 imagenes de clase0 (ladrillo), 3 de clase 1 que son las palmas y 3 clase 2 (puntos)
clases = ['clase0', 'clase1', 'clase2']
for clase in clases:
   subdirectory = os.path.join(images_directory, clase)


   for cont, file_name in enumerate(sorted(os.listdir(subdirectory))):
       if cont >= 12:
           break


       file_path = os.path.join(subdirectory, file_name)


       if os.path.isfile(file_path):
           imagen = io.imread(file_path)
           imagenes.append(imagen)
           etiquetas.append(clase) #Aqui se hace el etiquetado dependiendo de la carpeta origen


# Ventana cuadrada de 64x64, las imagenes son de 640x640
ventanita = 64
ventanas = []
ventanas_etiquetas = []


# Recorrer las imágenes y dividirlas en ventanas
for cont, imagen in enumerate(imagenes):
   alto, ancho = imagen.shape[:2]
   for i in range(0, alto, ventanita):
       for j in range(0, ancho, ventanita):
           # Extraer ventana de la imagen
           ventana = imagen[i:i + ventanita, j:j + ventanita]
           # Asegurarse de que la ventana sea del tamaño correcto
           if ventana.shape[0] == ventanita and ventana.shape[1] == ventanita:
               ventanas.append(ventana)
               ventanas_etiquetas.append(etiquetas[cont])  # Asignar la etiqueta




# Funcion del GLCM
def extraer_caracteristicas_glcm(ventana, distances, angles):
   # Convertir ventana a escala de grises si es una imagen RGB
   if len(ventana.shape) == 3:
       ventana_gris = np.dot(ventana[..., :3], [0.2989, 0.5870, 0.1140])
   else:
       ventana_gris = ventana


   caracteristicas = []


   for dist in distances:
       for angle in angles:
           # Construir matriz de Co-ocurrencia (en niveles de gris)
           glcm = graycomatrix(ventana_gris.astype(np.uint8), distances=[dist], angles=[angle],
                               levels=256, symmetric=True, normed=True)


           # Extraer características de GLCM
           contraste = graycoprops(glcm, 'contrast')[0, 0]
           homogeneidad = graycoprops(glcm, 'homogeneity')[0, 0]
           energia = graycoprops(glcm, 'energy')[0, 0]
           entropia = -np.sum(glcm * np.log2(glcm + (glcm == 0)))
           caracteristicas.extend([contraste, homogeneidad, energia, entropia])


   return caracteristicas




def evaluar_clasificadores(distances, angles):
   X = []


   for ventana in ventanas:
       caracteristicas = extraer_caracteristicas_glcm(ventana, distances, angles)
       X.append(caracteristicas)


   X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(X, ventanas_etiquetas, test_size=0.2,
                                                                           random_state=42)


   # KNN
   knn = KNeighborsClassifier(n_neighbors=3)
   knn.fit(X_entrenamiento, y_entrenamiento)
   y_pred_knn = knn.predict(X_prueba)


   print(f"\nResultados con KNN usando distancias: {distances} y ángulos: {angles}")
   print(classification_report(y_prueba, y_pred_knn))
   print(f"Accuracy (KNN): {accuracy_score(y_prueba, y_pred_knn)}")


   #  SVM
   svm = SVC(kernel='linear')
   svm.fit(X_entrenamiento, y_entrenamiento)
   y_pred_svm = svm.predict(X_prueba)


   print(f"\nResultados con SVM usando distancias: {distances} y ángulos: {angles}")
   print(classification_report(y_prueba, y_pred_svm))
   print(f"Accuracy (SVM): {accuracy_score(y_prueba, y_pred_svm)}")


   # Tomar 5 ventanas aleatoriamente para ilustrar el programa
   fig, axs = plt.subplots(1, 5, figsize=(15, 5))
   fig.suptitle('Clasificación de 5 ventanas aleatorias de prueba')


   indices_aleatorios = random.sample(range(len(X_prueba)), 5)


   for i, idx in enumerate(indices_aleatorios):
       ventana_original_idx = X.index(X_prueba[idx])  # Índice real de la ventana en la lista original
       ventana_original = ventanas[ventana_original_idx]  # Obtener la ventana original
       etiqueta_real = y_prueba[idx]
       etiqueta_predicha = y_pred_knn[idx]


       axs[i].imshow(ventana_original, cmap='gray')
       axs[i].set_title(f'Real: {etiqueta_real}\nPred: {etiqueta_predicha}')
       axs[i].axis('off')


   plt.show()




configurations = [
   {'distancia': [1], 'orientacion': [0]},  # Distancia de 1, horizontalmente (0°)
   {'distancia': [3], 'orientacion': [0]},  # Distancia de 3, horizontalmente (0°)
   {'distancia': [1], 'orientacion': [np.pi / 4]},  # Distancia de 1, diagonal ascendente derecha (45°)
   {'distancia': [1], 'orientacion': [3 * np.pi / 4]},  # Distancia de 1, diagonal ascendente izquierda (135°)
   {'distancia': [1, 2], 'orientacion': [0, np.pi / 4]},  # Distancias 1 y 2, horizontalmente (0°) y diagonal ascendente derecha (45°)
   {'distancia': [1, 2], 'orientacion': [np.pi / 2, 3 * np.pi / 4]},  # Distancias 1 y 2, verticalmente (90°) y diagonal ascendente izquierda (135°)
   {'distancia': [1, 3], 'orientacion': [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]},  # Distancias 1 y 3, horizontalmente (0°), diagonal ascendente derecha (45°), verticalmente (90°) y diagonal ascendente izquierda (135°)
   {'distancia': [1, 4], 'orientacion': [0, np.pi / 2]},  # Distancias 1 y 4, horizontalmente (0°) y verticalmente (90°)
   {'distancia': [1, 2, 3], 'orientacion': [0, np.pi / 4, np.pi / 2]}  # Distancias 1, 2 y 3, horizontalmente (0°), diagonal ascendente derecha (45°) y verticalmente (90°)
]



# Ciclo para mostrar las metricas
for config in configurations:
   distancias = config['distancia']
   angulos = config['orientacion']
   evaluar_clasificadores(distancias, angulos)

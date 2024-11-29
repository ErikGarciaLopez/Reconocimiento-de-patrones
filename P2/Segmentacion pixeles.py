import numpy as np
import os
from PIL import Image, ImageFilter
from skimage import io, draw
import matplotlib.pyplot as plt
from matplotlib.widgets import PolygonSelector
#se deben seleccionar las clases en orden de banana, chile, huevo en la figura
# Colores excluidos, lista para medias y covarianzas y datos del fondo y el platano
acumulado = 0
priori = []
medias = []
colores_excluir = [np.array([175, 50, 32]), np.array([222, 91, 61]), np.array([151, 63, 41]),
                   np.array([213, 115, 106]), np.array([202, 140, 145]),
                   np.array([255, 140, 109]), np.array([106, 38, 19]),
                   np.array([185, 79, 55]), np.array([177, 98, 91]),
                   np.array([226.38698107, 90.25534188, 63.83707265]),
                   np.array([208, 110, 39]),np.array([205, 99, 37])]
umbral_distancia = 30

fondomed = np.array([222.46339678,  86.90785391,  61.44530337])
fondovar = np.array([[125.75563668 , 51.77164678 , 47.19401452],
 [ 51.77164678,  47.19693844 , 31.14250208],
 [ 47.19401452,  31.14250208 , 25.32974721]])

banmed = np.array([212.69594469, 189.86605505 , 56.21662013])
banvar = np.array([[450.64225698, 564.63829575, 279.65085721],
 [564.63829575, 737.45251354, 363.45099233],
 [279.65085721, 363.45099233 ,390.5381718 ]])

chmed = np.array([52.14456739, 78.99299293, 26.28224197])
chvar = np.array([[522.05255134, 384.55960224, 519.81883663],
 [384.55960224, 878.12501571, 798.80866994],
 [519.81883663, 798.80866994 ,879.13549295]])

def calcular_Yk(X, media, covarianza, prior):
    diff = X - media
    cov_inv = np.linalg.inv(covarianza)
    term1 = -0.5 * np.dot(np.dot(diff.T, cov_inv), diff)
    term2 = -0.5 * np.log(np.linalg.det(covarianza))
    term3 = np.log(prior)
    Yk = term1 + term2 + term3
    return Yk

def calcular_media_covarianza(imagen, vertices):
    global acumulado
    vertices = np.array(vertices)
    mascara = np.zeros(imagen.shape[:2], dtype=bool)
    rr, cc = draw.polygon(vertices[:, 1], vertices[:, 0], mascara.shape)
    mascara[rr, cc] = True
    pixeles_seleccionados = imagen[mascara]

    distancias = np.array([np.linalg.norm(pixeles_seleccionados[:, :3] - color, axis=1) for color in colores_excluir])
    mascara_validos = np.all(distancias > umbral_distancia, axis=0)

    pixeles_validos = pixeles_seleccionados[mascara_validos]

    acumulado += pixeles_validos.shape[0] #esta variable se usa para calcular el fondo

    if pixeles_validos.size > 0:
        media = np.mean(pixeles_validos, axis=0)
        covarianza = np.cov(pixeles_validos, rowvar=False)
        return media, covarianza, mascara, mascara_validos, pixeles_validos.shape[0]
    else:
        return None, None, None, None, 0

def on_select(verts):
    global vertices
    global pixeles_totales
    vertices = verts
    media, covarianza, mascara, mascara_validos, pixeles_validos = calcular_media_covarianza(imagen, vertices)

    if media is not None:
        print(f"Media: {media}, \nCovarianza:\n{covarianza}")
        medias.append(media)
        medias.append(covarianza)

        imagen_final = np.zeros_like(imagen)
        imagen_final[mascara] = imagen[mascara] * mascara_validos[:, None]
        probabilidadClase = pixeles_validos / pixeles_totales
        priori.append(probabilidadClase)

        plt.figure()
        plt.imshow(imagen_final)
        plt.title("Imagen con Máscara Aplicada")
        plt.axis('off')
        plt.show()
    else:
        print("No se encontraron píxeles en el área seleccionada.")

# Cargar la imagen con filtro gaussiano con radio de 2
imagen = io.imread('Ent2.jpg')
filas, columnas, _ = imagen.shape
pixeles_totales = imagen.shape[0] * imagen.shape[1]
imagen_gris = np.zeros((filas, columnas), dtype=np.uint8)

for k in range(3):
    fig, ax = plt.subplots()
    ax.imshow(imagen)
    ax.set_title("Selecciona el área de interés")
    selector = PolygonSelector(ax, on_select)
    plt.show()

    print("______________________________________________________")

pixeles_fondo = pixeles_totales - acumulado
print(f"Total de píxeles del fondo: {pixeles_fondo}")
priori.append(pixeles_fondo / pixeles_totales)
print(priori)

# Listas para almacenar imágenes y sus nombres
images = []
file_names = []
blurred_images = []

# Directorio de imágenes
images_directory = "Prueba"

# Cargar imágenes del directorio y aplicar filtro gaussiano con radio =4
for file_name in sorted(os.listdir(images_directory)):
    file_path = os.path.join(images_directory, file_name)
    try:
        image = io.imread(file_path)
        images.append(image)
        file_names.append(file_name)

        image_pil = Image.fromarray(image)
        blurred_image = image_pil.filter(ImageFilter.GaussianBlur(radius=4))
        blurred_images.append(np.array(blurred_image))

    except Exception as e:
        print(f"Error al cargar la imagen {file_name}: {e}")

# Aplicar el proceso de pixel a pixel a cada una de las imagenes preprocesadas de la lista
for imagen, file_name in zip(blurred_images, file_names):
    print(f"Procesando imagen desenfocada: {file_name}")

    # Inicializar imagen de salida en escala de grises
    filas, columnas, _ = imagen.shape
    imagen_gris = np.zeros((filas, columnas), dtype=np.uint8)

    # Recorrer los píxeles de la imagen
    for i in range(imagen.shape[0]):  # Filas
        for j in range(imagen.shape[1]):  # Columnas
            pixel = imagen[i, j, :3]  # Obtener el valor RGB del píxel

            # Probabilidades, algunos datos se calcularon en otro programa
            Y_platano = calcular_Yk(pixel, banmed, banvar, priori[0])
            Y_chiles = calcular_Yk(pixel, chmed, chvar, priori[1])
            Y_huevo = calcular_Yk(pixel, medias[4], medias[5], priori[2])
            Y_fondo = calcular_Yk(pixel, fondomed, fondovar, priori[3])


            Y_vals = [Y_platano, Y_chiles, Y_huevo, Y_fondo]
            max_Yk = np.argmax(Y_vals)


            if max_Yk == 0:
                imagen_gris[i, j] = 170  # Clase plátano
            elif max_Yk == 1:
                imagen_gris[i, j] = 85  # Clase huevo
            elif max_Yk == 2:
                imagen_gris[i, j] = 255  # Clase chiles
            else:
                imagen_gris[i, j] = 0  # Fondo

    # Resultado
    plt.figure()
    plt.imshow(imagen_gris, cmap='gray')
    plt.title(f"Imagen Clasificada: {file_name}")
    plt.axis('off')
    plt.show()

print(medias)
print(medias[0][0])
print(medias[1] * 2)

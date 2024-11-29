import os
import json
from PIL import Image, ImageFilter
from skimage import io, color, filters, measure, morphology
import numpy as np
from skimage.draw import polygon
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import regionprops, label
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from skimage.feature import canny
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.morphology import remove_small_holes

## Entrenamiento

# Función para extraer características de una región etiquetada
def extract_features(image, mask):
    gray_image = np.array(image.convert("L"))
    masked_image = np.where(mask, gray_image, 0)

    region_area = mask.sum()

    glcm = graycomatrix(masked_image, distances=[1, 2, 3, 4], angles=[0, np.pi / 4, np.pi / 6, np.pi / 3, np.pi / 2, 2 * np.pi / 3, 5 * np.pi / 6, 3 * np.pi / 4],
                        levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast').mean()
    energy = graycoprops(glcm, 'energy').mean()
    correlation = graycoprops(glcm, 'correlation').mean()
    dissimilarity = graycoprops(glcm, 'dissimilarity').mean()
    asm = graycoprops(glcm, 'ASM').mean()

    mean_intensity = masked_image[mask].mean()
    std_intensity = masked_image[mask].std()

    labeled_mask = label(mask)
    props = regionprops(labeled_mask)
    eccentricity = props[0].eccentricity if props else 0
    solidity = props[0].solidity if props else 0

    features = [region_area, contrast, energy, correlation, mean_intensity,
                std_intensity, eccentricity, solidity,dissimilarity,asm]
    return features


def load_data(data_dir):
    X, y = [], []
    for file_name in os.listdir(data_dir):
        if file_name.endswith(".json"):
            with open(os.path.join(data_dir, file_name)) as f:
                data = json.load(f)
            image_path = os.path.join(data_dir, data["imagePath"])
            image = Image.open(image_path)

            for shape in data["shapes"]:
                label_name = shape["label"]
                points = shape["points"]

                mask = np.zeros((image.height, image.width), dtype=bool)
                poly = np.array(points, dtype=np.int32)
                rr, cc = polygon(poly[:, 1], poly[:, 0], mask.shape)
                mask[rr, cc] = True

                features = extract_features(image, mask)
                X.append(features)
                y.append(label_name)

    return X, y


train_dir = "Entrenamiento"
test_dir = "Prueba"

X_train, y_train = load_data(train_dir)
X_test, y_test = load_data(test_dir)

svm = SVC(kernel='linear')
svm.fit(X_train, y_train)

y_pred = svm.predict(X_test)
print(classification_report(y_test, y_pred))

precision_global = accuracy_score(y_test, y_pred)
print(f"Precisión global: {precision_global:.2f}")


#################################################

## Deteccion
image_path = "t7.jpg"
image = Image.open(image_path)

gray_image = np.array(image.convert("L"))


smoothed_image = filters.gaussian(gray_image, sigma=4)


threshold_value = .5 # Umbral
binary_mask = smoothed_image < threshold_value


cleaned_mask = morphology.remove_small_objects(binary_mask, min_size=900)
cleaned_mask = remove_small_holes(cleaned_mask, area_threshold=500)
cleaned_mask = clear_border(cleaned_mask)


edges = canny(cleaned_mask.astype(float), sigma=2)


labeled_regions = measure.label(cleaned_mask)
properties = measure.regionprops(labeled_regions)


valid_regions = []
for prop in properties:
    area = prop.area
    eccentricity = prop.eccentricity

    if 1000 < area < 12000:
        valid_regions.append(prop)


fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(image)


for region in valid_regions:
    minr, minc, maxr, maxc = region.bbox
    ax.add_patch(plt.Rectangle((minc, minr), maxc - minc, maxr - minr,
                               edgecolor='red', facecolor='none', linewidth=2))

    # Extraer características y predecir clase
    mask = labeled_regions == region.label
    features = extract_features(image, mask)
    predicted_class = svm.predict([features])[0]
    print(predicted_class)

    # Asignar color según clase
    if predicted_class == "tumor":
        color = 'red'
    elif predicted_class == "sano":
        color = 'blue'
    else:
        color = 'green'


    ax.add_patch(plt.Rectangle((minc, minr), maxc - minc, maxr - minr,
                               edgecolor=color, facecolor='none', linewidth=2))

ax.set_title(f"Células detectadas: {len(valid_regions)}")
ax.axis("off")
plt.show()
# Clasificación supervisada con 2000 imágenes de gatos y perros

import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix

# --- 1. Cargar imágenes y etiquetas ---
dataset_path = 'imagenes/dataset/train'
imagenes = []
labels = []

for file in os.listdir(dataset_path):
    if file.startswith('cat'):
        labels.append(0)
        imagenes.append(os.path.join(dataset_path, file))
    elif file.startswith('dog'):
        labels.append(1)
        imagenes.append(os.path.join(dataset_path, file))

y = np.array(labels)

# Convertir imágenes a vectores (grayscale 64x64)
X = []
for img_path in imagenes:
    img = imread(img_path, as_gray=True)          # Escala de grises
    img_resized = resize(img, (64,64))           # Redimensionar
    X.append(img_resized.flatten())               # Convertir a vector
X = np.array(X)

print("Número de imágenes:", X.shape[0])
print("Número de características por imagen:", X.shape[1])

# --- 2. Dividir dataset ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# --- 3. Entrenar modelos ---

# a) k-NN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
pred_knn = knn.predict(X_test)

# b) Regresión logística
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train, y_train)
pred_logreg = logreg.predict(X_test)

# c) Árbol de decisión
tree = DecisionTreeClassifier(random_state=42)
tree.fit(X_train, y_train)
pred_tree = tree.predict(X_test)

# --- 4. Evaluar resultados ---
print("\n=== k-NN ===")
print(confusion_matrix(y_test, pred_knn))
print(classification_report(y_test, pred_knn))

print("\n=== Regresión Logística ===")
print(confusion_matrix(y_test, pred_logreg))
print(classification_report(y_test, pred_logreg))

print("\n=== Árbol de Decisión ===")
print(confusion_matrix(y_test, pred_tree))
print(classification_report(y_test, pred_tree))

# --- 5. Mostrar algunas predicciones ---
plt.figure(figsize=(10,4))
for i in range(10):
    plt.subplot(2,5,i+1)
    plt.imshow(X_test[i].reshape(64,64), cmap='gray')
    plt.title(f"Real: {'Gato' if y_test[i]==0 else 'Perro'}\nPred: {'Gato' if pred_knn[i]==0 else 'Perro'}")
    plt.axis('off')
plt.show()

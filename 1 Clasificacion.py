#Este programa en Python utiliza la biblioteca scikit-learn para entrenar y evaluar un modelo de clasificación 
#con el algoritmo K-Nearest Neighbors (KNN) usando el famoso conjunto de datos Iris.

from sklearn.datasets import load_iris #carga el conjunto de datos Iris, dataset contiene características de flores de tres especies distintas.
from sklearn.model_selection import train_test_split #Importa la función para dividir los datos en conjuntos de entrenamiento y prueba.
from sklearn.neighbors import KNeighborsClassifier #Importa el clasificador KNN, se basa en la proximidad de los datos para hacer predicciones.
from sklearn.metrics import classification_report, confusion_matrix #importa una matriz de confusión y un reporte de clasificación

X, y = load_iris(return_X_y=True)
#Carga el dataset
# X matriz con las características (longitud y ancho de sépalos y pétalos).
# Y vector con las etiquetas (0, 1, 2) que representan las especies.

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)
#Divide los datos en entrenamiento (70%) y prueba (30%).

clf = KNeighborsClassifier(n_neighbors=3)
#Crea un clasificador KNN con , es decir, el modelo se basa en los 3 vecinos más cercanos para decidir la clase.

clf.fit(X_tr, y_tr) # Entrena el modelo
y_pred = clf.predict(X_te) #Usa el modelo entrenado para predecir las etiquetas del conjunto de prueba.
print(confusion_matrix(y_te, y_pred))
print(classification_report(y_te, y_pred))

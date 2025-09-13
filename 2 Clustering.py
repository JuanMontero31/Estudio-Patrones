#Este programa utiliza el algoritmo de K-Means para realizar clustering (agrupamiento no supervisado) sobre datos generados artificialmente. 

from sklearn.datasets import make_blobs #generar datos sintéticos agrupados en "blobs" (grupos). 
from sklearn.cluster import KMeans #Importa el algoritmo de K-Means, que agrupa datos en clústeres según su cercanía.
from sklearn.metrics import silhouette_score #Importa una métrica que evalúa qué tan bien definidos están los clústeres.
import matplotlib.pyplot as plt


X, _ = make_blobs(n_samples=300, centers=4, random_state=42)
#Genera 300 muestras distribuidas en 4 centros (clústeres).
# X: contiene las coordenadas de los puntos.
# _: ignora las etiquetas verdaderas, ya que K-Means es no supervisado.

kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(X)
labels = kmeans.labels_
centers = kmeans.cluster_centers_
#Crea y ajusta el modelo K-Means para encontrar 4 clústeres en los datos.

labels = kmeans.labels_
print("Silhouette score:", silhouette_score(X, labels))

# Visualizar
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50)
plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='X', s=200, label='Centroides')
plt.title("Clustering con K-Means")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.grid(True)
plt.show()


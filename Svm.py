#SCRIPT 1
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_blobs
import matplotlib

# ====================================
# 1. Generar 100 muestras con nueva distribución
# ====================================
# Cambiamos la distribución: más dispersión y centros separados
X, y = make_blobs(n_samples=100, n_features=2, centers=2, cluster_std=2.0, random_state=21)

# Lista de kernels a probar
kernels = ['linear', 'poly', 'rbf', 'sigmoid']

# Creamos una figura con subplots para comparar
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()  # aplanamos el array de ejes para recorrer fácilmente

# Colores para los puntos de las clases
cmap_points = matplotlib.colors.ListedColormap(['k', 'g'])
cmap_regions = matplotlib.colors.ListedColormap(['r', 'y'])

for i, kernel in enumerate(kernels):
    ax = axes[i]
    
    # ====================================
    # 2. Crear y entrenar el modelo con cada kernel
    # ====================================
    clf = svm.SVC(kernel=kernel, C=100, gamma='scale')
    clf.fit(X, y)
    
    # ====================================
    # 3. Crear el mesh para graficar fronteras
    # ====================================
    xlim = (X[:, 0].min() - 1, X[:, 0].max() + 1)
    ylim = (X[:, 1].min() - 1, X[:, 1].max() + 1)
    xx = np.linspace(xlim[0], xlim[1], 200)
    yy = np.linspace(ylim[0], ylim[1], 200)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z_pred = clf.predict(xy).reshape(XX.shape)
    
    # ====================================
    # 4. Graficar
    # ====================================
    ax.pcolormesh(XX, YY, Z_pred, cmap=cmap_regions, alpha=0.1)
    ax.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=cmap_points, edgecolors='k')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_title(f"SVM con kernel = {kernel}")
    ax.grid()

plt.tight_layout()
plt.show()



#SCRIPT 2
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_circles
import matplotlib

# ====================================
# 1. Crear datos NO linealmente separables
# ====================================
# Usamos make_circles para generar datos en forma de anillo
X, y = make_circles(n_samples=100, factor=0.5, noise=0.05, random_state=42)

# Lista de kernels a probar
kernels = ['linear', 'poly', 'rbf', 'sigmoid']

# Colores
cmap_points = matplotlib.colors.ListedColormap(['k', 'g'])
cmap_regions = matplotlib.colors.ListedColormap(['r', 'y'])

# Crear figura
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()

for i, kernel in enumerate(kernels):
    ax = axes[i]

    # ====================================
    # 2. Entrenar modelo SVM con cada kernel
    # ====================================
    clf = svm.SVC(kernel=kernel, C=10, gamma='scale')
    clf.fit(X, y)

    # ====================================
    # 3. Crear mesh para graficar
    # ====================================
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    xy = np.c_[xx.ravel(), yy.ravel()]
    Z_pred = clf.predict(xy).reshape(xx.shape)

    # ====================================
    # 4. Graficar resultados
    # ====================================
    ax.pcolormesh(xx, yy, Z_pred, cmap=cmap_regions, alpha=0.1)
    ax.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=cmap_points, edgecolors='k')
    ax.set_title(f"SVM con kernel = {kernel}")
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.grid()

plt.tight_layout()
plt.show()




#SCRIPT 3
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ====================================
# 1. Cargar dataset Iris
# ====================================
iris = load_iris()
# Usamos solo 2 características para poder graficar
X = iris.data[:, 2:4]  # petal length y petal width
y = iris.target        # 3 clases

# Normalizar para mejorar el rendimiento
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# ====================================
# 2. Probar todos los kernels
# ====================================
kernels = ['linear', 'poly', 'rbf', 'sigmoid']

# Crear figure con subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()

for i, kernel in enumerate(kernels):
    ax = axes[i]
    # Entrenar modelo con cada kernel
    clf = svm.SVC(kernel=kernel, C=1.0, gamma='scale', decision_function_shape='ovr')
    clf.fit(X_train, y_train)
    
    # ==========================
    # Crear grid para graficar
    # ==========================
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))
    
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # ==========================
    # Graficar resultados
    # ==========================
    ax.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k')
    ax.set_title(f"SVM con kernel = {kernel}")
    ax.set_xlabel("Petal length (normalizado)")
    ax.set_ylabel("Petal width (normalizado)")
    ax.grid()

plt.tight_layout()
plt.show()

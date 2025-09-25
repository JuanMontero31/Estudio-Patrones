import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. Cargar el dataset Iris
iris = load_iris()
X = iris.data[:, :2]  # Solo usamos las primeras dos características para graficar
y = iris.target
classes = np.unique(y)

# 2. Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 3. Modelo 1: Gaussian Naive Bayes
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred_gnb = gnb.predict(X_test)
acc_gnb = accuracy_score(y_test, y_pred_gnb)

# 4. Modelo 2: Gaussian Mixture Model
gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=42)
gmm.fit(X_train)
y_pred_gmm = gmm.predict(X_test)
acc_gmm = accuracy_score(y_test, y_pred_gmm)

# 5. Implementación manual del Teorema de Bayes
# Calcular media y varianza por clase
mean_var = {}
for c in classes:
    X_c = X_train[y_train == c]
    mean_var[c] = (X_c.mean(axis=0), X_c.var(axis=0))

# Calcular prior
priors = {c: np.mean(y_train == c) for c in classes}

# Función densidad normal (para 2 características)
def gaussian_likelihood(x, mu, var):
    eps = 1e-6
    coeff = 1 / np.sqrt(2 * np.pi * var + eps)
    exponent = -((x - mu) ** 2) / (2 * var + eps)
    return np.prod(coeff * np.exp(exponent))

# Predicción Bayes manual
y_pred_manual = []
for x in X_test:
    posteriors = []
    for c in classes:
        mu, var = mean_var[c]
        likelihood = gaussian_likelihood(x, mu, var)
        posterior = likelihood * priors[c]
        posteriors.append(posterior)
    y_pred_manual.append(np.argmax(posteriors))
y_pred_manual = np.array(y_pred_manual)
acc_manual = accuracy_score(y_test, y_pred_manual)

# 6. Visualización de fronteras de decisión
def plot_decision_boundaries(model, X, y, title):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    # Para GMM y GNB usamos predict directo, para manual hacemos predicción manual
    if title == "Frontera de decisión - Bayes Manual":
        Z = []
        for point in np.c_[xx.ravel(), yy.ravel()]:
            posteriors = []
            for c in classes:
                mu, var = mean_var[c]
                likelihood = gaussian_likelihood(point, mu, var)
                posterior = likelihood * priors[c]
                posteriors.append(posterior)
            Z.append(np.argmax(posteriors))
        Z = np.array(Z).reshape(xx.shape)
    else:
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

    plt.figure(figsize=(7, 5))
    plt.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.coolwarm)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=30, edgecolor='k', cmap=plt.cm.coolwarm)
    plt.title(title)
    plt.xlabel("Sepal length")
    plt.ylabel("Sepal width")
    plt.show()

# Graficamos las regiones de decisión
plot_decision_boundaries(gnb, X_train, y_train, "Frontera de decisión - Gaussian Naive Bayes")
plot_decision_boundaries(gmm, X_train, y_train, "Frontera de decisión - Gaussian Mixture Model")
plot_decision_boundaries(None, X_train, y_train, "Frontera de decisión - Bayes Manual")

# 7. Resultados
print("=== RESULTADOS ===")
print(f"Accuracy - Gaussian Naive Bayes: {acc_gnb:.2f}")
print(f"Accuracy - Gaussian Mixture Model: {acc_gmm:.2f}")
print(f"Accuracy - Bayes Manual: {acc_manual:.2f}")

import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. Cargar dataset
iris = load_iris()
X = iris.data
y = iris.target

# 2. Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Crear y entrenar el modelo
arbol = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=42)
arbol.fit(X_train, y_train)

# 4. Evaluar modelo
y_pred = arbol.predict(X_test)
print("Exactitud del árbol de decisión:", accuracy_score(y_test, y_pred))

# 5. Visualizar árbol
plt.figure(figsize=(12, 6))
plot_tree(arbol, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
plt.title("Árbol de Decisión - Dataset Iris")
plt.show()

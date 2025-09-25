from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report

# 1. Generar datos simulados (100 ejemplos, 5 características)
#   (simularemos que la clase 1 = spam, clase 0 = no spam)
X, y = make_classification(n_samples=100, n_features=5, n_informative=3,
                           n_redundant=0, n_classes=2, random_state=42)

# 2. Dividir en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Entrenar un modelo sencillo (árbol de decisión)
modelo = DecisionTreeClassifier(random_state=42)
modelo.fit(X_train, y_train)

# 4. Hacer predicciones sobre el conjunto de prueba
y_pred = modelo.predict(X_test)

# 5. Calcular matriz de confusión
matriz = confusion_matrix(y_test, y_pred)
print("Matriz de Confusión:\n", matriz)

# 6. Reporte de métricas
print("\nReporte de Clasificación:\n")
print(classification_report(y_test, y_pred, target_names=["No Spam", "Spam"]))

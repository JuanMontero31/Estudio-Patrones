from sklearn.datasets import make_regression #Genera datos sintéticos para problemas de regresión
from sklearn.linear_model import LinearRegression #Importa el modelo de regresión lineal, ajusta una línea recta para predecir valores continuos.
from sklearn.model_selection import train_test_split #Permite dividir los datos en conjuntos de entrenamiento y prueba.
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt


X, y = make_regression(n_samples=150, n_features=1, noise=10, random_state=42)
#Crea 150 muestras con una sola característica (n_features=1) y añade ruido aleatorio (noise=10) para simular datos reales.

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)
reg = LinearRegression().fit(X_tr, y_tr)
#Crea y ajusta el modelo de regresión lineal usando los datos de entrenamiento.

y_pred = reg.predict(X_te)
#Predice los valores de salida para el conjunto de prueba.

print("MSE:", mean_squared_error(y_te, y_pred))
print("R2:", r2_score(y_te, y_pred))

# Graficar
plt.figure(figsize=(8, 6))
plt.scatter(X_te, y_te, color='blue', label='Datos reales')
plt.plot(X_te, y_pred, color='red', linewidth=2, label='Línea de regresión')
plt.title("Regresión Lineal")
plt.xlabel("Feature")
plt.ylabel("Target")
plt.legend()
plt.grid(True)
plt.show()

#Si los puntos están cerca de la línea, significa que el modelo está haciendo buenas predicciones.

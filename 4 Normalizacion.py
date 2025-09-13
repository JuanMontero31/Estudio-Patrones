import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Simulamos un dataset con 2 características: peso (30-300g) y color (0-1)
X = np.array([[30, 0.8],
              [150, 0.6],
              [300, 0.9],
              [200, 0.7]])

scaler = MinMaxScaler()
X_norm = scaler.fit_transform(X)

print("Datos originales:\n", X)
print("\nDatos normalizados:\n", X_norm)

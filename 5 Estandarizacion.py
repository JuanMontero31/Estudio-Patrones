import numpy as np
from sklearn.preprocessing import StandardScaler

# Definir los datos
X = np.array([[30, 0.8],
              [150, 0.6],
              [300, 0.9],
              [200, 0.7]])

# Aplicar estandarización
scaler_std = StandardScaler()
X_std = scaler_std.fit_transform(X)

# Mostrar resultados
print("Datos originales:\n", X)
print("\nDatos estandarizados:\n", X_std)
print("\nMedia de cada característica:", X_std.mean(axis=0))
print("Desviación estándar de cada característica:", X_std.std(axis=0))
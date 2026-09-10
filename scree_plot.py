import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Leer los datos
datos = pd.read_csv("datos_PCA_30_empleados_5_indicadores.csv")

# Quitar la columna Empleado
X = datos.drop(columns=["Empleado"])

# Calcular matriz de varianzas y covarianzas
matriz_cov = X.cov()

# Calcular valores y vectores propios
valores_propios, vectores_propios = np.linalg.eigh(matriz_cov)

# Ordenar los valores propios de mayor a menor
valores_propios = np.sort(valores_propios)[::-1]

# Calcular porcentaje de varianza explicada
porcentaje = valores_propios / valores_propios.sum() * 100

print("Valores propios:")
print(valores_propios)

print("\nPorcentaje de varianza explicada:")
print(porcentaje)

# Crear scree plot
componentes = range(1, len(valores_propios) + 1)

plt.plot(componentes, valores_propios, marker="o")
plt.xlabel("Componente principal")
plt.ylabel("Valor propio")
plt.title("Scree Plot")
plt.xticks(componentes)
plt.grid()

plt.savefig("scree_plot.png")
plt.show()

Situacion problema - Equipo 5
Reduccion de dimensionalidad y clasificacion de empleados mediante PCA
(matriz de varianzas y covarianzas, valores y vectores propios)
 
Entrada: datos_PCA_30_empleados_5_indicadores.csv
Salidas: matriz de covarianza, valores/vectores propios, nuevo indicador,
         clasificacion de empleados, scree plot y grafica de varianza explicada.
"""
 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
 
# ---------------------------------------------------------------
# 1) Leer los datos y separar el identificador del empleado
# ---------------------------------------------------------------
datos = pd.read_csv("datos_PCA_30_empleados_5_indicadores.csv")
X = datos.drop(columns=["Empleado"])
indicadores = list(X.columns)
 
# ---------------------------------------------------------------
# Paso 1: Matriz de varianzas y covarianzas
# ---------------------------------------------------------------
medias = X.mean()
matriz_cov = X.cov()  # matriz S (5x5), muestral (n-1)
 
print("Medias de los indicadores:")
print(medias, "\n")
print("Matriz de varianzas y covarianzas (S):")
print(matriz_cov, "\n")
 
# ---------------------------------------------------------------
# Paso 2: Valores y vectores propios de la matriz S
# ---------------------------------------------------------------
valores_propios, vectores_propios = np.linalg.eigh(matriz_cov.values)
 
# Ordenar de mayor a menor valor propio
orden = np.argsort(valores_propios)[::-1]
valores_propios = valores_propios[orden]
vectores_propios = vectores_propios[:, orden]
 
# Fijar el signo de cada vector propio (convencion: la entrada de
# mayor magnitud se define positiva) para que el resultado sea reproducible
for i in range(vectores_propios.shape[1]):
    idx_max = np.argmax(np.abs(vectores_propios[:, i]))
    if vectores_propios[idx_max, i] < 0:
        vectores_propios[:, i] *= -1
 
porcentaje = valores_propios / valores_propios.sum() * 100
 
print("Valores propios (orden descendente):")
print(np.round(valores_propios, 4))
print("\nPorcentaje de varianza explicada:")
print(np.round(porcentaje, 3))
 
# ---------------------------------------------------------------
# Paso 3: Mayor valor propio y su vector propio asociado
# ---------------------------------------------------------------
lambda_1 = valores_propios[0]
v1 = vectores_propios[:, 0]
 
print(f"\nMayor valor propio (lambda_1): {lambda_1:.4f}")
print("Vector propio asociado (v1):")
for ind, coef in zip(indicadores, v1):
    print(f"  {ind}: {coef:.4f}")
 
# ---------------------------------------------------------------
# Paso 4: Nuevo indicador y clasificacion de los empleados
# ---------------------------------------------------------------
X_centrado = X - medias
indicador_nuevo = X_centrado.values @ v1  # combinacion lineal (proyeccion sobre v1)
 
clasificacion = pd.DataFrame({
    "Empleado": datos["Empleado"],
    "Indicador_PC1": indicador_nuevo
}).sort_values("Indicador_PC1", ascending=False).reset_index(drop=True)
clasificacion["Ranking"] = np.arange(1, len(clasificacion) + 1)
 
print("\nClasificacion de empleados (de mayor a menor desempeno):")
print(clasificacion.to_string(index=False))
 
# ---------------------------------------------------------------
# Error de representar los datos en una sola dimension (PC1)
# ---------------------------------------------------------------
error_varianza = 1 - lambda_1 / valores_propios.sum()
 
Z1 = X_centrado.values @ vectores_propios[:, :1]           # proyeccion 5D -> 1D
X_reconstruido = Z1 @ vectores_propios[:, :1].T             # reconstruccion 1D -> 5D
error_frobenius = (
    np.linalg.norm(X_centrado.values - X_reconstruido, "fro") ** 2
    / np.linalg.norm(X_centrado.values, "fro") ** 2
)
 
print(f"\nError (varianza no explicada por PC1): {error_varianza * 100:.3f}%")
print(f"Error de reconstruccion (Frobenius) con 1 componente: {error_frobenius * 100:.3f}%")
 
# ---------------------------------------------------------------
# Graficas: Scree plot y varianza explicada acumulada
# ---------------------------------------------------------------
componentes = range(1, len(valores_propios) + 1)
 
plt.figure(figsize=(6.5, 4.2))
plt.plot(componentes, valores_propios, marker="o")
plt.xlabel("Componente principal")
plt.ylabel("Valor propio")
plt.title("Scree Plot")
plt.xticks(list(componentes))
plt.grid()
plt.tight_layout()
plt.savefig("scree_plot.png", dpi=200)
plt.show()
 
plt.figure(figsize=(6.5, 4.2))
plt.bar(componentes, porcentaje, label="% Varianza explicada")
plt.plot(componentes, np.cumsum(porcentaje), color="red", marker="o", label="% acumulado")
plt.xlabel("Componente principal")
plt.ylabel("% de varianza")
plt.title("Varianza explicada por componente principal")
plt.xticks(list(componentes))
plt.legend()
plt.tight_layout()
plt.savefig("varianza_explicada.png", dpi=200)
plt.show()
 
# Exportar resultados a CSV
matriz_cov.round(4).to_csv("matriz_covarianza.csv")
clasificacion.to_csv("clasificacion_empleados.csv", index=False)


"""

Parcial 4: Cadena de Markov

Autores:
    - Sergio Orellana 221122
    - Rodrigo Mansilla 22611
    - Nelson García 22434
    - Carlos Valladares 221164
    - Gabriel Paz 221087

Fecha: 2024-05-15

Problema 3: 
Menganita quiere impresionar a Chispudito con sus conocimientos de probabilidad. Ella decide utilizar la máquina de Galton con 6 niveles de clavos y 7 casilleros donde caen las pelotitas para mostrárselo a Chispudito.
  
Para ayudarle a Menganita. debe encontrar la probabilidad de que la pelotita caiga en cada uno de los casilleros.

Programa:
    - Implementar la cadena de Markov para la máquina de Galton.
    - Simular la máquina de Galton con la Ley de los Grandes Números.
    - Comparar los resultados de la simulación con la cadena de Markov.

"""

# Se importan las librerías necesarias
import numpy as np
import matplotlib.pyplot as plt

# Cadena de Markov
print("Cadena de Markov")

P = np.array([
    #0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27
    [0, 0.5, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #Clavo 0
    [0, 0, 0, 0.5, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #Clavo 1
    [0, 0, 0, 0, 0.5, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #Clavo 2
    [0, 0, 0, 0, 0, 0, 0.5, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #Clavo 3
    [0, 0, 0, 0, 0, 0, 0, 0.5, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #Clavo 4
    [0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #Clavo 5
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #Clavo 6
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #Clavo 7
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #Clavo 8
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #Clavo 9
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #Clavo 10
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #Clavo 11
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0], #Clavo 12
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0.5, 0, 0, 0, 0, 0, 0, 0, 0], #Clavo 13
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0.5, 0, 0, 0, 0, 0, 0, 0], #Clavo 14
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0.5, 0, 0, 0, 0, 0], #Clavo 15
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0.5, 0, 0, 0, 0], #Clavo 16
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0.5, 0, 0, 0], #Clavo 17
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0.5, 0, 0], #Clavo 18
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0.5, 0], #Clavo 19
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0.5],  #Clavo 20
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], #Casillero 21
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], #Casillero 22
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], #Casillero 23
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], #Casillero 24
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], #Casillero 25
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], #Casillero 26
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1] #Casillero 27
])
"""
P es la matriz de transición, donde cada fila representa un clavo y cada columna representa la probabilidad de transición a otros clavos o casilleros.
"""

# Inicializar un vector de estado con igual probabilidad en el primer clavo
estado = np.zeros(P.shape[0])
estado[0] = 1  # Asumiendo que la pelota comienza en el clavo 0
"""
estado es un vector que representa la probabilidad de estar en cada clavo al comienzo.

Inicializamos este vector con 1 en el primer clavo.
"""

# Iterar la matriz de transición muchas veces
for _ in range(1000):  # Número grande para asegurar convergencia
    estado = np.dot(estado, P)

"""
Multiplicamos el vector estado por la matriz de transición P repetidamente para simular el proceso de la cadena de Markov hasta alcanzar un estado estable.
"""

# Imprimir las probabilidades
print("Probabilidades de cada casillero:")
for i, prob in enumerate(estado[-7:], 21):  # Los últimos 7 estados son los casilleros
    print(f"Casillero {i}: {prob:.5f}")
"""
Imprimimos las probabilidades de que la pelota caiga en cada casillero, que son los últimos 7 estados de la matriz.
"""

# Extraer las probabilidades de los casilleros
probabilidades_casilleros = estado[-7:]

# Nombres de los casilleros
casilleros = [f"Casillero {i}" for i in range(21, 28)]

# Crear la gráfica de barras
plt.figure(figsize=(10, 6))
bars = plt.bar(casilleros, probabilidades_casilleros, color='red', alpha=0.6, label='Cadena de Markov')

# Agregar las etiquetas de probabilidad encima de cada barra
for bar, prob in zip(bars, probabilidades_casilleros):
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f"{prob:.5f}", ha='center', va='bottom')

plt.xlabel('Casilleros')
plt.ylabel('Probabilidad')
plt.title('Distribución de Probabilidades en los Casilleros')
plt.ylim(0, max(probabilidades_casilleros) * 1.1)  # Ajustar límite superior del eje Y
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend()
plt.show()
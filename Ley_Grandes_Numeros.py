"""

Parcial 4: Ley de los Grandes Números

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
import random
import matplotlib.pyplot as plt
import networkx as nx

# Función para simular la máquina de Galton usando la ley de los grandes números
def simulate_galton_board_lgn(n_levels, n_balls):
    counts = [0] * (n_levels + 1)
    # counts: Inicializa una lista de conteo para cada posición de los casilleros.
    
    # Realiza la simulación de la máquina de Galton
    for _ in range(n_balls):
        # position: Inicializa la posición en 0 para cada bola.
        position = 0
        # Realiza el recorrido de la bola por cada nivel de la máquina de Galton.
        for _ in range(n_levels):
            # Se elige aleatoriamente si la bola se mueve a la izquierda o a la derecha.
            position += random.choice([0, 1])

        # Se incrementa el conteo de la posición final de la bola.
        counts[position] += 1
    
    # Calcula las probabilidades de cada casillero.
    probabilities = [count / n_balls for count in counts]

    # Retorna las probabilidades calculadas.
    return probabilities

# Función principal
def main():
    # Niveles de la máquina de Galton y cantidad de bolas
    n_levels = 6
    n_balls = 1000000

    # Simulación de la máquina de Galton usando la Ley de los Grandes Números obteniendo las probabilidades
    probabilities_lgn = simulate_galton_board_lgn(n_levels, n_balls)
    
    print("\nProbabilidades usando la Ley de los Grandes Números:")
    # Imprime las probabilidades de cada casillero.
    for i, prob in enumerate(probabilities_lgn):
        print(f'Casillero {i + 21}: {prob:.5f}')

    # Crear los nombres de los casilleros
    bins = [f'Casillero {i + 21}' for i in range(n_levels + 1)]

    # Crear el gráfico de barras
    plt.figure(figsize=(12, 6))
    bars = plt.bar(bins, probabilities_lgn, alpha=0.6, label='Ley de los Grandes Números')
    plt.xlabel('Casilleros')
    plt.ylabel('Probabilidad')
    plt.title('Distribución de Probabilidades en los Casilleros')
    plt.legend()

    # Agregar anotaciones encima de cada barra
    for bar, prob in zip(bars, probabilities_lgn):
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, f'{prob:.5f}', ha='center', va='bottom') 

    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

# Punto de entrada del programa
if __name__ == "__main__":
    main()

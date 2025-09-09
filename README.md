# Laboratorio-2---convolucion-y-correlacion
En este laboratorio se desarrollo a partir de unas señales la convolución, correlación cruzada y frecuencia de Nyquist 

## Introducción
En este laboratorio  se observo cómo se comportan las señales tanto en el tiempo como en la frecuencia. Lo haremos aplicando tres técnicas fundamentales: la convolución, la correlación y la transformada de Fourier. Además del análisis de una señal electrooculograma (EOG).

## Parte A
<p align="center">
    <img <img width="1155" height="1000" alt="image" src="https://github.com/user-attachments/assets/b8c7986e-e425-430a-8730-fa0a47cef2b5" />
</p>

## Señal y[n] resultante de la convolución

```python
import numpy as np
import matplotlib.pyplot as plt

# EJEMPLO 1 - Karen

h1 = [5, 6, 0, 0, 8, 0, 3]
x1 = [1, 0, 7, 2, 2, 5, 5, 2, 2, 3]
y1 = np.convolve(x1, h1)

# EJEMPLO 2 - Antonia

h2 = [5, 6, 0, 0, 8, 4, 3]
x2 = [1, 0, 2, 5, 0, 6, 0, 9, 7, 0]
y2 = np.convolve(x2, h2)

# EJEMPLO 3 - Laura

h3 = [5, 6, 0, 0, 8, 2, 8]
x3 = [1, 0, 7, 2, 6, 4, 3, 8, 0, 7]
y3 = np.convolve(x3, h3)

# GRAFICAR RESULTADOS

fig, axs = plt.subplots(3, 3, figsize=(14,10))

# Karen
axs[0,0].stem(range(len(x1)), x1, basefmt=" ")
axs[0,0].set_title("Karen : x[n]")
axs[0,1].stem(range(len(h1)), h1, basefmt=" ")
axs[0,1].set_title("Karen: h[n]")
axs[0,2].stem(range(len(y1)), y1, basefmt=" ")
axs[0,2].set_title("Karen: y[n] = x*h")

# Antonia
axs[1,0].stem(range(len(x2)), x2, basefmt=" ")
axs[1,0].set_title("Antonia: x[n]")
axs[1,1].stem(range(len(h2)), h2, basefmt=" ")
axs[1,1].set_title("Antonia: h[n]")
axs[1,2].stem(range(len(y2)), y2, basefmt=" ")
axs[1,2].set_title("Antonia: y[n] = x*h")

# Laura
axs[2,0].stem(range(len(x3)), x3, basefmt=" ")
axs[2,0].set_title("Laura: x[n]")
axs[2,1].stem(range(len(h3)), h3, basefmt=" ")
axs[2,1].set_title("Laura: h[n]")
axs[2,2].stem(range(len(y3)), y3, basefmt=" ")
axs[2,2].set_title("Laura: y[n] = x*h")

plt.tight_layout()
plt.show()

# Mostrar valores numéricos
print("Karen - y[n] =", y1.tolist())
print("Antonia - y[n] =", y2.tolist())
print("Laura - y[n] =", y3.tolist())
```

<p align="center">
    <img <img width="1389" height="989" alt="image" src="https://github.com/user-attachments/assets/300a745d-cdfe-4ab3-a412-432d85f5e715" />
</p>

Este código en Python calcula la convolución discreta entre dos señales utilizando la función np.convolve() de NumPy. Primero, se definen dos listas, h y x, que representan la respuesta al impulso de un sistema y una señal de entrada, respectivamente. Luego, se aplica la convolución entre estas dos señales usando np.convolve(x, h, mode='full'), lo que genera una nueva señal y cuya longitud es la suma de las longitudes de x y h menos uno. 

La convolución es una operación fundamental en procesamiento de señales, ya que permite analizar cómo una señal se ve afectada por un sistema. Finalmente, el código imprime las señales h, x y y para visualizar los datos y el resultado de la convolución.

## Parte B

## Parte C

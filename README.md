# Laboratorio-2---convolución-y-correlación
En este laboratorio se desarrollo a partir de unas señales la convolución, correlación cruzada y frecuencia de Nyquist 

## Introducción
En este laboratorio  se observo cómo se comportan las señales tanto en el tiempo como en la frecuencia. Lo haremos aplicando tres técnicas fundamentales: la convolución, la correlación y la transformada de Fourier. Además del análisis de una señal electrooculograma (EOG).

## importación de librerias 
```python
!pip install wfdb
import matplotlib.pyplot as plt
import numpy as np
import wfdb
import pandas as pd
import os
from scipy.stats import norm
import seaborn as sns
from scipy.fft import fft, fftfreq
from scipy.signal import welch
```

Este bloque importa librerías clave para analizar señales biológicas: ` wfdb`  para leer datos fisiológicos, ` numpy`  y `  pandas`  para manejo numérico y de datos, ` matplotlib`  y `  seaborn`  para gráficos, y funciones de `scipy` para calcular la transformada de Fourier y la densidad espectral de potencia, herramientas fundamentales para estudiar las características temporales y frecuenciales de la señal.

<h1 align="center"><i><b>PARTE A DEL LABORATORIO</b></i></h1>

<p align="center">
<img width="397" height="601" alt="image" src="https://github.com/user-attachments/assets/39417bd5-72e7-4804-afea-c7df0d841e01" />
</p>

<p align="center">
    <img <img width="900" height="900" alt="image" src="https://github.com/user-attachments/assets/b8c7986e-e425-430a-8730-fa0a47cef2b5" />
</p>

## Señal y[n] resultante de la convolución

```python

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
    <img <img width="900" height="989" alt="image" src="https://github.com/user-attachments/assets/300a745d-cdfe-4ab3-a412-432d85f5e715" />
</p>

Este código en Python calcula la convolución discreta entre dos señales utilizando la función `np.convolve()` de` NumPy`. Primero, se definen dos listas, `h` y `x`, que representan la respuesta al impulso de un sistema y una señal de entrada, respectivamente. Luego, se aplica la convolución entre estas dos señales usando `np.convolve(x, h, mode='full')`, lo que genera una nueva señal y cuya longitud es la suma de las longitudes de `x` y `h` menos uno. 

La convolución es una operación fundamental en procesamiento de señales, ya que permite analizar cómo una señal se ve afectada por un sistema. Finalmente, el código imprime las señales `h`, `x` y y para visualizar los datos y el resultado de la convolución.

<h1 align="center"><i><b>PARTE B DEL LABORATORIO</b></i></h1>
<p align="center">
<img width="311" height="608" alt="image" src="https://github.com/user-attachments/assets/9065b311-a45d-46a8-be42-f79db087f73d" />
</p>

<p align="center">
<img width="600" height="365" alt="image" src="https://github.com/user-attachments/assets/9c230bfd-b256-43fb-aa1b-c730169ba8e8" />
</p>

    Tenemos dos señales sinusoidales con la misma frecuencia (100Hz).

<p align="center">
    <img width="900" height="419" alt="image" src="https://github.com/user-attachments/assets/ea0980be-05ba-41c2-9b43-2d785afde5c0" />
    </p>
    
    Sustituimos el valor de Ts (periodo de muestreo) para calcular w (frecuencia angular discreta)  y Fs (frecuencia de muestreo). 
    Esta frecuencia angular discreta la reemplazamos como el argumento en la ecuación de cada señal.
    
<p align="center">
 <img width="800" height="527" alt="image" src="https://github.com/user-attachments/assets/4f1da754-c7d2-4634-a9d6-2f2bf2e9bfb2" />
    </p>
    
    Usando la frecuencia de 100Hz y la frecuencia de muestreo de 800Hz, se calcula el número de muestras por periodo.
    Posteriormente, se calculan los valores de X1(n) y X2(n) desde n=0, hasta n=8.

<p align="center">
<img width="800" height="597" alt="image" src="https://github.com/user-attachments/assets/66be9013-b4a5-4ed7-82f6-ac3c55dd1476" />
    </p>
    
    Se calcula la correlación entre estas dos señales. 
    El número total de muestras es de 9, por lo tanto la longitud de la secuencia es del 17, con valores de -8 a 8. 
    Finalmente se genera una tabla con los resultados de r(k).

<p align="center">
<img width="900" height="780" alt="image" src="https://github.com/user-attachments/assets/0938be7a-fd08-43c6-8715-6a66fc5b2a9c" />
    </p>
    
    Se grafican aproximadamente los valores de k (eje x) v.s los valores de r(k) (eje y).

    
<p align="center">
<img width="1000" height="312" alt="image" src="https://github.com/user-attachments/assets/df2d4b0e-dd44-4283-8fe9-0cd6ab795040" />
    </p>

    De esta convolución podemos interpretar que:
    el seno y el coseno son ortogonales en la ventana ya que r(0)=0.
    La función es de carácter impar ya que -r(k) = r(-k).
    Los picos en K = 2 y -2, describen que la mejor alineación entre ambas señales existe trasladando 2 muestras (desfase de 90 grados).

### ¿En qué situaciones resulta útil aplicar la correlación cruzada en el procesamiento digital de señales?
-Estimar retardos de tiempo entre señales. Esto se aplica en radar, sonar o acústica, donde se mide cuánto tarda una señal en llegar a diferentes receptores y así localizar la fuente.  
-Sincronizar señales digitales. Comparando la señal recibida con una secuencia de "referencia", el sistema es capaz de alinear el reloj de muestreo y recuperar esta información transmitida.  
-Detección de patrones. Se pueden detectar complejos QRS en el ECG comparando la señal real con una forma de onda típica almacenada como template.  
-Identificar ecos o multitrayectorias. En acústica y encomunicaciones inalámbricas para reconocer copias retardadas de la misma señal causadas por reflexiones.  
-Estimar desfases o similitud entre señales periódicas. Entre señales periódica se puede determinar en qué punto dos ondas senoidales están más alineadas o desfasadas, a través de la posición del máximo o mínimo de la función de correlación.  

<h1 align="center"><i><b>PARTE C DEL LABORATORIO</b></i></h1>
inicalmente para la adquisición de la señal EOG se utilizó el código proporcionado que emplea la librería `nidaqmx`, la cual permite interactuar con dispositivos NI DAQ para la captura de señales analógicas. En el código se configura el canal de entrada analógica, la frecuencia de muestreo (800 Hz, cumpliendo el criterio de Nyquist), y el tiempo total de adquisición (5 segundos). Luego, se realiza la lectura finita de muestras y se guarda la señal en un vector. Finalmente, se genera un gráfico que muestra la señal adquirida en función del tiempo, permitiendo visualizar claramente la señal EOG en formato digital lista para su posterior análisis.


<p align="center">
<img width="250" height="600" alt="image" src="https://github.com/user-attachments/assets/3852f774-2bbb-4476-a4b5-468acdf76399" />
  </p>
  
```python
Librería de uso de la DAQ
!python -m pip install nidaqmx     

Driver NI DAQ mx
!python -m nidaqmx installdriver   

Created on Thu Aug 21 08:36:05 2025
@author: Carolina Corredor
"""

# Librerías: 
import nidaqmx                     # Librería daq. Requiere haber instalado el driver nidaqmx
from nidaqmx.constants import AcquisitionType # Para definir que adquiera datos de manera consecutiva
import matplotlib.pyplot as plt    # Librería para graficar
import numpy as np                 # Librería de funciones matemáticas

#%% Adquisición de la señal por tiempo definido

fs = 800           # Frecuencia de muestreo en Hz. Recordar cumplir el criterio de Nyquist
duracion = 5       # Periodo por el cual desea medir en segundos
senal = []          # Vector vacío en el que se guardará la señal
dispositivo = 'Dev3/ai0' # Nombre del dispositivo/canal (se puede cambiar el nombre en NI max)

total_muestras = int(fs * duracion)

with nidaqmx.Task() as task:
    # Configuración del canal
    task.ai_channels.add_ai_voltage_chan(dispositivo)
    # Configuración del reloj de muestreo
    task.timing.cfg_samp_clk_timing(
        fs,
        sample_mode=AcquisitionType.FINITE,   # Adquisición finita
        samps_per_chan=total_muestras        # Total de muestras que quiero
    )

    # Lectura de todas las muestras de una vez
    senal = task.read(number_of_samples_per_channel=total_muestras)

t = np.arange(len(senal))/fs # Crea el vector de tiempo 
plt.plot(t,senal)
plt.axis([0,duracion,-0.7,0.11])
plt.grid()
plt.title(f"fs={fs}Hz, duración={duracion}s, muestras={len(senal)}")
plt.show()
```

```python
df = pd.read_csv('senal_guardada2.csv')
x = df.iloc[:, 0]
y = df.iloc[:, 1]
plt.figure(figsize=(10, 5))
plt.plot(x,y,color='purple')
plt.title('Señal de EOG extraida del generador')
plt.xlabel('Tiempo (s)')
plt.ylabel('Voltaje (mV)')
plt.grid(True)
plt.show()
signal2= df.iloc[:, 1]
color='red'
```
Este código carga una señal EOG desde un archivo `CSV `usando `pandas`, extrae las columnas de `tiempo` y `voltaje`, y luego grafica la señal en función del tiempo con `matplotlib`, mostrando la variación del voltaje en milivoltios y agregando etiquetas y una cuadrícula para facilitar la visualización.
*grafica*

<p align="center">
<img width="700" height="470" alt="image" src="https://github.com/user-attachments/assets/c7c55d42-851e-4fd6-add1-f6800ad75bac" />
</p>

```python
senal=signal2
media = np.mean(senal)
mediana = np.median(senal)
desviacion = np.std(senal)
maximo = np.max(senal)
minimo = np.min(senal)


print(f"Media: {media}")
print(f"Mediana: {mediana}")
print(f"Desviación estándar: {desviacion}")
print(f"Máximo: {maximo}")
print(f"Mínimo: {minimo}")
```
Se calcularon los estadísticos descriptivos fundamentales de la señal EOG para caracterizar su comportamiento en el dominio temporal. 
*La` media `indica el valor promedio general.

*la `mediana` representa el punto central de los datos.

*la `desviación` estándar muestra la variabilidad o dispersión de la señal.

 *El `máximo` y `mínimo` reflejan los valores extremos o picos. 
 
 Estos parámetros son esenciales para comprender la distribución y estabilidad de la señal antes de realizar análisis más profundos.
 
 **resultados**
 
*Media: 0.15874137409805553
*Mediana: 0.13174945232458413
*Desviación estándar: 0.1580258539091611
*Máximo: 0.7196162504842505
*Mínimo: -0.49320164741948247

```python

N = len(signal2)
yf = fft(signal2)
xf = fftfreq(N, 1/fs)


xf_pos = xf[:N//2]
yf_pos = np.abs(yf[:N//2])


max_freq = 100   # cambia según tu aplicación
mask = xf_pos <= max_freq
xf_rec = xf_pos[mask]
yf_rec = yf_pos[mask]


f_psd, Pxx = welch(signal2.values, fs, nperseg=1024)


plt.figure(figsize=(12, 6))


plt.subplot(2, 1, 1)
plt.plot(xf_rec, yf_rec)
plt.title("Transformada de Fourier (Magnitud) - Recortada")
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Amplitud")
plt.grid(True)

plt.subplot(2, 1, 2)
plt.semilogy(f_psd, Pxx)
plt.title("Densidad espectral de potencia (PSD)")
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Potencia/Hz (dB/Hz)")
plt.grid(True)

plt.tight_layout()
plt.show()
```

Se calcula la Transformada de Fourier para analizar la señal en frecuencia, mostrando solo las frecuencias positivas hasta 100 Hz para mayor claridad. Además, se estima la densidad espectral de potencia con el método de Welch y se grafican ambos resultados para visualizar la distribución de energía en la señal.

*graficas*

<p align="center">
<img width="1000" height="590" alt="image" src="https://github.com/user-attachments/assets/8b8a204a-5b72-4e7a-8ba7-386d69d273f4" />
</p>

```python
f_psd, Pxx = welch(signal2.values, fs, nperseg=1024)

limite_frecuencia = 200
mask = f_psd <= limite_frecuencia

freq_media = np.sum(f_psd[mask] * Pxx[mask]) / np.sum(Pxx[mask])

acum_psd = np.cumsum(Pxx[mask])
acum_psd /= acum_psd[-1]  # Normalizar a 1
freq_mediana = f_psd[mask][np.where(acum_psd >= 0.5)[0][0]]

var_frec = np.sum(((f_psd[mask] - freq_media)**2) * Pxx[mask]) / np.sum(Pxx[mask])
desv_est_frec = np.sqrt(var_frec)

print(f"Frecuencia media: {freq_media:.2f} Hz")
print(f"Frecuencia mediana: {freq_mediana:.2f} Hz")
print(f"Desviación estándar: {desv_est_frec:.2f} Hz")

plt.figure(figsize=(10, 5))
plt.bar(f_psd[mask], Pxx[mask], width=f_psd[1]-f_psd[0], color='teal')
plt.title("Histograma de frecuencias (PSD)")
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Potencia")
plt.grid(True)
plt.show()
```

Finalmente se calcula la densidad espectral de potencia (PSD) de la señal hasta 200 Hz, un rango relevante para señales biológicas. A partir de la PSD se obtienen tres estadísticas clave:

*la `frecuencia media` es el promedio ponderado por la potencia.

*la `frecuencia mediana` divide la energía total en dos partes iguales.

*la `desviación estándar`mide la dispersión de la energía en frecuencias.

posteriormente, se grafica un histograma que muestra la distribución de potencia a lo largo del espectro de frecuencias, facilitando la visualización de dónde se concentra la energía de la señal.

*histograma y resultados*
*Frecuencia media: 18.74 Hz
*Frecuencia mediana: 9.77 Hz
*Desviación estándar: 24.43 Hz

<p align="center">
<img width="872" height="470" alt="image" src="https://github.com/user-attachments/assets/cc839060-cbc4-47ea-b81c-5c59fc73e24b" />
</p>

Adicionalmente se realizo la clasifiacion de la señal :

**Determinística o Aleatoria:** La señal EOG es generalmente aleatoria, ya que presenta variaciones impredecibles debido a la actividad natural del ojo y el ruido biológico, aunque puede tener componentes periódicos asociados a movimientos repetitivos.

**Periódica o Apperiodica:** La señal EOG es apperiodica o no estrictamente periódica, dado que sus características no se repiten de forma exacta en el tiempo, reflejando la dinámica irregular de los movimientos oculares.

**Analógica o Digital:** Originalmente, la señal EOG es analógica, ya que es una señal continua en el tiempo y en amplitud. Sin embargo, al ser adquirida y almacenada en un computador mediante un proceso de muestreo, se convierte en una señal digital para su procesamiento.

<h1 align="center"><i><b>Bibliografia</b></i></h1>




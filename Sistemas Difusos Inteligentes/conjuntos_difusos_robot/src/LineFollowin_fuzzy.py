import numpy as np
import matplotlib.pyplot as plt
#*********************************************************
#                                                        *
#  FUNCIONES PRINCIPALES PARA LOS GRADOS DE MEMBRESIA
#                                                        *
#*********************************************************

# Función trapezoidal
def trapezoidal(x, a, b, c, d):
    return np.maximum(np.minimum(np.minimum((x - a)/(b - a + 1e-6), 1), (d - x)/(d - c + 1e-6)), 0)

# Función triangular
def triangular(x, a, b, c):
    return np.maximum(np.minimum((x - a)/(b - a + 1e-6), (c - x)/(c - b + 1e-6)), 0)

# Rango de reflexión
x = np.linspace(0, 1.0, 500)

#*********************************************************
#                                                        *
#*********************************************************

#GRAFICAMOS LOS DATOS DE LA REFLEXION

# Definición de funciones de membresía para el sensor de REFLEXION
low = trapezoidal(x, 0.0, 0.1, 0.2, 0.3)       # Low (trapezoidal)
gray = triangular(x, 0.2, 0.4, 0.5)            # Gray (triangular)
dark_gray = triangular(x, 0.4, 0.6, 0.8)       # DarkGray (triangular)
high = trapezoidal(x, 0.7, 0.8, 0.9, 1.0)      # High (trapezoidal)

# Graficar todas las funciones en una sola figura
plt.figure(figsize=(10, 6))
plt.plot(x, low, label='Low', color='blue')
plt.plot(x, gray, label='Gray', color='green')
plt.plot(x, dark_gray, label='DarkGray', color='orange')
plt.plot(x, high, label='High', color='red')

# Configuración del gráfico
plt.title('Funciones de membresía para Reflexión')
plt.xlabel('Reflexión')
plt.ylabel('Grado de pertenencia')
plt.ylim(0, 1.1)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


#GRAFICAMOS LOS DATOS DE LA DISTANCIA

# Rango de distancia
x = np.linspace(0, 200, 400)

# Definición de funciones
muy_cerca = triangular(x, 0.0, 0.0, 10)          # triangular
cerca = triangular(x, 8, 15, 20)                 # triangular
media = triangular(x, 15, 25.5, 30)              # triangular
lejos = triangular(x, 25, 40.5, 50)              # triangular
muy_lejos = trapezoidal(x, 45, 60, 200, 200)     # trapezoidal

# Graficar
plt.figure(figsize=(10, 6))
plt.plot(x, muy_cerca, label='Muy cerca')
plt.plot(x, cerca, label='Cerca')
plt.plot(x, media, label='Distancia media')
plt.plot(x, lejos, label='Lejos')
plt.plot(x, muy_lejos, label='Muy lejos')

# Configuración
plt.title('Funciones de membresía (Triangulares y Trapezoidales)')
plt.xlabel('Distancia (cm)')
plt.ylabel('Grado de pertenencia')
plt.ylim(0, 1.1)
plt.legend()
plt.grid(True)
plt.show()
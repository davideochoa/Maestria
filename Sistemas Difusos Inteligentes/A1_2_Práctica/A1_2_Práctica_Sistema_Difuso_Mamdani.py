"""
A1.2 Práctica - Sistema Difuso Mamdani
Evaluación de Satisfacción del Cliente
Fecha: 2025-11-12
"""

import matplotlib
matplotlib.use('TkAgg')  # 👈 Usa backend interactivo para mostrar ventanas
import matplotlib.pyplot as plt
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# =============================================================================
# PASO 1: DEFINICIÓN DE VARIABLES DIFUSAS
# =============================================================================

# Variable de entrada 1: Calidad del Servicio (escala 0-10)
calidad = ctrl.Antecedent(np.arange(0, 11, 1), 'calidad')

# Variable de entrada 2: Tiempo de Espera (0-60 min)
tiempo_espera = ctrl.Antecedent(np.arange(0, 61, 1), 'tiempo_espera')

# Variable de salida: Satisfacción del Cliente (0-100)
satisfaccion = ctrl.Consequent(np.arange(0, 101, 1), 'satisfaccion')

# =============================================================================
# PASO 2: FUNCIONES DE MEMBRESÍA
# =============================================================================

# Calidad del servicio
calidad['mala'] = fuzz.trimf(calidad.universe, [0, 0, 5])
calidad['regular'] = fuzz.trimf(calidad.universe, [3, 5, 7])
calidad['buena'] = fuzz.trimf(calidad.universe, [5, 10, 10])

# Tiempo de espera
tiempo_espera['corto'] = fuzz.trimf(tiempo_espera.universe, [0, 0, 20])
tiempo_espera['medio'] = fuzz.trimf(tiempo_espera.universe, [10, 30, 50])
tiempo_espera['largo'] = fuzz.trimf(tiempo_espera.universe, [40, 60, 60])

# Satisfacción
satisfaccion['baja'] = fuzz.trimf(satisfaccion.universe, [0, 0, 50])
satisfaccion['media'] = fuzz.trimf(satisfaccion.universe, [25, 50, 75])
satisfaccion['alta'] = fuzz.trimf(satisfaccion.universe, [50, 100, 100])

# =============================================================================
# PASO 3: REGLAS DIFUSAS
# =============================================================================

regla1 = ctrl.Rule(calidad['buena'] & tiempo_espera['corto'], satisfaccion['alta'])
regla2 = ctrl.Rule(calidad['buena'] & tiempo_espera['medio'], satisfaccion['media'])
regla3 = ctrl.Rule(calidad['buena'] & tiempo_espera['largo'], satisfaccion['media'])
regla4 = ctrl.Rule(calidad['regular'] & tiempo_espera['corto'], satisfaccion['media'])
regla5 = ctrl.Rule(calidad['regular'] & tiempo_espera['medio'], satisfaccion['media'])
regla6 = ctrl.Rule(calidad['regular'] & tiempo_espera['largo'], satisfaccion['baja'])
regla7 = ctrl.Rule(calidad['mala'] & tiempo_espera['corto'], satisfaccion['baja'])
regla8 = ctrl.Rule(calidad['mala'] & tiempo_espera['medio'], satisfaccion['baja'])
regla9 = ctrl.Rule(calidad['mala'] & tiempo_espera['largo'], satisfaccion['baja'])

# =============================================================================
# PASO 4: SISTEMA DE CONTROL
# =============================================================================

sistema_control = ctrl.ControlSystem([
    regla1, regla2, regla3, regla4, regla5,
    regla6, regla7, regla8, regla9
])
sistema = ctrl.ControlSystemSimulation(sistema_control)

# =============================================================================
# PASO 5: VISUALIZACIÓN DE FUNCIONES DE MEMBRESÍA (.view)
# =============================================================================

print("Mostrando funciones de membresía...\n")

calidad.view()
tiempo_espera.view()
satisfaccion.view()

plt.show()

# =============================================================================
# PASO 6: EVALUACIÓN DE CASOS DE PRUEBA
# =============================================================================

casos = [
    {"calidad": 9, "tiempo": 10, "descripcion": "Excelente servicio, espera corta"},
    {"calidad": 8, "tiempo": 35, "descripcion": "Buen servicio, espera moderada"},
    {"calidad": 5, "tiempo": 45, "descripcion": "Servicio regular, espera larga"},
    {"calidad": 3, "tiempo": 50, "descripcion": "Servicio malo, espera larga"},
    {"calidad": 7, "tiempo": 15, "descripcion": "Buen servicio, espera corta"},
]

print("=" * 70)
print("RESULTADOS DE SATISFACCIÓN DEL CLIENTE")
print("=" * 70)

for caso in casos:
    sistema.input['calidad'] = caso['calidad']
    sistema.input['tiempo_espera'] = caso['tiempo']
    sistema.compute()
    resultado = sistema.output['satisfaccion']

    if resultado >= 70:
        nivel = "ALTA ✓"
    elif resultado >= 40:
        nivel = "MEDIA ~"
    else:
        nivel = "BAJA ✗"

    print(f"{caso['descripcion']}: {resultado:.2f}/100 ({nivel})")

# =============================================================================
# PASO 7: VISUALIZACIÓN DEL RESULTADO DIFUSO (.view sim)
# =============================================================================

print("\nMostrando resultado difuso del último caso...\n")
satisfaccion.view(sim=sistema)
plt.show()

# =============================================================================
# PASO 8: SUPERFICIE DE CONTROL 3D
# =============================================================================

print("Generando superficie de control 3D...\n")

calidad_range = np.arange(0, 11, 0.5)
tiempo_range = np.arange(0, 61, 2)
x, y = np.meshgrid(calidad_range, tiempo_range)
z = np.zeros_like(x)

# Cálculo
for i in range(x.shape[0]):
    for j in range(x.shape[1]):
        sistema.input['calidad'] = x[i, j]
        sistema.input['tiempo_espera'] = y[i, j]
        sistema.compute()
        z[i, j] = sistema.output['satisfaccion']

# Gráfica
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(x, y, z, cmap='viridis', linewidth=0.2, antialiased=True)

ax.set_xlabel('Calidad del Servicio')
ax.set_ylabel('Tiempo de Espera (min)')
ax.set_zlabel('Satisfacción del Cliente')
ax.set_title('Superficie de Control - Sistema Difuso Mamdani', fontsize=12, fontweight='bold')

plt.show()

print("✅ Listo: todas las gráficas se han mostrado correctamente.")
"""
A1.2 Práctica - Sistema Difuso Mamdani
Evaluación de Satisfacción del Cliente
Autor: [Tu Nombre]
Fecha: 2025-11-11

Descripción:
Este sistema difuso evalúa la satisfacción del cliente basándose en dos variables:
1. Calidad del servicio (0-10)
2. Tiempo de espera (0-60 minutos)
"""

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Qt5Agg')  # o 'QtAgg' dependiendo de tu versión

# =============================================================================
# PASO 1: DEFINICIÓN DE VARIABLES DIFUSAS
# =============================================================================

# Variable de entrada 1: Calidad del Servicio (escala 0-10)
calidad = ctrl.Antecedent(np.arange(0, 11, 1), 'calidad')

# Variable de entrada 2: Tiempo de Espera en minutos (0-60)
tiempo_espera = ctrl.Antecedent(np.arange(0, 61, 1), 'tiempo_espera')

# Variable de salida: Satisfacción del Cliente (escala 0-100)
satisfaccion = ctrl.Consequent(np.arange(0, 101, 1), 'satisfaccion')

# =============================================================================
# PASO 2: DEFINICIÓN DE FUNCIONES DE MEMBRESÍA
# =============================================================================

# Funciones de membresía para CALIDAD DEL SERVICIO
# Utilizamos funciones triangulares (trimf) para modelar los conjuntos difusos
calidad['mala'] = fuzz.trimf(calidad.universe, [0, 0, 5])
calidad['regular'] = fuzz.trimf(calidad.universe, [3, 5, 7])
calidad['buena'] = fuzz.trimf(calidad.universe, [5, 10, 10])

# Funciones de membresía para TIEMPO DE ESPERA
tiempo_espera['corto'] = fuzz.trimf(tiempo_espera.universe, [0, 0, 20])
tiempo_espera['medio'] = fuzz.trimf(tiempo_espera.universe, [10, 30, 50])
tiempo_espera['largo'] = fuzz.trimf(tiempo_espera.universe, [40, 60, 60])

# Funciones de membresía para SATISFACCIÓN (salida)
satisfaccion['baja'] = fuzz.trimf(satisfaccion.universe, [0, 0, 50])
satisfaccion['media'] = fuzz.trimf(satisfaccion.universe, [25, 50, 75])
satisfaccion['alta'] = fuzz.trimf(satisfaccion.universe, [50, 100, 100])

# =============================================================================
# PASO 3: DEFINICIÓN DE REGLAS DIFUSAS (BASE DE CONOCIMIENTO)
# =============================================================================

"""
Reglas de inferencia basadas en lógica experta:
- Si la calidad es buena y el tiempo es corto → satisfacción alta
- Si la calidad es mala o el tiempo es largo → satisfacción baja
- Casos intermedios → satisfacción media
"""

# Regla 1: Calidad buena + Tiempo corto = Satisfacción alta
regla1 = ctrl.Rule(calidad['buena'] & tiempo_espera['corto'], satisfaccion['alta'])

# Regla 2: Calidad buena + Tiempo medio = Satisfacción media
regla2 = ctrl.Rule(calidad['buena'] & tiempo_espera['medio'], satisfaccion['media'])

# Regla 3: Calidad buena + Tiempo largo = Satisfacción media
regla3 = ctrl.Rule(calidad['buena'] & tiempo_espera['largo'], satisfaccion['media'])

# Regla 4: Calidad regular + Tiempo corto = Satisfacción media
regla4 = ctrl.Rule(calidad['regular'] & tiempo_espera['corto'], satisfaccion['media'])

# Regla 5: Calidad regular + Tiempo medio = Satisfacción media
regla5 = ctrl.Rule(calidad['regular'] & tiempo_espera['medio'], satisfaccion['media'])

# Regla 6: Calidad regular + Tiempo largo = Satisfacción baja
regla6 = ctrl.Rule(calidad['regular'] & tiempo_espera['largo'], satisfaccion['baja'])

# Regla 7: Calidad mala + Tiempo corto = Satisfacción baja
regla7 = ctrl.Rule(calidad['mala'] & tiempo_espera['corto'], satisfaccion['baja'])

# Regla 8: Calidad mala + Tiempo medio = Satisfacción baja
regla8 = ctrl.Rule(calidad['mala'] & tiempo_espera['medio'], satisfaccion['baja'])

# Regla 9: Calidad mala + Tiempo largo = Satisfacción baja
regla9 = ctrl.Rule(calidad['mala'] & tiempo_espera['largo'], satisfaccion['baja'])

# =============================================================================
# PASO 4: CREACIÓN DEL SISTEMA DE CONTROL DIFUSO
# =============================================================================

# Crear el sistema de control con todas las reglas
sistema_control = ctrl.ControlSystem([regla1, regla2, regla3, regla4, regla5,
                                      regla6, regla7, regla8, regla9])

# Crear la simulación del sistema
sistema = ctrl.ControlSystemSimulation(sistema_control)

# =============================================================================
# PASO 5: EVALUACIÓN DEL SISTEMA CON CASOS DE PRUEBA
# =============================================================================

print("=" * 70)
print("SISTEMA DIFUSO MAMDANI - EVALUACIÓN DE SATISFACCIÓN DEL CLIENTE")
print("=" * 70)

# Casos de prueba
casos_prueba = [
    {"calidad": 9, "tiempo": 10, "descripcion": "Excelente servicio, espera corta"},
    {"calidad": 8, "tiempo": 35, "descripcion": "Buen servicio, espera moderada"},
    {"calidad": 5, "tiempo": 45, "descripcion": "Servicio regular, espera larga"},
    {"calidad": 3, "tiempo": 50, "descripcion": "Servicio malo, espera larga"},
    {"calidad": 7, "tiempo": 15, "descripcion": "Buen servicio, espera corta"},
]

resultados = []

for i, caso in enumerate(casos_prueba, 1):
    # Asignar valores de entrada al sistema
    sistema.input['calidad'] = caso['calidad']
    sistema.input['tiempo_espera'] = caso['tiempo']

    # Realizar el cómputo (fuzzificación, inferencia y defuzzificación)
    sistema.compute()

    # Obtener el resultado
    resultado = sistema.output['satisfaccion']
    resultados.append(resultado)

    # Mostrar resultados
    print(f"\nCaso {i}: {caso['descripcion']}")
    print(f"  • Calidad del servicio: {caso['calidad']}/10")
    print(f"  • Tiempo de espera: {caso['tiempo']} minutos")
    print(f"  • Satisfacción calculada: {resultado:.2f}/100")

    # Clasificación cualitativa
    if resultado >= 70:
        clasificacion = "ALTA ✓"
    elif resultado >= 40:
        clasificacion = "MEDIA ~"
    else:
        clasificacion = "BAJA ✗"
    print(f"  • Nivel de satisfacción: {clasificacion}")

# =============================================================================
# PASO 6: VISUALIZACIÓN DE FUNCIONES DE MEMBRESÍA
# =============================================================================

fig, axes = plt.subplots(3, 1, figsize=(10, 10))

# Gráfica de Calidad del Servicio
calidad.view(ax=axes[0])
axes[0].set_title('Funciones de Membresía - Calidad del Servicio',
                  fontsize=12, fontweight='bold')
axes[0].legend(loc='upper right')
axes[0].grid(True, alpha=0.3)

# Gráfica de Tiempo de Espera
tiempo_espera.view(ax=axes[1])
axes[1].set_title('Funciones de Membresía - Tiempo de Espera',
                  fontsize=12, fontweight='bold')
axes[1].legend(loc='upper right')
axes[1].grid(True, alpha=0.3)

# Gráfica de Satisfacción
satisfaccion.view(ax=axes[2])
axes[2].set_title('Funciones de Membresía - Satisfacción del Cliente',
                  fontsize=12, fontweight='bold')
axes[2].legend(loc='upper right')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('funciones_membresia.png', dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# PASO 7: VISUALIZACIÓN DE SUPERFICIE DE CONTROL
# =============================================================================

# Crear malla de valores para la superficie
calidad_range = np.arange(0, 11, 0.5)
tiempo_range = np.arange(0, 61, 2)
x, y = np.meshgrid(calidad_range, tiempo_range)
z = np.zeros_like(x)

# Calcular satisfacción para cada combinación
print("\n" + "=" * 70)
print("Generando superficie de control 3D...")
for i in range(x.shape[0]):
    for j in range(x.shape[1]):
        sistema.input['calidad'] = x[i, j]
        sistema.input['tiempo_espera'] = y[i, j]
        sistema.compute()
        z[i, j] = sistema.output['satisfaccion']

# Crear gráfica 3D
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(x, y, z, cmap='viridis', alpha=0.8,
                       edgecolor='none', antialiased=True)

ax.set_xlabel('Calidad del Servicio', fontsize=11, fontweight='bold')
ax.set_ylabel('Tiempo de Espera (min)', fontsize=11, fontweight='bold')
ax.set_zlabel('Satisfacción del Cliente', fontsize=11, fontweight='bold')
ax.set_title('Superficie de Control - Sistema Difuso Mamdani',
             fontsize=13, fontweight='bold', pad=20)

# Añadir barra de color
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Satisfacción')

plt.savefig('superficie_control.png', dpi=300, bbox_inches='tight')
plt.show()

print("✓ Visualizaciones generadas correctamente")
print("=" * 70)
print("\nArchivos generados:")
print("  • funciones_membresia.png")
print("  • superficie_control.png")
print("=" * 70)
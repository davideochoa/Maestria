"""
A1.2 Práctica - Sistema Difuso Mamdani
Evaluación de Satisfacción del Cliente
Autor: Sistema de Lógica Difusa
Fecha: 2025-11-11

Descripción:
Este sistema difuso evalúa la satisfacción del cliente basándose en dos variables:
1. Calidad del servicio (0-10)
2. Tiempo de espera (0-60 minutos)
"""

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

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
# PASO 3: VISUALIZACIÓN DE FUNCIONES DE MEMBRESÍA
# =============================================================================

fig, axes = plt.subplots(3, 1, figsize=(10, 10))

# Gráfica de Calidad del Servicio
calidad.view(ax=axes[0])
axes[0].set_title('Funciones de Membresía - Calidad del Servicio', fontsize=12, fontweight='bold')
axes[0].legend(loc='upper right')
axes[0].grid(True, alpha=0.3)

# Gráfica de Tiempo de Espera
tiempo_espera.view(ax=axes[1])
axes[1].set_title('Funciones de Membresía - Tiempo de Espera', fontsize=12, fontweight='bold')
axes[1].legend(loc='upper right')
axes[1].grid(True, alpha=0.3)

# Gráfica de Satisfacción
satisfaccion.view(ax=axes[2])
axes[2].set_title('Funciones de Membresía - Satisfacción del Cliente', fontsize=12, fontweight='bold')
axes[2].legend(loc='upper right')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('funciones_membresia.png', dpi=300, bbox_inches='tight')
plt.show()

print("✓ Funciones de membresía definidas y visualizadas correctamente")
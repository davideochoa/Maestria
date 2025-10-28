# -*- coding: utf-8 -*-
# Requisitos: pip install numpy matplotlib

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------
# Función de membresía: Triangular
# Parámetros: [a, b, c] con a <= b <= c
# - Sube linealmente de 0 a 1 entre a y b
# - Baja linealmente de 1 a 0 entre b y c
# - Fuera de [a, c] vale 0
# ---------------------------------------------------------
def triangular_membership(x_values, params):
    a = params[0]
    b = params[1]
    c = params[2]

    membership = np.zeros_like(x_values, dtype=float)

    # Tramo ascendente: entre a y b
    if a != b:
        # IMPORTANTE: usar "&" (and vectorizado) con paréntesis
        mask_asc = (x_values >= a) & (x_values <= b)
        membership[mask_asc] = (x_values[mask_asc] - a) / (b - a)

    # Tramo descendente: entre b y c
    if b != c:
        mask_desc = (x_values >= b) & (x_values <= c)
        membership[mask_desc] = (c - x_values[mask_desc]) / (c - b)

    # Aseguramos pico en b
    membership[np.isclose(x_values, b)] = 1.0

    membership = np.clip(membership, 0.0, 1.0)
    return membership


# ---------------------------------------------------------
# Función de membresía: Trapezoidal
# Parámetros: [a, b, c, d] con a <= b <= c <= d
# - Sube linealmente de 0 a 1 entre a y b
# - Meseta (1) entre b y c
# - Baja linealmente de 1 a 0 entre c y d
# - Fuera de [a, d] vale 0
# ---------------------------------------------------------
def trapezoidal_membership(x_values, params):
    a = params[0]
    b = params[1]
    c = params[2]
    d = params[3]

    membership = np.zeros_like(x_values, dtype=float)

    # Tramo ascendente: [a, b)
    if a != b:
        mask_asc = (x_values >= a) & (x_values < b)
        membership[mask_asc] = (x_values[mask_asc] - a) / (b - a)

    # Meseta: [b, c]
    mask_plateau = (x_values >= b) & (x_values <= c)
    membership[mask_plateau] = 1.0

    # Tramo descendente: (c, d]
    if c != d:
        mask_desc = (x_values > c) & (x_values <= d)
        membership[mask_desc] = (d - x_values[mask_desc]) / (d - c)

    membership = np.clip(membership, 0.0, 1.0)
    return membership


# ---------------------------------------------------------
# Utilidad para graficar un conjunto de curvas en una figura
# ---------------------------------------------------------
def plot_membership_sets(x_values, sets_dict, title, x_label, output_png):
    plt.figure(figsize=(9, 5.5))

    for label in sets_dict:
        y_values = sets_dict[label]
        plt.plot(x_values, y_values, linewidth=2, label=label)

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel('μ(x)')
    plt.ylim(-0.05, 1.05)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_png, dpi=200)
    plt.show()

    print("Guardada:", output_png)


def main():
    # Universos de discurso
    desviacion_x = np.arange(0, 101, 1)   # 0..100
    distancia_x  = np.arange(0, 201, 1)   # 0..200
    angulo_x     = np.arange(0, 181, 1)   # 0..180

    # Desviación (triangulares)
    desviacion_params = {}
    desviacion_params["Muy izquierda"] = [0, 0, 25]
    desviacion_params["Izquierda"]     = [0, 25, 50]
    desviacion_params["Centrado"]      = [25, 50, 75]
    desviacion_params["Derecha"]       = [50, 75, 100]
    desviacion_params["Muy derecha"]   = [75, 100, 100]

    # Distancia (trapezoidales)
    distancia_params = {}
    distancia_params["Muy cerca"] = [0, 0, 20, 40]
    distancia_params["Cerca"]     = [20, 40, 60, 80]
    distancia_params["Media"]     = [60, 80, 120, 140]
    distancia_params["Lejos"]     = [120, 160, 200, 200]

    # Ángulo (triangulares)
    angulo_params = {}
    angulo_params["Sin giro (0°)"]        = [0, 0, 45]
    angulo_params["Giro leve (45°)"]      = [0, 45, 90]
    angulo_params["Giro moderado (90°)"]  = [45, 90, 135]
    angulo_params["Giro fuerte (135°)"]   = [90, 135, 180]
    angulo_params["Giro completo (180°)"] = [135, 180, 180]

    # Cálculo de curvas

    desviacion_sets = {}
    for nombre in desviacion_params:
        params = desviacion_params[nombre]
        y_vals = triangular_membership(desviacion_x, params)
        desviacion_sets[nombre] = y_vals

    distancia_sets = {}
    for nombre in distancia_params:
        params = distancia_params[nombre]
        y_vals = trapezoidal_membership(distancia_x, params)
        distancia_sets[nombre] = y_vals

    angulo_sets = {}
    for nombre in angulo_params:
        params = angulo_params[nombre]
        y_vals = triangular_membership(angulo_x, params)
        angulo_sets[nombre] = y_vals

    # Gráficas

    plot_membership_sets(
        x_values=desviacion_x,
        sets_dict=desviacion_sets,
        title="Desviación de la línea (Triangulares)",
        x_label="Reflexión (%)",
        output_png="desviacion_triangular.png"
    )

    plot_membership_sets(
        x_values=distancia_x,
        sets_dict=distancia_sets,
        title="Distancia al obstáculo (Trapezoidales)",
        x_label="Distancia (cm)",
        output_png="distancia_trapezoidal.png"
    )

    plot_membership_sets(
        x_values=angulo_x,
        sets_dict=angulo_sets,
        title="Ángulo de giro (Triangulares)",
        x_label="Ángulo (°)",
        output_png="angulo_triangular.png"
    )

    # Comparativa

    comparativa_x = np.arange(0, 101, 1)

    tri_params = [25, 50, 75]
    tri_y = triangular_membership(comparativa_x, tri_params)

    trap_params = [15, 35, 65, 85]
    trap_y = trapezoidal_membership(comparativa_x, trap_params)

    plt.figure(figsize=(11, 4.5))

    plt.subplot(1, 2, 1)
    plt.plot(comparativa_x, tri_y, color='blue', linewidth=3, label='Triangular [25,50,75]')
    plt.fill_between(comparativa_x, 0, tri_y, alpha=0.25, color='blue')
    plt.title("Función Triangular (ejemplo)")
    plt.xlabel("x")
    plt.ylabel("μ(x)")
    plt.ylim(-0.1, 1.1)
    plt.grid(alpha=0.3)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(comparativa_x, trap_y, color='red', linewidth=3, label='Trapezoidal [15,35,65,85]')
    plt.fill_between(comparativa_x, 0, trap_y, alpha=0.25, color='red')
    plt.title("Función Trapezoidal (ejemplo)")
    plt.xlabel("x")
    plt.ylim(-0.1, 1.1)
    plt.grid(alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig("comparativa_tri_vs_trap.png", dpi=200)
    plt.show()

    print("Guardada:", "comparativa_tri_vs_trap.png")


if __name__ == "__main__":
    main()
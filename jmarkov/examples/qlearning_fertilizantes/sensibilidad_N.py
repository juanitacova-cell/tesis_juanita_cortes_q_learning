"""
Por: Juanita Valentina Cortés Vargas

Sensibilidad de los hiperparámetros de Q-learning (Sutton y Barto) para N.

Uso, desde la carpeta jmarkov:
    py examples/qlearning_fertilizantes/sensibilidad_N.py

Usa el módulo jmarkov.mdp.qlearning SIN cambios (alpha y epsilon fijos,
Q inicial en 0, una sola trayectoria) y el mismo MDP de qlearning_N.py.

FASE 1: alpha x n_pasos, con epsilon fijo (alpha fijo en cada corrida).
FASE 2: epsilon, con el mejor alpha y n_pasos de la fase 1.
Cada combinación se corre con varias semillas.

Medida principal: parte de la ventaja del MDP sobre la regla miope que
captura Q-learning:
    100 % = costo mínimo del MDP; 0 % = igual a la regla miope; < 0 = peor.


"""
from pathlib import Path
import itertools
import sys
import time

import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# DISEÑO DE LA SENSIBILIDAD
# -----------------------------------------------------------------------------
FASE_1 = {
    "alpha": [0.1, 0.03, 0.01, 0.003],
    "n_pasos": [1_000_000, 3_000_000],
    "epsilon": [0.1],
}
FASE_2_EPSILON = [0.05, 0.1, 0.3, 1.0]
SEMILLAS = [1, 2, 3]

# -----------------------------------------------------------------------------
# IMPORTAR EL MDP Y LAS FUNCIONES DE qlearning_N.py
# -----------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent))
import qlearning_N as base                              # noqa: E402
from jmarkov.mdp.qlearning import qlearning             # noqa: E402

ARCHIVO = base.SALIDA / f"sensibilidad_{base.COMMODITY}.csv"
MICROSEG_POR_PASO = 7.0   # medido; solo para estimar la duración


def preparar():
    original = base.cargar_mdp_independiente()
    data = original.construir_mdp(base.COMMODITY)
    initial_state, feasible_actions, _ = base.preparar_qlearning(original, data)

    # step más rápido que el de qlearning_N.py, con la MISMA distribución:
    # se sortea el estado siguiente con las probabilidades acumuladas de la
    # fila (np.searchsorted) en vez de rng.choice. No cambia el algoritmo.
    etiquetas = [str(a) for a in data["actions"]]
    columna = {a: j for j, a in enumerate(etiquetas)}
    acumuladas = {a: np.cumsum(data["transition_matrices"][data["actions"][j]], axis=1)
                  for j, a in enumerate(etiquetas)}
    for a in acumuladas:
        acumuladas[a][:, -1] = 1.0
    R = data["returns"]
    # Mismo costo que ve Q-learning en qlearning_N.py (según COSTO_REFORMULADO).
    costo_demanda = base.costo_inevitable(original, data)

    def step(state, action, rng):
        siguiente = int(np.searchsorted(acumuladas[action][state], rng.random(),
                                        side="right"))
        return siguiente, float(R[state, columna[action]]) - costo_demanda[state]

    _, acciones_ref = base.resolver_referencia(original, data)
    V_opt = base.costo_politica(original, data, acciones_ref)
    miope = base.reglas_simples(original, data, feasible_actions)[
        "Regla miope (lo más barato hoy)"]
    V_miope = base.costo_politica(original, data, miope)
    return original, data, initial_state, feasible_actions, step, acciones_ref, V_opt, V_miope


def correr(fase, alpha, epsilon, n_pasos, semilla, contexto):
    original, data, initial_state, feasible_actions, step, acciones_ref, V_opt, V_miope = contexto
    n = len(data["states"])
    modelo = qlearning(initial_state, feasible_actions, step, original.BETA)
    inicio = time.perf_counter()
    _, politica = modelo.solve(n_steps=n_pasos, alpha=alpha, epsilon=epsilon,
                               minimize=True, seed=semilla)
    tiempo = time.perf_counter() - inicio

    visitados = {s for (s, _) in modelo.visit_counts}
    fila = {"fase": fase, "alpha": alpha, "epsilon": epsilon, "n_pasos": n_pasos,
            "semilla": semilla, "estados_visitados": len(visitados),
            "tiempo_s": round(tiempo, 1)}
    if all(i in politica for i in range(n)):
        acciones = [str(politica[i]) for i in range(n)]
        V_ql = base.costo_politica(original, data, acciones)
        sobrecosto = float(np.mean(V_ql - V_opt))
        ventaja_miope = float(np.mean(V_miope - V_opt))
        fila.update({
            "coincidencia_%": 100 * np.mean([a == r for a, r in zip(acciones, acciones_ref)]),
            "sobrecosto_USD": sobrecosto,
            "sobrecosto_%": 100 * sobrecosto / float(np.mean(V_opt)),
            "ventaja_capturada_%": 100 * (1 - sobrecosto / ventaja_miope),
        })
    return fila


def guardar(fila):
    ARCHIVO.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([fila]).to_csv(ARCHIVO, mode="a", index=False,
                                header=not ARCHIVO.exists(), encoding="utf-8-sig")


def resumen(tabla, columnas):
    return (tabla.groupby(columnas)
            .agg(ventaja_capturada_media=("ventaja_capturada_%", "mean"),
                 ventaja_capturada_min=("ventaja_capturada_%", "min"),
                 coincidencia_media=("coincidencia_%", "mean"),
                 sobrecosto_medio_USD=("sobrecosto_USD", "mean"))
            .round(2).reset_index())


def ejecutar_fase(fase, combinaciones, contexto):
    filas = []
    total = len(combinaciones) * len(SEMILLAS)
    for k, ((alpha, epsilon, n_pasos), semilla) in enumerate(
            itertools.product(combinaciones, SEMILLAS), start=1):
        fila = correr(fase, alpha, epsilon, n_pasos, semilla, contexto)
        guardar(fila)
        filas.append(fila)
        print(f"  [{k}/{total}] alpha={alpha:<7} epsilon={epsilon:<5} "
              f"n_pasos={n_pasos:>11,} semilla={semilla} -> "
              f"ventaja capturada {fila.get('ventaja_capturada_%', float('nan')):7.1f} % | "
              f"coincidencia {fila.get('coincidencia_%', float('nan')):5.1f} % | "
              f"{fila['tiempo_s']:.0f} s", flush=True)
    return pd.DataFrame(filas)


def main():
    if ARCHIVO.exists():
        ARCHIVO.unlink()   # empezar un CSV nuevo en cada ejecución

    print("Preparando el MDP y la referencia...")
    contexto = preparar()
    original = contexto[0]

    combos_1 = list(itertools.product(FASE_1["alpha"], FASE_1["epsilon"], FASE_1["n_pasos"]))
    pasos_1 = sum(c[2] for c in combos_1) * len(SEMILLAS)
    pasos_2 = (max(FASE_1["n_pasos"]) * len(SEMILLAS)
               * len(set(FASE_2_EPSILON) - set(FASE_1["epsilon"])))
    print("=" * 78)
    print(f"SENSIBILIDAD DE Q-LEARNING — {base.COMMODITY} (beta = {original.BETA:.6f})")
    
    print("=" * 78)
    print(f"FASE 1: alpha {FASE_1['alpha']} x n_pasos {FASE_1['n_pasos']}, "
          f"epsilon {FASE_1['epsilon']}")
    print(f"FASE 2: epsilon {FASE_2_EPSILON} con el mejor alpha y n_pasos")
    print(f"Semillas: {SEMILLAS}")
    
    print(f"Resultados: {ARCHIVO}\n")

    print("FASE 1")
    tabla_1 = ejecutar_fase(1, combos_1, contexto)
    res_1 = resumen(tabla_1, ["alpha", "n_pasos"])
    print("\nRESUMEN FASE 1 (promedio de las semillas)")
    print(res_1.to_string(index=False))

    mejor = res_1.sort_values("ventaja_capturada_media", ascending=False).iloc[0]
    alpha_b, n_b = float(mejor["alpha"]), int(mejor["n_pasos"])
    print(f"\nMejor combinación de la fase 1: alpha = {alpha_b}, n_pasos = {n_b:,} "
          f"(ventaja capturada {mejor['ventaja_capturada_media']:.1f} %)")

    print("\nFASE 2")
    # Los epsilon que ya se corrieron en la fase 1 se reutilizan, no se repiten.
    repetidos = tabla_1[(tabla_1["alpha"] == alpha_b) & (tabla_1["n_pasos"] == n_b)
                        & (tabla_1["epsilon"].isin(FASE_2_EPSILON))]
    combos_2 = [(alpha_b, eps, n_b) for eps in FASE_2_EPSILON
                if eps not in set(repetidos["epsilon"])]
    if len(repetidos):
        print(f"  (epsilon {sorted(set(repetidos['epsilon']))} se toma de la fase 1)")
    tabla_2 = pd.concat([repetidos, ejecutar_fase(2, combos_2, contexto)],
                        ignore_index=True)
    res_2 = resumen(tabla_2, ["epsilon"])
    print("\nRESUMEN FASE 2 (promedio de las semillas)")
    print(res_2.to_string(index=False))

    print(f"\nTodas las corridas quedaron en: {ARCHIVO}")
    print("Lectura: 100 % = costo mínimo del MDP; 0 % = igual a la regla miope; "
          "negativo = peor que la miope.")


if __name__ == "__main__":
    main()

"""
Por: Juanita Valentina Cortés Vargas

Sensibilidad de Q-learning (alpha fijo) en el MDP CONJUNTO N-P-K, con histórico.

Uso, desde la carpeta jmarkov:
    py examples/qlearning_fertilizantes/sensibilidad_conjunto.py
    py examples/qlearning_fertilizantes/sensibilidad_conjunto.py --semillas 3 --procesos 4

Misma lógica que sensibilidad_ampliada.py, aplicada a qlearning_conjunto.py:
- Rejilla completa: prueba TODAS las combinaciones de VALORES_ALPHA x
  VALORES_EPSILON (10 x 10 = 100 combinaciones).
- Cada combinación se corre con varias semillas, en paralelo.
- Cada REGISTRAR_CADA pasos calcula el costo EXACTO de la política conjunta del
  momento sobre los 3375 estados (histórico). Si algún estado todavía no se ha
  visitado, se le asigna la acción de la regla miope y se registra cuántos son.
- Intervalos de confianza del 95 %  del costo final.

Resultados:
    outputs/tables/qlearning_conjunto/historico_sensibilidad_conjunto.csv
    outputs/tables/qlearning_conjunto/resumen_sensibilidad_conjunto_ic95.csv
Gráficas: py examples/qlearning_fertilizantes/graficas_sensibilidad.py
"""
from pathlib import Path
import argparse
import itertools
import multiprocessing as mp
import os
import sys
import time

import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# DISEÑO
# -----------------------------------------------------------------------------
# La base solo se usa para las gráficas de cortes; el diseño corre la rejilla completa.
CONFIG_BASE = {"alpha": 0.03, "epsilon": 0.7}
VALORES_ALPHA = [0.001, 0.002, 0.003, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2]
VALORES_EPSILON = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9, 1.0]
N_PASOS = 50_000_000         # longitud de cada entrenamiento
REGISTRAR_CADA = 1_000_000   # cada cuántos pasos se evalúa la política (histórico)
N_SEMILLAS = 3               # semillas 1, 2, ..., N_SEMILLAS
MICROSEG_POR_PASO = 23.0     # solo para estimar la duración

CARPETA = Path(__file__).resolve().parent
sys.path.insert(0, str(CARPETA))
from sensibilidad_ampliada import (intervalo,intervalo_t,NIVEL_CONFIANZA,   # noqa: E402
                                   imprimir_rejilla)

SALIDA = Path(__file__).resolve().parents[3] / "outputs" / "tables" / "qlearning_conjunto"
ARCHIVO_HISTORICO = SALIDA / "historico_sensibilidad_conjunto.csv"
ARCHIVO_RESUMEN = SALIDA / "resumen_sensibilidad_conjunto_ic95.csv"
COLUMNAS = ["commodity", "alpha", "epsilon", "semilla", "paso", "estados_visitados",
            "estados_sin_accion", "pares_actualizados", "costo_USD", "sobrecosto_USD",
            "sobrecosto_%", "ventaja_capturada_%", "coincidencia_%", "tiempo_corrida_s"]


# -----------------------------------------------------------------------------
# CONTEXTO (se construye una vez por proceso)
# -----------------------------------------------------------------------------
_CTX = {}


def contexto():
    if _CTX:
        return _CTX
    import qlearning_conjunto as qc
    original, piezas = qc.construir_piezas()
    initial_state, feasible_actions, step = qc.preparar_qlearning(piezas)
    optima, miope = qc.politicas_de_referencia(piezas)
    _CTX.update(qc=qc, original=original, piezas=piezas, initial_state=initial_state,
                feasible_actions=feasible_actions, step=step, optima=optima, miope=miope,
                estados=qc.estados_conjuntos(piezas),
                V_opt=qc.costo_politica_conjunta(original, piezas, optima),
                V_miope=qc.costo_politica_conjunta(original, piezas, miope))
    return _CTX


def evaluar(ctx, politica):
    """Costo exacto (costo original) de la política conjunta actual."""
    sin_accion = [s for s in ctx["estados"] if s not in politica]
    completa = dict(politica)
    for s in sin_accion:
        completa[s] = ctx["miope"][s]
    V = ctx["qc"].costo_politica_conjunta(ctx["original"], ctx["piezas"], completa)
    sobrecosto = float(np.mean(V - ctx["V_opt"]))
    ventaja_miope = float(np.mean(ctx["V_miope"] - ctx["V_opt"]))
    return {
        "estados_sin_accion": len(sin_accion),
        "costo_USD": float(np.mean(V)),
        "sobrecosto_USD": sobrecosto,
        "sobrecosto_%": 100 * sobrecosto / float(np.mean(ctx["V_opt"])),
        "ventaja_capturada_%": 100 * (1 - sobrecosto / ventaja_miope),
        "coincidencia_%": 100 * float(np.mean(
            [politica.get(s) == ctx["optima"][s] for s in ctx["estados"]])),
    }


def entrenar_con_historico(tarea):
    """Una corrida: mismo ciclo que qlearning.solve, registrando la política."""
    alpha, epsilon, semilla = tarea
    ctx = contexto()
    from jmarkov.mdp.qlearning import qlearning
    modelo = qlearning(ctx["initial_state"], ctx["feasible_actions"], ctx["step"],
                       ctx["original"].BETA)
    inicio = time.perf_counter()

    # --- Igual que qlearning.solve (alpha y epsilon fijos, minimize=True) ---
    rng = np.random.default_rng(semilla)
    modelo.Q = {}
    state = modelo.initial_state(rng)
    modelo._initialize_state(state)
    modelo.visit_counts = {}
    filas = []
    for paso in range(1, N_PASOS + 1):
        action = modelo._select_action(state, epsilon, rng)
        next_state, costo = modelo.step(state, action, rng)
        modelo._update_q(state, action, -costo, next_state, alpha)
        pair = (state, action)
        modelo.visit_counts[pair] = modelo.visit_counts.get(pair, 0) + 1
        state = next_state
        # --- Registro del histórico (no modifica Q) ---
        if paso % REGISTRAR_CADA == 0:
            fila = {"commodity": "NPK", "alpha": alpha, "epsilon": epsilon,
                    "semilla": semilla, "paso": paso,
                    "estados_visitados": len({s for (s, _) in modelo.visit_counts}),
                    "pares_actualizados": len(modelo.visit_counts)}
            fila.update(evaluar(ctx, modelo.get_policy()))
            filas.append(fila)
    tiempo = time.perf_counter() - inicio
    for fila in filas:
        fila["tiempo_corrida_s"] = round(tiempo, 1)
    return filas


# -----------------------------------------------------------------------------
# DISEÑO, EJECUCIÓN Y RESUMEN
# -----------------------------------------------------------------------------
def tareas_del_diseno(semillas):
    """Rejilla completa: todas las combinaciones (alpha, epsilon)."""
    combos = list(itertools.product(VALORES_ALPHA, VALORES_EPSILON))
    return [(a, e, s) for (a, e), s in itertools.product(combos, semillas)]


def ya_corridas():
    if not ARCHIVO_HISTORICO.exists():
        return set()
    h = pd.read_csv(ARCHIVO_HISTORICO)
    completas = h[h["paso"] == N_PASOS]
    return {(float(r.alpha), float(r.epsilon), int(r.semilla)) for r in completas.itertuples()}


def guardar(filas):
    SALIDA.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(filas, columns=COLUMNAS).to_csv(
        ARCHIVO_HISTORICO, mode="a", index=False,
        header=not ARCHIVO_HISTORICO.exists(), encoding="utf-8-sig")


def en_rejilla(datos, semillas):
    """Filas de la rejilla actual y de las semillas 1..n (todas las celdas con el mismo n)."""
    a = datos["alpha"].apply(lambda v: any(np.isclose(v, VALORES_ALPHA)))
    e = datos["epsilon"].apply(lambda v: any(np.isclose(v, VALORES_EPSILON)))
    return datos[a & e & datos["semilla"].isin(semillas)]


def resumir(semillas):
    """IC95 del costo final para cada combinación (alpha, epsilon) de la rejilla."""
    h = pd.read_csv(ARCHIVO_HISTORICO)
    final = en_rejilla(h[h["paso"] == N_PASOS], semillas)
    filas = []
    for (a, e), g in final.groupby(["alpha", "epsilon"]):
        fila = {"commodity": "NPK", "alpha": a, "epsilon": e, "n_pasos": N_PASOS}
        for metrica in ["costo_USD", "sobrecosto_USD", "ventaja_capturada_%",
                        "estados_sin_accion"]:
            n, media, s, li, ls = intervalo(g[metrica])
            ti, ts = intervalo_t(g[metrica])
            fila.update({f"{metrica}_media": media, f"{metrica}_desv": s,
                         f"{metrica}_ic_inf": li, f"{metrica}_ic_sup": ls,
                         f"{metrica}_ic_t_inf": ti, f"{metrica}_ic_t_sup": ts})
        fila["n_semillas"] = n
        filas.append(fila)
    resumen = pd.DataFrame(filas)
    resumen.to_csv(ARCHIVO_RESUMEN, index=False, encoding="utf-8-sig")
    return resumen


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--semillas", type=int, default=N_SEMILLAS)
    parser.add_argument("--procesos", type=int, default=max(1, (os.cpu_count() or 2) - 1),
                        help="corridas en paralelo (por defecto, núcleos - 1)")
    args = parser.parse_args()

    semillas = list(range(1, args.semillas + 1))
    todas = tareas_del_diseno(semillas)
    hechas = ya_corridas()
    pendientes = [t for t in todas if (float(t[0]), float(t[1]), t[2]) not in hechas]
    procesos = max(1, min(args.procesos, len(pendientes)))
    tandas = -(-len(pendientes) // procesos) if pendientes else 0
    horas = tandas * N_PASOS * MICROSEG_POR_PASO / 1e6 / 3600

    print("=" * 78)
    print("SENSIBILIDAD DE Q-LEARNING — MDP CONJUNTO N-P-K (alpha fijo, costo reformulado)")
    print("=" * 78)
    print(f"  alpha probados:   {VALORES_ALPHA}")
    print(f"  epsilon probados: {VALORES_EPSILON}")
    print(f"  rejilla: {len(VALORES_ALPHA)} x {len(VALORES_EPSILON)} = "
          f"{len(VALORES_ALPHA) * len(VALORES_EPSILON)} combinaciones")
    print(f"  pasos: {N_PASOS:,} | histórico cada {REGISTRAR_CADA:,} pasos | "
          f"semillas: 1 a {args.semillas} | IC: {NIVEL_CONFIANZA:.0%} (bootstrap)")
    print(f"  corridas: {len(todas)} en total, {len(todas) - len(pendientes)} ya hechas, "
          f"{len(pendientes)} pendientes")
    print(f"  procesos en paralelo: {procesos} | duración estimada: ~{horas:.1f} h "
          f"(cada corrida ~{N_PASOS * MICROSEG_POR_PASO / 1e6 / 60:.0f} min)")
    print("  Las corridas avanzan en silencio; se imprime una línea al terminar cada una.")
    print(f"  resultados: {ARCHIVO_HISTORICO}\n")

    inicio = time.perf_counter()
    if pendientes:
        with mp.Pool(procesos) as pool:
            for k, filas in enumerate(pool.imap_unordered(entrenar_con_historico, pendientes),
                                      start=1):
                guardar(filas)
                f = filas[-1]
                transcurrido = time.perf_counter() - inicio
                restante = transcurrido / k * (len(pendientes) - k)
                print(f"  [{k}/{len(pendientes)}] alpha={f['alpha']:<6} epsilon={f['epsilon']:<5} "
                      f"semilla={f['semilla']:<2} -> sobrecosto USD {f['sobrecosto_USD']:>12,.0f} | "
                      f"ventaja capturada {f['ventaja_capturada_%']:6.1f} % | "
                      f"estados sin acción {f['estados_sin_accion']} | "
                      f"faltan ~{restante / 3600:.1f} h", flush=True)

    resumen = resumir(semillas)
    print("\nRESUMEN: sobrecosto final (USD), media e IC 95 % (bootstrap)")
    imprimir_rejilla(resumen, "NPK")
    tabla = resumen.pivot(index="alpha", columns="epsilon", values="estados_sin_accion_media")
    print("\n  NPK: estados sin acción al final (media)  [filas = alpha, columnas = epsilon]")
    print(tabla.round(1).to_string())
    print(f"\nResumen guardado en: {ARCHIVO_RESUMEN}")
    print("Para las gráficas: py examples/qlearning_fertilizantes/graficas_sensibilidad.py")


if __name__ == "__main__":
    main()

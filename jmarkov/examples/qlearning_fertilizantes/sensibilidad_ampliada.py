"""
Por: Juanita Valentina Cortés Vargas

Sensibilidad ampliada de Q-learning (alpha fijo) con muchas semillas e histórico.

Uso, desde la carpeta jmarkov:
    py examples/qlearning_fertilizantes/sensibilidad_ampliada.py            (N, P y K)
    py examples/qlearning_fertilizantes/sensibilidad_ampliada.py --commodity N
    py examples/qlearning_fertilizantes/sensibilidad_ampliada.py --semillas 10 --procesos 4

Qué hace:
- Rejilla completa: prueba TODAS las combinaciones de VALORES_ALPHA x
  VALORES_EPSILON (10 x 10 = 100 combinaciones por insumo).
- Corre cada combinación con varias semillas, para calcular intervalos de
  confianza del costo de la política aprendida.
- Durante el entrenamiento, cada REGISTRAR_CADA pasos toma la política actual
  y calcula su costo EXACTO: ese es el "histórico" (cómo evoluciona el costo
  mientras Q-learning aprende). Con él también se obtiene el efecto del número
  de pasos, sin correr de nuevo.

Resultados:
    outputs/tables/qlearning_independientes/historico_sensibilidad.csv
    outputs/tables/qlearning_independientes/resumen_sensibilidad_ic95.csv
Las gráficas se generan con graficas_sensibilidad.py.
"""
from pathlib import Path
import argparse
import importlib
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
# Configuración base de cada insumo (la elegida en la sensibilidad anterior).
# En la rejilla solo se usa para las gráficas de cortes (alpha con epsilon base
# y epsilon con alpha base); el diseño corre todas las combinaciones.
CONFIG_BASE = {
    "N": {"alpha": 0.001, "epsilon": 0.9},
    "P": {"alpha": 0.001, "epsilon": 0.4},
    "K": {"alpha": 0.001, "epsilon": 0.4},
}
VALORES_ALPHA = [0.001, 0.002, 0.003, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2]
VALORES_EPSILON = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9, 1.0]
N_PASOS = 3_000_000          # longitud de cada entrenamiento
REGISTRAR_CADA = 50_000      # cada cuántos pasos se evalúa la política (histórico)
N_SEMILLAS = 10              # semillas 1, 2, ..., N_SEMILLAS
NIVEL_CONFIANZA = 0.95
MICROSEG_POR_PASO = 9.0      # solo para estimar la duración

CARPETA = Path(__file__).resolve().parent
sys.path.insert(0, str(CARPETA))
SALIDA = Path(__file__).resolve().parents[3] / "outputs" / "tables" / "qlearning_independientes"
ARCHIVO_HISTORICO = SALIDA / "historico_sensibilidad.csv"
ARCHIVO_RESUMEN = SALIDA / "resumen_sensibilidad_ic95.csv"


# -----------------------------------------------------------------------------
# CONTEXTO DE CADA INSUMO (se construye una vez por proceso)
# -----------------------------------------------------------------------------
_CONTEXTOS = {}


def contexto(commodity):
    """MDP, funciones para Q-learning y costos de referencia de un insumo."""
    if commodity in _CONTEXTOS:
        return _CONTEXTOS[commodity]
    base = importlib.import_module(f"qlearning_{commodity}")   # incluye el ajuste de P
    original = base.cargar_mdp_independiente()
    data = original.construir_mdp(commodity)
    initial_state, feasible_actions, _ = base.preparar_qlearning(original, data)

    # step con la misma distribución que qlearning_c.py, pero más rápido:
    # sorteo con probabilidades acumuladas en vez de rng.choice.
    etiquetas = [str(a) for a in data["actions"]]
    columna = {a: j for j, a in enumerate(etiquetas)}
    acumuladas = {a: np.cumsum(data["transition_matrices"][data["actions"][j]], axis=1)
                  for j, a in enumerate(etiquetas)}
    for a in acumuladas:
        acumuladas[a][:, -1] = 1.0
    R = data["returns"]
    costo_demanda = base.costo_inevitable(original, data)   # costo reformulado

    def step(state, action, rng):
        siguiente = int(np.searchsorted(acumuladas[action][state], rng.random(),
                                        side="right"))
        return siguiente, float(R[state, columna[action]]) - costo_demanda[state]

    _, acciones_ref = base.resolver_referencia(original, data)
    V_opt = base.costo_politica(original, data, acciones_ref)
    miope = base.reglas_simples(original, data, feasible_actions)[
        "Regla miope (lo más barato hoy)"]
    V_miope = base.costo_politica(original, data, miope)
    _CONTEXTOS[commodity] = dict(base=base, original=original, data=data,
                                 initial_state=initial_state,
                                 feasible_actions=feasible_actions, step=step,
                                 acciones_ref=acciones_ref, V_opt=V_opt, V_miope=V_miope)
    return _CONTEXTOS[commodity]


# -----------------------------------------------------------------------------
# ENTRENAMIENTO CON HISTÓRICO
# -----------------------------------------------------------------------------
def evaluar(ctx, politica):
    """Costo exacto (costo original) de la política actual."""
    n = len(ctx["data"]["states"])
    if any(i not in politica for i in range(n)):
        return None
    acciones = [str(politica[i]) for i in range(n)]
    V = ctx["base"].costo_politica(ctx["original"], ctx["data"], acciones)
    sobrecosto = float(np.mean(V - ctx["V_opt"]))
    ventaja_miope = float(np.mean(ctx["V_miope"] - ctx["V_opt"]))
    return {
        "costo_USD": float(np.mean(V)),
        "sobrecosto_USD": sobrecosto,
        "sobrecosto_%": 100 * sobrecosto / float(np.mean(ctx["V_opt"])),
        "ventaja_capturada_%": 100 * (1 - sobrecosto / ventaja_miope),
        "coincidencia_%": 100 * float(np.mean(
            [a == r for a, r in zip(acciones, ctx["acciones_ref"])])),
    }


def entrenar_con_historico(tarea):
    """Una corrida: mismo ciclo que qlearning.solve, registrando la política."""
    commodity, alpha, epsilon, semilla = tarea
    ctx = contexto(commodity)
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
            metricas = evaluar(ctx, modelo.get_policy())
            fila = {"commodity": commodity, "alpha": alpha, "epsilon": epsilon,
                    "semilla": semilla, "paso": paso,
                    "estados_visitados": len({s for (s, _) in modelo.visit_counts})}
            fila.update(metricas or {})
            filas.append(fila)
    tiempo = time.perf_counter() - inicio
    for fila in filas:
        fila["tiempo_corrida_s"] = round(tiempo, 1)
    return filas


# -----------------------------------------------------------------------------
# DISEÑO, EJECUCIÓN Y RESUMEN
# -----------------------------------------------------------------------------
def tareas_del_diseno(commodities, semillas):
    """Rejilla completa: todas las combinaciones (alpha, epsilon) de cada insumo."""
    combos = list(itertools.product(VALORES_ALPHA, VALORES_EPSILON))
    return [(c, a, e, s) for c in commodities
            for (a, e), s in itertools.product(combos, semillas)]


def ya_corridas():
    if not ARCHIVO_HISTORICO.exists():
        return set()
    h = pd.read_csv(ARCHIVO_HISTORICO)
    completas = h[h["paso"] == N_PASOS]
    return {(r.commodity, float(r.alpha), float(r.epsilon), int(r.semilla))
            for r in completas.itertuples()}


COLUMNAS = ["commodity", "alpha", "epsilon", "semilla", "paso", "estados_visitados",
            "costo_USD", "sobrecosto_USD", "sobrecosto_%", "ventaja_capturada_%",
            "coincidencia_%", "tiempo_corrida_s"]


def guardar(filas):
    # Orden de columnas fijo: si en un registro temprano todavía falta algún
    # estado por visitar, sus métricas quedan vacías (NaN) sin desordenar el CSV.
    SALIDA.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(filas, columns=COLUMNAS).to_csv(ARCHIVO_HISTORICO, mode="a", index=False,
                               header=not ARCHIVO_HISTORICO.exists(),
                               encoding="utf-8-sig")


def intervalo(valores, n_remuestras=10_000):
    """Media e intervalo de confianza bootstrap (percentil) de la media.

    Se toman n_remuestras muestras con reemplazo de los resultados de las
    semillas, se calcula la media de cada una y el intervalo va del percentil
    2,5 al 97,5 de esas medias (para un nivel de 95 %). A diferencia del
    intervalo t, no supone simetría: como el sobrecosto es >= 0 y muy
    asimétrico (muchas semillas en 0 y algunos picos), el intervalo t puede
    dar límites negativos, que no tienen sentido.
    """
    x = np.asarray(valores, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    media = float(np.mean(x)) if n else np.nan
    if n < 2:
        return n, media, np.nan, np.nan, np.nan
    s = float(np.std(x, ddof=1))
    rng = np.random.default_rng(0)          # fijo: el intervalo es reproducible
    medias = rng.choice(x, size=(n_remuestras, n), replace=True).mean(axis=1)
    cola = 100 * (1 - NIVEL_CONFIANZA) / 2
    return n, media, s, float(np.percentile(medias, cola)), float(np.percentile(medias, 100 - cola))


def intervalo_t(valores):
    """Intervalo t de Student clásico (solo se guarda como referencia)."""
    from scipy import stats
    x = np.asarray(valores, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 2:
        return np.nan, np.nan
    media, s = float(np.mean(x)), float(np.std(x, ddof=1))
    t = stats.t.ppf(0.5 + NIVEL_CONFIANZA / 2, n - 1)
    return media - t * s / np.sqrt(n), media + t * s / np.sqrt(n)

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
    for (c, a, e), g in final.groupby(["commodity", "alpha", "epsilon"]):
        fila = {"commodity": c, "alpha": a, "epsilon": e, "n_pasos": N_PASOS}
        for metrica in ["costo_USD", "sobrecosto_USD", "ventaja_capturada_%"]:
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


def imprimir_rejilla(resumen, etiqueta):
    """Tabla alpha x epsilon con la ventaja capturada media (%) y las 5 mejores."""
    tabla = resumen.pivot(index="alpha", columns="epsilon", values="ventaja_capturada_%_media")
    print(f"\n  {etiqueta}: ventaja capturada media (%)  [filas = alpha, columnas = epsilon]")
    print(tabla.round(1).to_string())
    mejores = resumen.nsmallest(5, "sobrecosto_USD_media")
    print(f"\n  {etiqueta}: 5 mejores combinaciones (menor sobrecosto medio)")
    for _, r in mejores.iterrows():
        print(f"    alpha = {r['alpha']:<6} epsilon = {r['epsilon']:<5} media "
              f"{r['sobrecosto_USD_media']:>12,.0f}   IC [{r['sobrecosto_USD_ic_inf']:>12,.0f} ; "
              f"{r['sobrecosto_USD_ic_sup']:>12,.0f}]   ventaja capturada "
              f"{r['ventaja_capturada_%_media']:6.1f} %")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--commodity", nargs="+", default=["N", "P", "K"],
                        choices=["N", "P", "K"])
    parser.add_argument("--semillas", type=int, default=N_SEMILLAS)
    parser.add_argument("--procesos", type=int, default=max(1, (os.cpu_count() or 2) - 1),
                        help="corridas en paralelo (por defecto, núcleos - 1)")
    args = parser.parse_args()

    semillas = list(range(1, args.semillas + 1))
    todas = tareas_del_diseno(args.commodity, semillas)
    hechas = ya_corridas()
    pendientes = [t for t in todas if (t[0], float(t[1]), float(t[2]), t[3]) not in hechas]

    horas = len(pendientes) * N_PASOS * MICROSEG_POR_PASO / 1e6 / 3600 / args.procesos
    print("=" * 78)
    print("SENSIBILIDAD EN REJILLA DE Q-LEARNING (alpha fijo, costo reformulado)")
    print("=" * 78)
    print(f"  alpha probados:   {VALORES_ALPHA}")
    print(f"  epsilon probados: {VALORES_EPSILON}")
    print(f"  rejilla: {len(VALORES_ALPHA)} x {len(VALORES_EPSILON)} = "
          f"{len(VALORES_ALPHA) * len(VALORES_EPSILON)} combinaciones por insumo")
    print(f"  pasos: {N_PASOS:,} | histórico cada {REGISTRAR_CADA:,} pasos | "
          f"semillas: 1 a {args.semillas} | IC: {NIVEL_CONFIANZA:.0%}")
    print(f"  corridas: {len(todas)} en total, {len(todas) - len(pendientes)} ya hechas, "
          f"{len(pendientes)} pendientes")
    print(f"  procesos en paralelo: {args.procesos} | duración estimada: ~{horas:.1f} h")
    print(f"  resultados: {ARCHIVO_HISTORICO}\n")

    inicio = time.perf_counter()
    if pendientes:
        with mp.Pool(args.procesos) as pool:
            for k, filas in enumerate(pool.imap_unordered(entrenar_con_historico, pendientes),
                                      start=1):
                guardar(filas)
                f = filas[-1]
                transcurrido = time.perf_counter() - inicio
                restante = transcurrido / k * (len(pendientes) - k)
                print(f"  [{k}/{len(pendientes)}] {f['commodity']} alpha={f['alpha']:<6} "
                      f"epsilon={f['epsilon']:<5} semilla={f['semilla']:<3} -> "
                      f"sobrecosto USD {f.get('sobrecosto_USD', float('nan')):>12,.0f} | "
                      f"ventaja capturada {f.get('ventaja_capturada_%', float('nan')):6.1f} % | "
                      f"faltan ~{restante / 60:.0f} min", flush=True)

    resumen = resumir(semillas)
    print("\nRESUMEN: sobrecosto final (USD), media e IC 95 % (bootstrap)")
    for c, g in resumen.groupby("commodity", sort=False):
        imprimir_rejilla(g, c)
    print(f"\nResumen guardado en: {ARCHIVO_RESUMEN}")
    print("Para las gráficas: py examples/qlearning_fertilizantes/graficas_sensibilidad.py")


if __name__ == "__main__":
    main()

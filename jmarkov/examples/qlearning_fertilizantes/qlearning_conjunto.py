"""
Por: Juanita Valentina Cortés Vargas

Q-learning (Sutton y Barto, sec. 6.5) para el MDP CONJUNTO N-P-K.

Uso, desde la carpeta jmarkov:
    py examples/qlearning_fertilizantes/qlearning_conjunto.py

Qué hace:
1. Construye las piezas de los tres MDP independientes con las MISMAS funciones
   del script src/06_mdp_value_iteration_independientes*.py, con la matriz de
   precios del fósforo ajustada (FILA_MODIFICADA de qlearning_P.py).
2. Arma el problema conjunto para el módulo de Q-learning:
     estado  = (estado de N, estado de P, estado de K)   -> 15^3 = 3375 estados
     acción  = (compra de N, compra de P, compra de K)   -> 27 factibles por estado
     costo   = suma de los costos de los tres insumos
   El inventario siguiente de cada insumo es determinista y los tres precios
   evolucionan de forma independiente, igual que en el MDP conjunto del
   script 06A (su matriz de precios es el producto de Kronecker de las tres).
3. Entrena con jmarkov.mdp.qlearning tal como está (solve con alpha y epsilon
   fijos, Q inicial en 0, una sola trayectoria, sin reinicios).
4. Compara con la referencia. El MDP conjunto no se puede resolver con dtmdp
   (125 matrices de 3375 x 3375), pero el problema es separable: capacidades y
   compras están fijadas por insumo y el costo es aditivo. Por eso la política
   óptima conjunta es (pi*_N, pi*_P, pi*_K), obtenida con dtmdp en cada MDP
   independiente, y su costo se calcula de forma exacta sobre los 3375 estados.
5. Tres comparaciones, todas con el costo exacto sobre los 3375 estados:
     a) política óptima del MDP (por separabilidad);
     b) regla miope (lo más barato hoy);
     c) Q-learning SEPARADO por insumo: se entrenan qlearning_N, qlearning_P y
        qlearning_K con sus propios HIPERPARAMETROS y se combinan sus políticas,
        (pi_N, pi_P, pi_K), igual que la óptima.
"""
from pathlib import Path
import itertools
import sys
import time

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, identity
from scipy.sparse.linalg import spsolve

# -----------------------------------------------------------------------------
# HIPERPARÁMETROS 
# -----------------------------------------------------------------------------
HIPERPARAMETROS = {
    "n_pasos": 50_000_000,  # longitud de la trayectoria (número de experiencias)
    "alpha": 0.03,          # tamaño de paso, fijo
    "epsilon": 0.7,         # probabilidad de explorar, fija
    "semilla": 42,          # para reproducir la corrida
}

# Costo que ve Q-learning (igual que en los modelos independientes):
# True: precio * (compra - demanda), sumado sobre los tres insumos.
# La comparación con el óptimo se hace siempre con el costo original.
COSTO_REFORMULADO = True

MICROSEG_POR_PASO = 22.0    # solo para estimar la duración

# -----------------------------------------------------------------------------
# RUTAS E IMPORTACIONES
# -----------------------------------------------------------------------------
CARPETA = Path(__file__).resolve().parent
sys.path.insert(0, str(CARPETA))
import qlearning_P as base_P                       # noqa: E402  (aplica el ajuste de P)
from jmarkov.mdp.qlearning import qlearning        # noqa: E402

COMMODITIES = ("N", "P", "K")
SALIDA = Path(__file__).resolve().parents[3] / "outputs" / "tables" / "qlearning_conjunto"


# -----------------------------------------------------------------------------
# PIEZAS DE LOS TRES MDP INDEPENDIENTES
# -----------------------------------------------------------------------------
def construir_piezas():
    """MDP de cada insumo (con la matriz de P ajustada) y su referencia dtmdp."""
    original = base_P.cargar_mdp_independiente()   # script 06 con el ajuste de P
    piezas = {}
    for c in COMMODITIES:
        data = original.construir_mdp(c)
        etiquetas = [str(a) for a in data["actions"]]
        R = data["returns"]
        n = len(data["states"])
        factibles = {i: tuple(a for j, a in enumerate(etiquetas)
                              if R[i, j] < original.COSTO_INFACTIBLE) for i in range(n)}
        matrices = {a: np.asarray(data["transition_matrices"][data["actions"][j]])
                    for j, a in enumerate(etiquetas)}
        precios = original.cargar_precios_representativos(c)
        inevitable = np.array([float(precios.loc[cat]) * original.DEMANDA[c]
                               / original.COST_SCALE for (_, cat) in data["state_tuples"]])
        _, acciones_ref = base_P.resolver_referencia(original, data)
        piezas[c] = dict(data=data, etiquetas=etiquetas, columna={a: j for j, a in enumerate(etiquetas)},
                         R=R, n=n, factibles=factibles, matrices=matrices,
                         inevitable=inevitable if COSTO_REFORMULADO else np.zeros(n),
                         acciones_ref=acciones_ref,
                         compras={str(a): float(q) for a, q in zip(data["actions"], data["compras"])})
    return original, piezas


# -----------------------------------------------------------------------------
# LAS TRES FUNCIONES QUE RECIBE EL MÓDULO DE Q-LEARNING
# -----------------------------------------------------------------------------
def preparar_qlearning(piezas):
    """Estado: (i_N, i_P, i_K), índices de los estados independientes (0..14).
    Acción: (a_N, a_P, a_K), etiquetas de compra de cada insumo."""
    pN, pP, pK = (piezas[c] for c in COMMODITIES)
    acumuladas = {c: {a: np.cumsum(M, axis=1) for a, M in piezas[c]["matrices"].items()}
                  for c in COMMODITIES}
    for c in COMMODITIES:
        for a in acumuladas[c]:
            acumuladas[c][a][:, -1] = 1.0
    acciones = {}

    def initial_state(rng):
        # "Inicializar S": un estado conjunto al azar, una sola vez.
        return tuple(int(rng.integers(piezas[c]["n"])) for c in COMMODITIES)

    def feasible_actions(state):
        # Las compras factibles de cada insumo dependen solo de su inventario.
        if state not in acciones:
            acciones[state] = tuple(itertools.product(
                pN["factibles"][state[0]], pP["factibles"][state[1]], pK["factibles"][state[2]]))
        return acciones[state]

    def step(state, action, rng):
        # Cada insumo pasa a su estado siguiente con la fila de su matriz:
        # inventario determinista y precio sorteado de forma independiente.
        u = rng.random(3)
        siguiente = []
        costo = 0.0
        for k, c in enumerate(COMMODITIES):
            p = piezas[c]
            fila = acumuladas[c][action[k]][state[k]]
            siguiente.append(int(np.searchsorted(fila, u[k], side="right")))
            costo += float(p["R"][state[k], p["columna"][action[k]]]) - p["inevitable"][state[k]]
        return tuple(siguiente), costo

    return initial_state, feasible_actions, step


# -----------------------------------------------------------------------------
# COSTO EXACTO DE UNA POLÍTICA CONJUNTA
# -----------------------------------------------------------------------------
def estados_conjuntos(piezas):
    return list(itertools.product(*(range(piezas[c]["n"]) for c in COMMODITIES)))


def costo_politica_conjunta(original, piezas, politica):
    """Resuelve (I - beta P_pi) V = c_pi sobre los 3375 estados (costo original, USD).

    La fila de P_pi del estado (i, j, k) con acción (a, b, c) es el producto de
    Kronecker de las filas de cada insumo, porque sus transiciones son independientes.
    """
    estados = estados_conjuntos(piezas)
    nN, nP, nK = (piezas[c]["n"] for c in COMMODITIES)
    filas, columnas, valores = [], [], []
    costos = np.empty(len(estados))
    for s_idx, s in enumerate(estados):
        a = politica[s]
        fila = np.kron(np.kron(piezas["N"]["matrices"][a[0]][s[0]],
                               piezas["P"]["matrices"][a[1]][s[1]]),
                       piezas["K"]["matrices"][a[2]][s[2]])
        nz = np.nonzero(fila)[0]
        filas.extend([s_idx] * len(nz))
        columnas.extend(nz.tolist())
        valores.extend(fila[nz].tolist())
        costos[s_idx] = sum(float(piezas[c]["R"][s[k], piezas[c]["columna"][a[k]]])
                            for k, c in enumerate(COMMODITIES))
    n = len(estados)
    P = csr_matrix((valores, (filas, columnas)), shape=(n, n))
    V = spsolve((identity(n, format="csr") - original.BETA * P).tocsc(), costos)
    return np.asarray(V) * original.COST_SCALE


def politica_qlearning_separado(original, piezas, miope_individual):
    """Entrena Q-learning en cada MDP independiente (qlearning_N/P/K.py).

    Usa las mismas funciones y HIPERPARAMETROS de cada script, así que cada
    política es la misma que se obtiene al correr ese script por separado.
    Devuelve la política de cada insumo (estado -> acción) y los hiperparámetros.
    """
    import importlib
    politicas, hiper = {}, {}
    for c in COMMODITIES:
        base = importlib.import_module(f"qlearning_{c}")
        h = base.HIPERPARAMETROS
        initial_state, feasible_actions, step = base.preparar_qlearning(original, piezas[c]["data"])
        modelo = qlearning(initial_state, feasible_actions, step, original.BETA)
        _, pol = modelo.solve(n_steps=h["n_pasos"], alpha=h["alpha"], epsilon=h["epsilon"],
                              minimize=True, seed=h["semilla"])
        # Si algún estado individual no se visitó, se rellena con la miope (igual que en el conjunto).
        politicas[c] = {i: pol.get(i, miope_individual[c][i]) for i in range(piezas[c]["n"])}
        hiper[c] = h
    return politicas, hiper


def politicas_de_referencia(piezas):
    """Óptima (por separabilidad) y regla miope, para cada estado conjunto."""
    optima, miope = {}, {}
    for s in estados_conjuntos(piezas):
        optima[s] = tuple(piezas[c]["acciones_ref"][s[k]] for k, c in enumerate(COMMODITIES))
        miope[s] = tuple(min(piezas[c]["factibles"][s[k]],
                             key=lambda a, c=c, i=s[k]: piezas[c]["R"][i, piezas[c]["columna"][a]])
                         for k, c in enumerate(COMMODITIES))
    return optima, miope


# -----------------------------------------------------------------------------
# PRINCIPAL
# -----------------------------------------------------------------------------
def main():
    h = HIPERPARAMETROS
    original, piezas = construir_piezas()
    initial_state, feasible_actions, step = preparar_qlearning(piezas)
    estados = estados_conjuntos(piezas)
    pares = sum(len(feasible_actions(s)) for s in estados)

    print("=" * 78)
    print("Q-LEARNING (Sutton y Barto) — MDP CONJUNTO N-P-K")
    print("=" * 78)
    print(f"Estados: {len(estados):,} | pares estado-acción factibles: {pares:,} | "
          f"beta: {original.BETA:.6f}")
    print("Matriz de precios de P: ajustada (FILA_MODIFICADA de qlearning_P.py)")
    print("\nHIPERPARÁMETROS")
    for clave, valor in h.items():
        print(f"  {clave:<10}{valor:,}" if isinstance(valor, int) else f"  {clave:<10}{valor}")
    print("\nCOSTO QUE VE Q-LEARNING: " + ("REFORMULADO  suma de precio * (compra - demanda)"
                                         if COSTO_REFORMULADO else "ORIGINAL  suma de precio * compra"))
    print(f"Experiencias promedio por par estado-acción: {h['n_pasos'] / pares:,.0f}")
    print(f"Duración estimada del entrenamiento: ~{h['n_pasos'] * MICROSEG_POR_PASO / 1e6 / 60:.0f} "
          "minutos (el módulo no muestra avance mientras entrena)")

    print("\nCalculando la referencia (dtmdp en cada MDP independiente)...")
    optima, miope = politicas_de_referencia(piezas)
    V_opt = costo_politica_conjunta(original, piezas, optima)
    V_miope = costo_politica_conjunta(original, piezas, miope)

    print("Entrenando Q-learning separado (qlearning_N, qlearning_P, qlearning_K)...")
    miope_ind = {c: {i: min(piezas[c]["factibles"][i],
                            key=lambda a, c=c, i=i: piezas[c]["R"][i, piezas[c]["columna"][a]])
                     for i in range(piezas[c]["n"])} for c in COMMODITIES}
    pol_sep, hiper_sep = politica_qlearning_separado(original, piezas, miope_ind)
    separada = {s: tuple(pol_sep[c][s[k]] for k, c in enumerate(COMMODITIES)) for s in estados}
    V_sep = costo_politica_conjunta(original, piezas, separada)
    for c in COMMODITIES:
        hs = hiper_sep[c]
        print(f"  {c}: n_pasos {hs['n_pasos']:,} | alpha {hs['alpha']} | "
              f"epsilon {hs['epsilon']} | semilla {hs['semilla']}")

    print("Entrenando Q-learning conjunto...")
    modelo = qlearning(initial_state, feasible_actions, step, original.BETA)
    inicio = time.perf_counter()
    _, politica = modelo.solve(n_steps=h["n_pasos"], alpha=h["alpha"],
                               epsilon=h["epsilon"], minimize=True, seed=h["semilla"])
    tiempo = time.perf_counter() - inicio

    visitados = {s for (s, _) in modelo.visit_counts}
    visitas_estado = {s: 0 for s in estados}
    for (s, _), v in modelo.visit_counts.items():
        visitas_estado[s] += v
    pares_actualizados = len(modelo.visit_counts)
    sin_accion = [s for s in estados if s not in politica]

    print(f"\nTiempo de entrenamiento: {tiempo / 60:.1f} min "
          f"({1e6 * tiempo / h['n_pasos']:.1f} µs por paso)")
    print("\nRESUMEN")
    print(f"  Estados visitados: {len(visitados):,} de {len(estados):,}")
    print(f"  Pares estado-acción actualizados: {pares_actualizados:,} de {pares:,}")
    v = np.array(list(visitas_estado.values()))
    print(f"  Actualizaciones por estado (mín / mediana / máx): "
          f"{v.min():,} / {int(np.median(v)):,} / {v.max():,}")

    # Para calcular el costo de una política se necesita una acción en TODOS los
    # estados. Si la trayectoria no pasó por algunos (no se fuerza la cobertura),
    # se les asigna la acción de la regla miope y se informa cuántos fueron.
    politica_evaluada = dict(politica)
    for s in sin_accion:
        politica_evaluada[s] = miope[s]
    if sin_accion:
        print(f"\n  {len(sin_accion):,} estados nunca aparecieron en la trayectoria y no tienen "
              "acción aprendida.\n  Para calcular el costo se les asigna la acción de la "
              f"regla miope: {sin_accion[:10]}" + (" ..." if len(sin_accion) > 10 else ""))
    V_ql = costo_politica_conjunta(original, piezas, politica_evaluada)

    coincide = np.array([politica.get(s) == optima[s] for s in estados])
    por_insumo = {c: np.mean([s in politica and politica[s][k] == optima[s][k] for s in estados])
                  for k, c in enumerate(COMMODITIES)}
    print(f"  Coincidencia de la acción conjunta: {coincide.sum():,} de {len(estados):,} "
          f"({100 * coincide.mean():.1f} %)")
    print("  Coincidencia por insumo: " + " | ".join(
        f"{c} {100 * p:.1f} %" for c, p in por_insumo.items()))

    print("\nCOSTO DE LA POLÍTICA vs COSTO MÍNIMO (promedio sobre los 3375 estados, USD)")
    print(f"  {'Política':<36}{'Costo':>18}{'Sobrecosto':>16}{'%':>10}")
    print(f"  {'Óptima (MDP por separabilidad)':<36}{V_opt.mean():>18,.0f}{0:>16,.0f}{0:>10.4f}")
    etiqueta_ql = "Q-learning" + (f" ({len(sin_accion)} est. con miope)" if sin_accion else "")
    candidatas = {etiqueta_ql: V_ql,
                  "Q-learning separado (N, P, K)": V_sep,
                  "Regla miope (lo más barato hoy)": V_miope}
    for nombre, V in candidatas.items():
        brecha = V - V_opt
        print(f"  {nombre:<36}{V.mean():>18,.0f}{brecha.mean():>16,.0f}"
              f"{100 * brecha.mean() / V_opt.mean():>10.4f}")
    ventaja = (V_miope - V_opt).mean()
    print(f"\n  Ventaja de la política óptima sobre la regla miope: USD {ventaja:,.0f}")
    print(f"  Parte de esa ventaja que captura Q-learning conjunto: "
          f"{100 * (1 - (V_ql - V_opt).mean() / ventaja):.1f} %")
    print(f"  Parte de esa ventaja que captura Q-learning separado: "
          f"{100 * (1 - (V_sep - V_opt).mean() / ventaja):.1f} %")
    print("  (100 % = óptimo; 0 % = igual a la miope; negativo = peor)")

    # CSV con la política estado por estado.
    SALIDA.mkdir(parents=True, exist_ok=True)
    filas = []
    for s_idx, s in enumerate(estados):
        fila = {}
        for k, c in enumerate(COMMODITIES):
            inv, cat = piezas[c]["data"]["state_tuples"][s[k]]
            fila[f"I_{c}_ton"] = round(inv, 2)
            fila[f"cat_{c}"] = cat
        for k, c in enumerate(COMMODITIES):
            fila[f"compra_{c}_qlearning_ton"] = (round(piezas[c]["compras"][politica[s][k]], 2)
                                                 if s in politica else None)
            fila[f"compra_{c}_separado_ton"] = round(piezas[c]["compras"][separada[s][k]], 2)
            fila[f"compra_{c}_optima_ton"] = round(piezas[c]["compras"][optima[s][k]], 2)
        fila["visitas"] = visitas_estado[s]
        fila["coincide"] = bool(coincide[s_idx])
        fila["costo_politica_qlearning_USD"] = None if V_ql is None else V_ql[s_idx]
        fila["costo_politica_separado_USD"] = V_sep[s_idx]
        fila["costo_minimo_USD"] = V_opt[s_idx]
        fila["sobrecosto_USD"] = None if V_ql is None else V_ql[s_idx] - V_opt[s_idx]
        filas.append(fila)
    archivo = SALIDA / "qlearning_conjunto_politica.csv"
    pd.DataFrame(filas).to_csv(archivo, index=False, encoding="utf-8-sig")
    print(f"\nResultados guardados en: {archivo}")


if __name__ == "__main__":
    main()

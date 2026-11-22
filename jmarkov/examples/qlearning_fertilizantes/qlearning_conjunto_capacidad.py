"""
Por: Juanita Valentina Cortés Vargas

Q-learning conjunto N-P-K con UNA CAPACIDAD PARA TODA LA PLANTA.
Copia de qlearning_conjunto.py con estos cambios (el resto es igual):
1. feasible_actions: además de las condiciones de cada insumo (cubrir la
   demanda y caer en la malla de inventario), la suma de inventario + compra de
   los tres insumos no puede superar la capacidad total C de la planta.
2. initial_state: el estado inicial se elige entre los estados cuyo inventario
   total cabe en la capacidad (inventario total <= C - demanda total).
3. Se prueban varias capacidades (REDUCCIONES_PCT). (2) se entrena una vez y
   (3) una vez por capacidad, en paralelo. Para cada capacidad se comparan, con la
   misma tabla de qlearning_conjunto.py:
     - óptimo del MDP (unión de las tres políticas óptimas, capacidad por insumo);
     - (1) Q-learning separado (N, P, K) unificado, capacidad por insumo;
     - (2) Q-learning conjunto, capacidad por insumo;
     - (3) Q-learning conjunto, capacidad conjunta;
     - regla miope.

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
    "alpha": 0.03,          # tamaño de paso, fijo (elegido con sensibilidad_conjunto.py)
    "epsilon": 0.7,         # probabilidad de explorar, fija
    "semilla": 42,          # para reproducir la corrida
}

# Reducciones de la capacidad total de la planta (66 000 t), en %. Con 10 % o
# más la planta no alcanza ni para la demanda de un mes (59 451 t).
REDUCCIONES_PCT = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
TOL = 1e-6

# Costo que ve Q-learning (igual que en los modelos independientes):
# True: precio * (compra - demanda), sumado sobre los tres insumos.
# La comparación con el óptimo se hace siempre con el costo original.
COSTO_REFORMULADO = True

MICROSEG_POR_PASO = 26.0    # solo para estimar la duración

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
def preparar_qlearning(piezas, capacidad=None, demanda_total=None):
    """Estado: (i_N, i_P, i_K), índices de los estados independientes (0..14).
    Acción: (a_N, a_P, a_K), etiquetas de compra de cada insumo.

    Si capacidad es None, el problema es el de qlearning_conjunto.py (capacidad
    por insumo). Si se da una capacidad total C, se agrega la restricción
    conjunta: sum_c (inventario_c + compra_c) <= C.
    """
    pN, pP, pK = (piezas[c] for c in COMMODITIES)
    inv = {c: [inv for inv, _ in piezas[c]["data"]["state_tuples"]] for c in COMMODITIES}
    if capacidad is not None:
        # CAMBIO 2: estados cuyo inventario total cabe en la capacidad.
        iniciales = {s for s in itertools.product(*(range(piezas[c]["n"]) for c in COMMODITIES))
                     if sum(inv[c][s[k]] for k, c in enumerate(COMMODITIES))
                     <= capacidad - demanda_total + TOL}
    acumuladas = {c: {a: np.cumsum(M, axis=1) for a, M in piezas[c]["matrices"].items()}
                  for c in COMMODITIES}
    for c in COMMODITIES:
        for a in acumuladas[c]:
            acumuladas[c][a][:, -1] = 1.0
    acciones = {}

    def initial_state(rng):
        # "Inicializar S": un estado conjunto al azar, una sola vez.
        # Mismo sorteo que qlearning_conjunto.py; con capacidad conjunta se repite
        # hasta obtener un estado que quepa (con 66 000 t siempre cabe al primer
        # intento, y (3) sigue exactamente la misma trayectoria que (2)).
        while True:
            s = tuple(int(rng.integers(piezas[c]["n"])) for c in COMMODITIES)
            if capacidad is None or s in iniciales:
                return s

    def feasible_actions(state):
        # Las compras factibles de cada insumo dependen solo de su inventario.
        if state not in acciones:
            combinaciones = itertools.product(
                pN["factibles"][state[0]], pP["factibles"][state[1]], pK["factibles"][state[2]])
            if capacidad is not None:
                # CAMBIO 1: lo almacenado de los tres insumos cabe en la planta.
                combinaciones = [a for a in combinaciones
                                 if sum(inv[c][state[k]] + piezas[c]["compras"][a[k]]
                                        for k, c in enumerate(COMMODITIES)) <= capacidad + TOL]
            acciones[state] = tuple(combinaciones)
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
# ENTRENAMIENTO (una corrida por proceso)
# -----------------------------------------------------------------------------
def entrenar(tarea):
    """tarea = ("individual", None) -> (2) Q-learning conjunto con capacidad por insumo.
    tarea = ("conjunta", r)        -> (3) Q-learning conjunto con capacidad reducida r %."""
    tipo, r = tarea
    original, piezas = construir_piezas()
    demanda_total = sum(original.DEMANDA[c] for c in COMMODITIES)
    capacidad = None if tipo == "individual" else original.CAPACIDAD_TOTAL * (1 - r / 100)
    initial_state, feasible_actions, step = preparar_qlearning(piezas, capacidad, demanda_total)
    h = HIPERPARAMETROS
    modelo = qlearning(initial_state, feasible_actions, step, original.BETA)
    inicio = time.perf_counter()
    _, politica = modelo.solve(n_steps=h["n_pasos"], alpha=h["alpha"],
                               epsilon=h["epsilon"], minimize=True, seed=h["semilla"])
    tiempo = time.perf_counter() - inicio
    visitados = len({s for (s, _) in modelo.visit_counts})
    return tipo, r, politica, visitados, len(modelo.visit_counts), tiempo


# -----------------------------------------------------------------------------
# PRINCIPAL
# -----------------------------------------------------------------------------
def main():
    import argparse
    import multiprocessing as mp
    import os

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--procesos", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--pasos", type=int, default=HIPERPARAMETROS["n_pasos"],
                        help="pasos de Q-learning conjunto (solo para pruebas rápidas)")
    args = parser.parse_args()
    HIPERPARAMETROS["n_pasos"] = args.pasos
    h = HIPERPARAMETROS

    original, piezas = construir_piezas()
    demanda_total = sum(original.DEMANDA[c] for c in COMMODITIES)
    estados = estados_conjuntos(piezas)
    inventario_total = np.array([sum(piezas[c]["data"]["state_tuples"][s[k]][0]
                                     for k, c in enumerate(COMMODITIES)) for s in estados])

    print("=" * 78)
    print("Q-LEARNING CONJUNTO CON CAPACIDAD PARA TODA LA PLANTA — MDP N-P-K")
    print("=" * 78)
    print(f"Demanda total: {demanda_total:,.0f} t/mes | capacidad base: "
          f"{original.CAPACIDAD_TOTAL:,.0f} t | reducciones: {REDUCCIONES_PCT} %")
    print("\nHIPERPARÁMETROS (Q-learning conjunto)")
    for clave, valor in h.items():
        print(f"  {clave:<10}{valor:,}" if isinstance(valor, int) else f"  {clave:<10}{valor}")
    tareas = [("individual", None)] + [("conjunta", r) for r in REDUCCIONES_PCT]
    procesos = max(1, min(args.procesos, len(tareas)))
    print(f"Corridas de Q-learning conjunto: {len(tareas)} | procesos en paralelo: {procesos} | "
          f"~{h['n_pasos'] * MICROSEG_POR_PASO / 1e6 / 60:.0f} min cada una")

    # Referencias, igual que en qlearning_conjunto.py.
    print("\nCalculando la referencia (dtmdp en cada MDP independiente)...")
    optima, miope = politicas_de_referencia(piezas)
    V_opt = costo_politica_conjunta(original, piezas, optima)
    V_miope = costo_politica_conjunta(original, piezas, miope)

    print("Entrenando (1) Q-learning separado (qlearning_N, qlearning_P, qlearning_K)...")
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

    # (2) y (3): Q-learning conjunto, en paralelo.
    print("\nEntrenando (2) y (3) Q-learning conjunto (se imprime una línea por corrida)...")
    resultados = {}
    with mp.Pool(procesos) as pool:
        for tipo, r, politica, visitados, pares, tiempo in pool.imap_unordered(entrenar, tareas):
            resultados[(tipo, r)] = (politica, visitados, pares, tiempo)
            nombre = "capacidad por insumo" if tipo == "individual" else f"capacidad conjunta -{r}%"
            print(f"  listo: {nombre:<26} | estados visitados {visitados:,} | {tiempo / 60:.1f} min",
                  flush=True)

    # Para evaluar se necesita una acción en todos los estados: los no visitados
    # (y, en (3), los que no caben en la capacidad) reciben la regla miope, como
    # en qlearning_conjunto.py. Los estados que no caben no entran en el promedio.
    pol_ind, vis_ind, _, t_ind = resultados[("individual", None)]
    V_ql = costo_politica_conjunta(original, piezas,
                                   {s: pol_ind.get(s, miope[s]) for s in estados})

    filas = []
    print("\nCOMPARACIÓN (costo promedio sobre los estados que caben en cada capacidad, USD)")
    for r in REDUCCIONES_PCT:
        C = original.CAPACIDAD_TOTAL * (1 - r / 100)
        caben = inventario_total <= C - demanda_total + TOL
        pol_cap, vis_cap, pares_cap, t_cap = resultados[("conjunta", r)]
        V_cap = costo_politica_conjunta(original, piezas,
                                        {s: pol_cap.get(s, miope[s]) for s in estados})
        candidatas = {
            "Óptimo MDP (capacidad por insumo)": V_opt,
            "(1) QL separado (cap. por insumo)": V_sep,
            "(2) QL conjunto (cap. por insumo)": V_ql,
            "(3) QL conjunto (cap. conjunta)": V_cap,
            "Regla miope": V_miope,
        }
        ventaja = (V_miope - V_opt)[caben].mean()
        print(f"\n  Capacidad -{r}%: {C:,.0f} t | estados que caben: {caben.sum():,} de {len(estados):,} | "
              f"(3) visitó {vis_cap:,} estados | {t_cap / 60:.1f} min")
        print(f"    {'Política':<36}{'Costo':>18}{'Sobrecosto':>14}{'VC %':>9}")
        fila = {"reduccion_%": r, "capacidad_t": C, "estados_que_caben": int(caben.sum()),
                "estados_visitados_(3)": vis_cap, "pares_actualizados_(3)": pares_cap,
                "tiempo_(3)_min": round(t_cap / 60, 1)}
        for nombre, V in candidatas.items():
            costo = V[caben].mean()
            sobre = (V - V_opt)[caben].mean()
            vc = 100 * (1 - sobre / ventaja)
            print(f"    {nombre:<36}{costo:>18,.0f}{sobre:>14,.0f}{vc:>9.1f}")
            fila.update({f"costo {nombre} (USD)": costo, f"sobrecosto {nombre} (USD)": sobre,
                         f"VC {nombre} (%)": vc})
        filas.append(fila)

    salida = SALIDA.parent / "qlearning_conjunto_capacidad"
    salida.mkdir(parents=True, exist_ok=True)
    archivo = salida / "comparacion_capacidad_conjunta.csv"
    pd.DataFrame(filas).to_csv(archivo, index=False, encoding="utf-8-sig")
    print(f"\n(2) Q-learning conjunto con capacidad por insumo: {vis_ind:,} estados visitados, "
          f"{t_ind / 60:.1f} min")
    print("Sobrecosto y VC se miden frente al óptimo del MDP con capacidad por insumo.")
    print(f"Resultados guardados en: {archivo}")


if __name__ == "__main__":
    main()

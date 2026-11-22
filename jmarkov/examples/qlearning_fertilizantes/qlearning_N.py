"""
Por: Juanita Valentina Cortés Vargas

Q-learning (Sutton y Barto, sec. 6.5) para el MDP independiente del NITRÓGENO.

Uso, desde la carpeta jmarkov:
    py examples/qlearning_fertilizantes/qlearning_N.py

Qué hace:
1. Construye el MDP de N con las MISMAS funciones del script
   src/06_mdp_value_iteration_independientes*.py (estados, acciones,
   matrices de transición y costos).
2. Lo resuelve con dtmdp + value iteration de jMarkov: es la referencia.
3. Lo resuelve con el módulo jmarkov.mdp.qlearning, tal como está en el
   libro: alpha y epsilon fijos, Q inicial en 0 y UNA sola trayectoria,
   sin obligar a visitar todos los estados.
4. Compara la política y los valores aprendidos con los del MDP.
"""
from pathlib import Path
import importlib.util
import sys
import time

import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# HIPERPARÁMETROS 
# -----------------------------------------------------------------------------
COMMODITY = "N"

HIPERPARAMETROS = {
    "n_pasos": 3_000_000,   # longitud de la trayectoria (número de experiencias)
    "alpha": 0.001,          # tamaño de paso, fijo (elegido con sensibilidad_N.py)
    "epsilon": 0.9,         # probabilidad de explorar, fija
    "semilla": 42,          # para reproducir la corrida
}

# COSTO QUE VE Q-LEARNING
# False: costo original,     c(s, a)  = precio * compra
# True:  costo reformulado,  c'(s, a) = precio * compra - precio * demanda
#                                     = precio * (compra - demanda)
# Lo restado depende solo del precio, que no depende de la decisión, así que
# se resta lo mismo a TODAS las políticas: la política óptima no cambia.
# Solo cambia lo que ve Q-learning durante el aprendizaje; la comparación
# con el MDP se hace siempre con el costo original.
COSTO_REFORMULADO = True

# -----------------------------------------------------------------------------
# RUTAS E IMPORTACIONES
# -----------------------------------------------------------------------------
REPOSITORIO = Path(__file__).resolve().parents[2]      # carpeta jmarkov
PROYECTO = Path(__file__).resolve().parents[3]         # carpeta del proyecto
sys.path.insert(0, str(REPOSITORIO))
from jmarkov.mdp.qlearning import qlearning            # noqa: E402
from jmarkov.mdp.dtmdp import dtmdp                    # noqa: E402

SALIDA = PROYECTO / "outputs" / "tables" / "qlearning_independientes"


def cargar_mdp_independiente():
    """Importar el script 06 sin ejecutar su main()."""
    candidatos = sorted((PROYECTO / "src").glob(
        "06_mdp_value_iteration_independientes*.py"))
    if not candidatos:
        raise FileNotFoundError(
            "No se encontró src/06_mdp_value_iteration_independientes*.py")
    spec = importlib.util.spec_from_file_location("mdp_indep", candidatos[0])
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


# -----------------------------------------------------------------------------
# REFERENCIA: dtmdp + value iteration
# -----------------------------------------------------------------------------
def resolver_referencia(original, data):
    modelo = dtmdp(data["states"], data["actions"], data["transition_matrices"],
                   data["returns"], original.BETA)
    valores, politica = modelo.solve(original.TOLERANCIA, minimize=True,
                                     method="value_iteration")
    return np.asarray(valores, dtype=float), [str(politica[s]) for s in data["states"]]


# -----------------------------------------------------------------------------
# COSTO EXACTO DE UNA POLÍTICA
# -----------------------------------------------------------------------------
def costo_politica(original, data, acciones):
    """Costo descontado exacto de seguir una política, en USD.

    Se fija la acción de cada estado y se resuelve el sistema lineal
    (I - beta P_pi) V = c_pi (evaluación de política de programación
    dinámica). No hay simulación: el resultado es exacto.
    """
    etiquetas = [str(a) for a in data["actions"]]
    n = len(data["states"])
    P = np.array([data["transition_matrices"][data["actions"][etiquetas.index(a)]][i]
                  for i, a in enumerate(acciones)])
    c = np.array([data["returns"][i, etiquetas.index(a)] for i, a in enumerate(acciones)])
    return np.linalg.solve(np.eye(n) - original.BETA * P, c) * original.COST_SCALE

#reglas para hacer las comparaciones
def reglas_simples(original, data, feasible_actions):
    """Dos políticas de comparación que no miran el futuro."""
    etiquetas = [str(a) for a in data["actions"]]
    R = data["returns"]
    compras = {str(a): float(c) for a, c in zip(data["actions"], data["compras"])}
    n = len(data["states"])
    # Miope: la compra con menor costo inmediato.
    miope = [min(feasible_actions(i), key=lambda a: R[i, etiquetas.index(a)])
             for i in range(n)]
    # Comprar la demanda: la compra más cercana a la demanda mensual.
    demanda = [min(feasible_actions(i),
                   key=lambda a: abs(compras[a] - original.DEMANDA[COMMODITY]))
               for i in range(n)]
    return {"Regla miope (lo más barato hoy)": miope,
            "Comprar siempre la demanda": demanda}


# -----------------------------------------------------------------------------
# LAS TRES FUNCIONES QUE RECIBE EL MÓDULO DE Q-LEARNING
# -----------------------------------------------------------------------------
def costo_inevitable(original, data):
    """Costo de comprar la demanda al precio actual, por estado (unidades escaladas).

    Es lo que se resta cuando COSTO_REFORMULADO = True; si no, es cero.
    """
    if not COSTO_REFORMULADO:
        return np.zeros(len(data["states"]))
    precios = original.cargar_precios_representativos(COMMODITY)
    return np.array([float(precios.loc[cat]) * original.DEMANDA[COMMODITY]
                     / original.COST_SCALE for (_, cat) in data["state_tuples"]])


def preparar_qlearning(original, data):
    """Estado = índice 0..14 del MDP; acción = etiqueta de jMarkov (A, B, ...)."""
    etiquetas = [str(a) for a in data["actions"]]
    n = len(data["states"])
    R = data["returns"]
    columna = {a: j for j, a in enumerate(etiquetas)}

    # Acciones factibles: las que el MDP no penaliza con COSTO_INFACTIBLE.
    factibles = {i: tuple(a for j, a in enumerate(etiquetas)
                          if R[i, j] < original.COSTO_INFACTIBLE)
                 for i in range(n)}

    costo_demanda = costo_inevitable(original, data)

    def initial_state(rng):
        # "Inicializar S": un estado al azar, una sola vez.
        return int(rng.integers(n))

    def feasible_actions(state):
        return factibles[state]

    def step(state, action, rng):
        # Estado siguiente según la fila de la matriz de transición del MDP.
        fila = data["transition_matrices"][data["actions"][columna[action]]][state]
        siguiente = int(rng.choice(n, p=fila))
        costo = float(R[state, columna[action]]) - costo_demanda[state]
        return siguiente, costo

    return initial_state, feasible_actions, step


# -----------------------------------------------------------------------------
# COMPARACIÓN
# -----------------------------------------------------------------------------
def comparar(data, modelo, politica, acciones_ref, V_politica_ql, V_opt):
    compras = {str(a): float(c) for a, c in zip(data["actions"], data["compras"])}
    filas = []
    for i, (inv, cat) in enumerate(data["state_tuples"]):
        visitas = sum(v for (s, _), v in modelo.visit_counts.items() if s == i)
        aprendido = visitas > 0
        filas.append({
            "inventario_ton": round(inv, 2),
            "cat_precio": cat,
            "visitas": visitas,
            "compra_qlearning_ton": round(compras[str(politica[i])], 2) if aprendido else None,
            "compra_mdp_ton": round(compras[acciones_ref[i]], 2),
            "coincide": (str(politica[i]) == acciones_ref[i]) if aprendido else None,
            "costo_politica_qlearning_USD": (V_politica_ql[i]
                                             if V_politica_ql is not None else None),
            "costo_minimo_mdp_USD": V_opt[i],
            "sobrecosto_USD": ((V_politica_ql[i] - V_opt[i])
                               if V_politica_ql is not None else None),
        })
    return pd.DataFrame(filas)


def main():
    original = cargar_mdp_independiente()
    data = original.construir_mdp(COMMODITY)
    initial_state, feasible_actions, step = preparar_qlearning(original, data)
    h = HIPERPARAMETROS

    print("=" * 78)
    print(f"Q-LEARNING (Sutton y Barto) — MDP INDEPENDIENTE DE {COMMODITY}")
    print("=" * 78)
    print(f"Estados: {len(data['states'])} | beta: {original.BETA:.6f}")
    print("\nHIPERPARÁMETROS")
    for clave, valor in h.items():
        print(f"  {clave:<10}{valor}")


    print("\nResolviendo la referencia con dtmdp + value iteration...")
    _, acciones_ref = resolver_referencia(original, data)

    print("Entrenando Q-learning...")
    modelo = qlearning(initial_state, feasible_actions, step, original.BETA)
    inicio = time.perf_counter()
    _, politica = modelo.solve(n_steps=h["n_pasos"], alpha=h["alpha"],
                                     epsilon=h["epsilon"], minimize=True,
                                     seed=h["semilla"])
    tiempo = time.perf_counter() - inicio

    # Costo exacto de la política de Q-learning (necesita una acción por estado).
    n = len(data["states"])
    faltantes = [i for i in range(n) if i not in politica]
    if faltantes:
        V_politica_ql = None
    else:
        V_politica_ql = costo_politica(original, data, [str(politica[i]) for i in range(n)])
    # Costo mínimo: evaluación EXACTA de la política óptima del MDP, con el
    # mismo método que se usa para Q-learning. (Los valores que devuelve
    # value iteration se detienen por tolerancia y pueden quedar algo lejos.)
    V_opt = costo_politica(original, data, acciones_ref)

    tabla = comparar(data, modelo, politica, acciones_ref, V_politica_ql, V_opt)
    aprendidos = tabla[tabla["visitas"] > 0]

    print(f"\nTiempo de entrenamiento: {tiempo:.1f} s")
    print("\nPOLÍTICA: Q-LEARNING vs MDP")
    print(tabla[["inventario_ton", "cat_precio", "visitas", "compra_qlearning_ton",
                 "compra_mdp_ton", "coincide"]].to_string(index=False))
    print("\nRESUMEN")
    print(f"  Estados visitados: {len(aprendidos)} de {len(tabla)}")
    if len(aprendidos):
        print(f"  Coincidencia de acciones (estados visitados): "
              f"{int(aprendidos['coincide'].sum())} de {len(aprendidos)} "
              f"({100 * aprendidos['coincide'].mean():.1f} %)")

    # ¿Obtenemos el costo mínimo?
    print("\nCOSTO DE LA POLÍTICA vs COSTO MÍNIMO DEL MDP")
    print("  (costo descontado esperado, promedio sobre los estados, USD)")
    print(f"  {'Política':<36}{'Costo':>18}{'Sobrecosto':>16}{'%':>10}")
    print(f"  {'MDP (costo mínimo)':<36}{V_opt.mean():>18,.0f}{0:>16,.0f}{0:>10.4f}")
    if V_politica_ql is None:
        print(f"  Q-learning: no se puede evaluar; estados sin acción: {faltantes}")
    candidatas = {} if V_politica_ql is None else {"Q-learning": V_politica_ql}
    for nombre, acciones in reglas_simples(original, data, feasible_actions).items():
        candidatas[nombre] = costo_politica(original, data, acciones)
    for nombre, V in candidatas.items():
        brecha = V - V_opt
        print(f"  {nombre:<36}{V.mean():>18,.0f}{brecha.mean():>16,.0f}"
              f"{100 * brecha.mean() / V_opt.mean():>10.4f}")
    if V_politica_ql is not None:
        miope = candidatas["Regla miope (lo más barato hoy)"] - V_opt
        ql = V_politica_ql - V_opt
        print(f"\n  Ventaja del MDP sobre la regla miope: USD {miope.mean():,.0f}")
        print(f"  Parte de esa ventaja que captura Q-learning: "
              f"{100 * (1 - ql.mean() / miope.mean()):.1f} %"
              "  (100 % = óptimo; 0 % = igual a la miope; negativo = peor)")

    SALIDA.mkdir(parents=True, exist_ok=True)
    archivo = SALIDA / f"qlearning_{COMMODITY}_politica.csv"
    tabla.to_csv(archivo, index=False, encoding="utf-8-sig")
    print(f"\nResultados guardados en: {archivo}")


if __name__ == "__main__":
    main()

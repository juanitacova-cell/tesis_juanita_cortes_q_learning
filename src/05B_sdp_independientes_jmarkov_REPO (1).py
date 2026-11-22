"""
05B_sdp_independientes_jmarkov.py
---------------------------------
Benchmark computacional: resuelve N, P y K POR SEPARADO con
`jmarkov.sdp.dtsdp` para validar la lógica del SDP de 5 épocas.

Estas tres instancias NO reemplazan la formulación conjunta. Se usan para:
  1) comprobar estados, acciones, costos y transiciones en jMarkov;
  2) obtener políticas interpretables con un tamaño pequeño;
  3) servir como benchmark para una futura implementación conjunta/escalable.

IMPORTANTE SOBRE jMarkov
========================
La versión inspeccionada de dtsdp.solve() presenta una inicialización que hace
problemático usar costos brutos con `minimize=True`, y además usa best_init=-100000
en maximización. Para evitar ambos problemas sin modificar la librería:

  - se transforma costo -> retorno = - costo;
  - se escala el costo por COST_SCALE=1e6;
  - se llama solve(minimize=False).

La política óptima no cambia por multiplicar todos los costos por una constante
positiva ni por maximizar su negativo.

El factor de descuento se fija muy cercano a 1 porque la documentación de
jMarkov exige beta < 1. Para un horizonte de solo 5 meses, BETA=0.999999 es una
aproximación prácticamente no descontada. Conviene validar este punto con el
asesor si se desea beta=1 exactamente.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from jmarkov.sdp.dtsdp import dtsdp

BASE_DIR = Path(__file__).resolve().parents[1]
TAB_DIR = BASE_DIR / "outputs" / "tables"
TAB_DIR.mkdir(parents=True, exist_ok=True)

T = 5
PERIODOS = np.arange(1, T + 1)
CONFIGURACION = "5_categorias"
METODO = "equal_width"
CAPACIDAD_TOTAL = 66_000.0

# Resolución propuesta para el benchmark: pasos de 5% de la demanda.
# Con la capacidad actual (~1.11 meses de demanda) genera 3 niveles de inventario:
# 0%, 5% y 10% de la demanda, y hasta 5 valores globales de compra.
RHO = 0.05

# El repositorio oficial de dtsdp acepta 0 <= beta <= 1.
# Para un horizonte finito de 5 meses sin descuento economico explicito:
BETA = 1.0

# Escalamiento puramente numérico; NO altera la política óptima.
COST_SCALE = 1_000_000.0
RETORNO_INFACTIBLE = -1_000_000.0

VENTAS_ICA_KG_ANIO = {
    "N": 456_865_757,
    "P": 56_474_803,
    "K": 200_075_415,
}
DEMANDA = {c: kg / 1000.0 / 12.0 for c, kg in VENTAS_ICA_KG_ANIO.items()}
DEMANDA_TOTAL = sum(DEMANDA.values())
CAPACIDAD = {c: CAPACIDAD_TOTAL * DEMANDA[c] / DEMANDA_TOTAL for c in DEMANDA}
DELTA = {c: RHO * DEMANDA[c] for c in DEMANDA}

# Pendiente de definir con el asesor/fuente. Se mantiene en cero en el benchmark.
PRECIO_VENTA = {"N": 0.0, "P": 0.0, "K": 0.0}

COMMODITY_ARCHIVO = {
    "N": "nitrogeno",
    "P": "fosfat",
    "K": "potash",
}


def detectar_separador(path: Path) -> str:
    """Detecta si un CSV usa coma o punto y coma."""
    primera_linea = path.read_text(encoding="utf-8-sig").splitlines()[0]
    return ";" if primera_linea.count(";") > primera_linea.count(",") else ","


def leer_csv_auto(path: Path, **kwargs) -> pd.DataFrame:
    """Lee un CSV detectando automaticamente ',' o ';'."""
    return pd.read_csv(
        path,
        sep=detectar_separador(path),
        encoding="utf-8-sig",
        **kwargs,
    )


def cargar_precios_representativos(c):
    df = leer_csv_auto(TAB_DIR / "discretizacion_resumen.csv")
    df = df[
        (df["configuracion"] == CONFIGURACION)
        & (df["metodo"] == METODO)
        & (df["commodity"] == COMMODITY_ARCHIVO[c])
    ].sort_values("categoria")
    if len(df) != 5:
        raise ValueError(f"{c}: se esperaban 5 categorías y se encontraron {len(df)}")
    return pd.Series(
        df["precio_punto_medio"].to_numpy(float),
        index=df["categoria"].astype(int).to_numpy(),
    )


def cargar_matriz_transicion(c):
    path = TAB_DIR / f"transiciones_matriz_{COMMODITY_ARCHIVO[c]}_{CONFIGURACION}.csv"
    P = leer_csv_auto(path, index_col=0)
    P.index = P.index.astype(str).str.strip().astype(int)
    P.columns = P.columns.astype(str).str.strip().astype(int)
    return P.astype(float)


def preparar_cadena_operativa(P, c, tol=1e-12):
    row_sums = P.sum(axis=1, skipna=True)
    validas = row_sums[row_sums > tol].index.tolist()
    invalidas = row_sums[row_sums <= tol].index.tolist()

    if invalidas:
        incoming = P.loc[validas, invalidas].fillna(0).to_numpy(float)
        if np.any(incoming > tol):
            raise ValueError(
                f"{c}: categorías sin transición saliente pero alcanzables: {invalidas}. "
                "No se imputan probabilidades automáticamente."
            )
        print(f"ADVERTENCIA {c}: se excluyen categorías inalcanzables sin salida {invalidas}.")

    P_op = P.loc[validas, validas].fillna(0).to_numpy(float)
    P_op = P_op / P_op.sum(axis=1, keepdims=True)
    return validas, P_op, invalidas


def niveles_inventario(c):
    max_carry = CAPACIDAD[c] - DEMANDA[c]
    n = int(np.floor(max_carry / DELTA[c] + 1e-12))
    return np.arange(n + 1, dtype=float) * DELTA[c]


def acciones_globales(c, niveles):
    vals = {
        round(DEMANDA[c] + inv_sig - inv_act, 10)
        for inv_act in niveles
        for inv_sig in niveles
        if DEMANDA[c] + inv_sig - inv_act >= -1e-9
    }
    return np.array(sorted(vals), dtype=float)


def es_factible(c, inv_actual, compra, niveles, tol=1e-6):
    pre_demanda = inv_actual + compra
    inv_sig = pre_demanda - DEMANDA[c]
    coincide_malla = np.any(np.isclose(inv_sig, niveles, atol=tol, rtol=0))
    return (
        compra >= -tol
        and pre_demanda + tol >= DEMANDA[c]
        and pre_demanda <= CAPACIDAD[c] + tol
        and coincide_malla
    )


def construir_instancia(c):
    precios = cargar_precios_representativos(c)
    cats, Pprecio, invalidas = preparar_cadena_operativa(cargar_matriz_transicion(c), c)
    precios = precios.loc[cats]

    niveles = niveles_inventario(c)
    compras = acciones_globales(c, niveles)

    # Estados: (nivel inventario, categoría de precio)
    state_tuples = [(float(inv), int(cat)) for inv in niveles for cat in cats]
    states = np.array([f"I={inv:.6f}|p={cat}" for inv, cat in state_tuples], dtype=object)

    # Acciones globales requeridas por jMarkov.
    # dtsdp.solve() crea Mat_Dec_optimo con dtype=str, que en NumPy queda como <U1.
    # Por eso usamos etiquetas de UNA SOLA LETRA para evitar truncamiento.
    letras = np.array(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"), dtype=object)
    if len(compras) > len(letras):
        raise ValueError("Hay más acciones que etiquetas de una letra disponibles.")
    action_labels = letras[:len(compras)].copy()

    state_index = {s: idx for idx, s in enumerate(state_tuples)}
    cat_index = {cat: idx for idx, cat in enumerate(cats)}

    nS = len(states)
    nA = len(action_labels)

    # R[t,s,a] = retorno escalado. Retorno = - costo.
    R = np.full((T, nS, nA), RETORNO_INFACTIBLE, dtype=float)

    # Matriz por acción. Como las probabilidades son homogéneas, construimos una
    # base y la reutilizamos en cada paso de tiempo.
    M_base = {}

    for a_idx, (label, compra) in enumerate(zip(action_labels, compras)):
        M = np.zeros((nS, nS), dtype=float)

        for s_idx, (inv_actual, cat_actual) in enumerate(state_tuples):
            factible = es_factible(c, inv_actual, compra, niveles)

            if not factible:
                # jMarkov exige que todas las filas sean estocásticas aun para
                # acciones que no son factibles en un estado. Self-loop + gran
                # penalización garantiza que no sean elegidas.
                M[s_idx, s_idx] = 1.0
                continue

            inv_sig = inv_actual + compra - DEMANDA[c]
            inv_sig = float(niveles[np.argmin(np.abs(niveles - inv_sig))])

            p_idx = cat_index[cat_actual]
            for next_p_idx, cat_sig in enumerate(cats):
                prob = Pprecio[p_idx, next_p_idx]
                next_state = (inv_sig, int(cat_sig))
                j = state_index[next_state]
                M[s_idx, j] += prob

            # Costo de compra - ingreso por venta (ingreso aún = 0).
            precio_actual = float(precios.loc[cat_actual])
            costo = precio_actual * compra - PRECIO_VENTA[c] * DEMANDA[c]
            retorno = -costo / COST_SCALE
            R[:, s_idx, a_idx] = retorno

        # Seguridad numérica: normalización de redondeo.
        row_sums = M.sum(axis=1, keepdims=True)
        M = M / row_sums
        M_base[label] = M

    # La implementación inspeccionada accede M[t+1][a] durante backward iteration.
    # Se incluyen claves 1..T para ser conservadores; el solve usa 1..T-1.
    transition_matrices = {
        k: {label: M_base[label].copy() for label in action_labels}
        for k in range(1, T)
    }

    return {
        "commodity": c,
        "states": states,
        "state_tuples": state_tuples,
        "actions": action_labels,
        "compras": compras,
        "transition_matrices": transition_matrices,
        "returns": R,
        "niveles": niveles,
        "cats": cats,
        "invalidas": invalidas,
    }


def resolver_instancia(c):
    data = construir_instancia(c)

    modelo = dtsdp(
        PERIODOS,
        data["states"],
        data["actions"],
        data["transition_matrices"],
        data["returns"],
        BETA,
    )

    # Se maximiza retorno = - costo. Evitamos minimize=True por la implementación
    # observada de la versión instalada.
    Ft_retorno, politica = modelo.solve(minimize=False)

    # Volvemos a unidades económicas aproximadas de costo (USD) para reportar.
    # Como beta≈1 y retorno=-costo/COST_SCALE:
    Ft_costo_aprox = -Ft_retorno * COST_SCALE

    filas = []
    action_to_ton = {label: a for label, a in zip(data["actions"], data["compras"])}

    for s_idx, (inv, cat) in enumerate(data["state_tuples"]):
        for t_idx, epoca in enumerate(PERIODOS):
            accion = politica[s_idx, t_idx]
            compra = float(action_to_ton[accion])
            inv_sig = inv + compra - DEMANDA[c]
            filas.append({
                "commodity": c,
                "epoca": int(epoca),
                "inventario_ton": inv,
                "cat_precio": int(cat),
                "accion": accion,
                "compra_ton": compra,
                "inventario_siguiente_ton": inv_sig,
                "valor_costo_aprox_desde_estado_USD": Ft_costo_aprox[s_idx, t_idx],
            })

    df_pol = pd.DataFrame(filas)
    out_csv = TAB_DIR / f"sdp_jmarkov_{c}_politica_5_epocas.csv"
    df_pol.to_csv(out_csv, index=False)

    resumen = pd.DataFrame([{
        "commodity": c,
        "metodo_jmarkov": "dtsdp_backward_iteration",
        "horizonte": "5_epocas",
        "n_estados": len(data["states"]),
        "n_acciones_globales": len(data["actions"]),
        "niveles_inventario": ";".join(f"{x:.4f}" for x in data["niveles"]),
        "categorias_precio_operativas": ";".join(map(str, data["cats"])),
        "categorias_excluidas": ";".join(map(str, data["invalidas"])),
        "rho": RHO,
        "beta": BETA,
        "cost_scale": COST_SCALE,
        "nota_minimizacion": "retorno=-costo; solve(minimize=False)",
    }])
    resumen.to_csv(TAB_DIR / f"sdp_jmarkov_{c}_resumen.csv", index=False)

    return df_pol, data


def main():
    print("=" * 90)
    print("MÓDULO B — TRES SDP INDEPENDIENTES CON jMarkov dtsdp")
    print("=" * 90)
    print(f"Horizonte: {T} épocas | RHO={RHO:.0%} | beta={BETA}")
    print("Se maximiza retorno=-costo escalado; la política equivale a minimizar costo.\n")

    resultados = []
    for c in ["N", "P", "K"]:
        df, data = resolver_instancia(c)
        resultados.append(df)
        print(
            f"{c}: {len(data['states'])} estados, {len(data['actions'])} acciones globales, "
            f"niveles inv={[round(x,2) for x in data['niveles']]}"
        )
        print(f"   salida: outputs/tables/sdp_jmarkov_{c}_politica_5_epocas.csv")

    combinado = pd.concat(resultados, ignore_index=True)
    combinado.to_csv(TAB_DIR / "sdp_jmarkov_independientes_politicas.csv", index=False)
    print("\nArchivo combinado: outputs/tables/sdp_jmarkov_independientes_politicas.csv")


if __name__ == "__main__":
    main()

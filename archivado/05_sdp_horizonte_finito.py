"""
05_sdp_horizonte_finito.py
--------------------------
Primera implementación computacional del problema de compra de commodities
como un SDP (programación dinámica estocástica) de horizonte finito de 5 meses.

La solución se obtiene por inducción hacia atrás (ecuación de Bellman) usando
NumPy. No se usa dtmdp de jMarkov en esta etapa porque dtmdp corresponde al
MDP homogéneo de horizonte infinito visto en la complementaria de Python.

MODELO BASE
===========
Commodities:
    N = urea
    P = roca fosfórica
    K = cloruro de potasio (potash)

Estado:
    (x, y, z, pN, pP, pK)
    x,y,z  = inventario disponible al inicio del mes [ton]
    pN,pP,pK = categoría de precio actual

Precios:
    10 intervalos de discretización equal-width por commodity.
    El precio representativo de cada categoría es su punto medio.

Demanda:
    Se usa como proxy la venta nacional 2024 de fertilizantes SIMPLE N/P/K del ICA,
    dividida entre 12 para obtener una demanda mensual constante.

Capacidad:
    Capacidad total de referencia = 66,000 ton a granel.
    Se distribuye entre N/P/K proporcionalmente a la demanda mensual.

Inventario y acciones:
    Delta_c = 10% de la demanda mensual del commodity c.
    El inventario se discretiza en múltiplos de Delta_c que caben después de
    satisfacer la demanda. Con los parámetros actuales esto da dos niveles:
    0 y Delta_c.

    Elegir el inventario final equivale a elegir la compra:
        a_c = d_c + inventario_final_c - inventario_inicial_c

    La acción continúa expresada en toneladas.

No se permiten faltantes.
Costo de almacenamiento h = 0.
No hay valor terminal del inventario en el modelo base.

Precio de venta:
    El código deja preparado el parámetro PRECIO_VENTA. Por ahora se fija en 0
    porque aún no se ha definido una fuente. Si los precios de venta y las
    demandas son exógenos y constantes, restar v_c*d_c cambia el valor de la
    función objetivo, pero no la política óptima.

IMPORTANTE SOBRE CATEGORÍAS SIN TRANSICIONES
===========================================
La discretización equal-width de 10 categorías de fósforo tiene una categoría
sin observaciones (actualmente la 5), por lo que su fila de transición MLE es
indefinida. El script conserva la discretización de 10 intervalos, pero excluye
como estados operativos las categorías sin transición saliente SI además son
inalcanzables desde las categorías válidas. Si una categoría sin transición
pudiera recibir probabilidad positiva, el script se detiene y exige definir un
tratamiento explícito en vez de inventar probabilidades.
"""

from pathlib import Path
import itertools
import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Rutas y parámetros generales
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
TAB_DIR = BASE_DIR / "outputs" / "tables"
TAB_DIR.mkdir(parents=True, exist_ok=True)

T = 5
CONFIGURACION = "10_categorias"
METODO = "equal_width"
RHO = 0.10
CAPACIDAD_TOTAL = 66_000.0  # toneladas de almacenamiento a granel

# Ventas nacionales 2024 ICA [kg/año] -> ton/mes
VENTAS_ICA_KG_ANIO = {
    "N": 456_865_757,
    "P": 56_474_803,
    "K": 200_075_415,
}
DEMANDA = {c: kg / 1000.0 / 12.0 for c, kg in VENTAS_ICA_KG_ANIO.items()}
DEMANDA_TOTAL = sum(DEMANDA.values())
CAPACIDAD = {c: CAPACIDAD_TOTAL * DEMANDA[c] / DEMANDA_TOTAL for c in DEMANDA}
DELTA = {c: RHO * DEMANDA[c] for c in DEMANDA}

# Parámetros pendientes de fuente. Mantener en 0 para el modelo base.
PRECIO_VENTA = {"N": 0.0, "P": 0.0, "K": 0.0}  # USD/ton

COMMODITY_ARCHIVO = {
    "N": "nitrogeno",
    "P": "fosfat",
    "K": "potash",
}


# -----------------------------------------------------------------------------
# Lectura de datos del proyecto
# -----------------------------------------------------------------------------
def cargar_precios_representativos():
    """Carga el punto medio de cada una de las 10 categorías de precio."""
    path = TAB_DIR / "discretizacion_resumen.csv"
    df = pd.read_csv(path)
    df = df[(df["configuracion"] == CONFIGURACION) & (df["metodo"] == METODO)]

    precios = {}
    for c, nombre in COMMODITY_ARCHIVO.items():
        sub = df[df["commodity"] == nombre].sort_values("categoria")
        if len(sub) != 10:
            raise ValueError(f"Se esperaban 10 categorías para {nombre}, se encontraron {len(sub)}")
        precios[c] = pd.Series(
            sub["precio_punto_medio"].to_numpy(float),
            index=sub["categoria"].astype(int).to_numpy(),
            name=c,
        )
    return precios


def cargar_matriz_transicion(commodity):
    """Carga la matriz MLE 10x10 de un commodity."""
    nombre = COMMODITY_ARCHIVO[commodity]
    path = TAB_DIR / f"transiciones_matriz_{nombre}_{CONFIGURACION}.csv"
    P = pd.read_csv(path, index_col=0)
    P.index = P.index.astype(int)
    P.columns = P.columns.astype(int)
    return P.astype(float)


def preparar_cadena_operativa(P: pd.DataFrame, commodity: str, tol=1e-12):
    """
    Elimina únicamente categorías sin transición saliente que además sean
    inalcanzables desde las filas válidas. No imputa probabilidades.
    """
    row_sums = P.sum(axis=1, skipna=True)
    validas = row_sums[row_sums > tol].index.tolist()
    invalidas = row_sums[row_sums <= tol].index.tolist()

    if invalidas:
        incoming = P.loc[validas, invalidas].fillna(0).to_numpy(float)
        if np.any(incoming > tol):
            raise ValueError(
                f"{commodity}: hay categorías sin transición saliente que sí son alcanzables: "
                f"{invalidas}. Se requiere definir una regla de estimación/suavizamiento."
            )
        print(
            f"ADVERTENCIA {commodity}: categorías sin transición saliente {invalidas}. "
            "Son inalcanzables desde las categorías válidas y se excluyen del espacio operativo."
        )

    P_op = P.loc[validas, validas].fillna(0).to_numpy(float)
    # Normalizar para corregir redondeos del CSV (p.ej. 0.9999 o 1.0001)
    sums = P_op.sum(axis=1, keepdims=True)
    P_op = P_op / sums
    return validas, P_op, invalidas


# -----------------------------------------------------------------------------
# Inventarios y acciones
# -----------------------------------------------------------------------------
def niveles_inventario(commodity):
    """Niveles de inventario al inicio/fin del mes compatibles con Delta."""
    max_carry = CAPACIDAD[commodity] - DEMANDA[commodity]
    delta = DELTA[commodity]
    n = int(np.floor(max_carry / delta + 1e-12))
    niveles = np.arange(n + 1, dtype=float) * delta
    return niveles


def accion_para_transicion_inv(commodity, inv_actual, inv_siguiente):
    """Compra necesaria [ton] para terminar el mes con inv_siguiente."""
    return DEMANDA[commodity] + inv_siguiente - inv_actual


def es_factible(commodity, inv_actual, compra, tol=1e-8):
    pre_demanda = inv_actual + compra
    return (
        compra >= -tol
        and pre_demanda + tol >= DEMANDA[commodity]
        and pre_demanda <= CAPACIDAD[commodity] + tol
    )


# -----------------------------------------------------------------------------
# Bellman
# -----------------------------------------------------------------------------
def expectativa_precios(V_dest, PN, PP, PK):
    """
    V_dest[pN',pP',pK'] -> E[V_dest | pN,pP,pK]
    bajo independencia condicional de las tres cadenas de precios.
    """
    return np.einsum("ia,jb,kc,abc->ijk", PN, PP, PK, V_dest, optimize=True)


def resolver_sdp():
    precios = cargar_precios_representativos()

    matrices_raw = {c: cargar_matriz_transicion(c) for c in ["N", "P", "K"]}
    cadenas = {}
    invalidas = {}
    for c in ["N", "P", "K"]:
        cats, P_op, inv = preparar_cadena_operativa(matrices_raw[c], c)
        cadenas[c] = (cats, P_op)
        invalidas[c] = inv

    catsN, PN = cadenas["N"]
    catsP, PP = cadenas["P"]
    catsK, PK = cadenas["K"]

    pN = precios["N"].loc[catsN].to_numpy(float)
    pP = precios["P"].loc[catsP].to_numpy(float)
    pK = precios["K"].loc[catsK].to_numpy(float)

    # Mallas de precio actual para calcular costos inmediatos de forma vectorizada.
    gridN = pN[:, None, None]
    gridP = pP[None, :, None]
    gridK = pK[None, None, :]

    niveles = {c: niveles_inventario(c) for c in ["N", "P", "K"]}
    nI = {c: len(niveles[c]) for c in niveles}
    inv_combos = list(itertools.product(range(nI["N"]), range(nI["P"]), range(nI["K"])))
    next_combos = inv_combos.copy()

    # Forma: inventario N, P, K, precio N, P, K
    shape = (nI["N"], nI["P"], nI["K"], len(catsN), len(catsP), len(catsK))
    V_next = np.zeros(shape, dtype=float)  # V_{T+1} = 0

    values = [None] * (T + 1)
    policies = [None] * (T + 1)

    ingreso_constante = sum(PRECIO_VENTA[c] * DEMANDA[c] for c in ["N", "P", "K"])

    for t in range(T, 0, -1):
        # Para cada inventario final posible, calcular una sola vez la esperanza
        # del valor futuro para todas las combinaciones de precios actuales.
        EV_por_next = {}
        for nxt in next_combos:
            V_dest = V_next[nxt[0], nxt[1], nxt[2], :, :, :]
            EV_por_next[nxt] = expectativa_precios(V_dest, PN, PP, PK)

        V_t = np.full(shape, np.inf, dtype=float)
        # política: índice de inventario final elegido para N/P/K
        pol_t = np.full(shape + (3,), -1, dtype=np.int8)

        for cur in inv_combos:
            x = niveles["N"][cur[0]]
            y = niveles["P"][cur[1]]
            z = niveles["K"][cur[2]]

            best = np.full((len(catsN), len(catsP), len(catsK)), np.inf, dtype=float)
            best_next = np.full(best.shape + (3,), -1, dtype=np.int8)

            for nxt in next_combos:
                xp = niveles["N"][nxt[0]]
                yp = niveles["P"][nxt[1]]
                zp = niveles["K"][nxt[2]]

                aN = accion_para_transicion_inv("N", x, xp)
                aP = accion_para_transicion_inv("P", y, yp)
                aK = accion_para_transicion_inv("K", z, zp)

                if not (
                    es_factible("N", x, aN)
                    and es_factible("P", y, aP)
                    and es_factible("K", z, aK)
                ):
                    continue

                costo = gridN * aN + gridP * aP + gridK * aK - ingreso_constante
                q = costo + EV_por_next[nxt]

                mejora = q < best
                best[mejora] = q[mejora]
                for dim, idx in enumerate(nxt):
                    arr = best_next[..., dim]
                    arr[mejora] = idx

            if np.isinf(best).any():
                raise RuntimeError(f"Hay estados sin acción factible para inventario {cur}")

            V_t[cur[0], cur[1], cur[2], :, :, :] = best
            pol_t[cur[0], cur[1], cur[2], :, :, :, :] = best_next

        values[t] = V_t
        policies[t] = pol_t
        V_next = V_t
        print(f"Época {t} resuelta.")

    meta = {
        "precios": precios,
        "cats": {"N": catsN, "P": catsP, "K": catsK},
        "matrices": {"N": PN, "P": PP, "K": PK},
        "niveles": niveles,
        "invalidas": invalidas,
    }
    return values, policies, meta


# -----------------------------------------------------------------------------
# Exportación de resultados
# -----------------------------------------------------------------------------
def exportar_politica(values, policies, meta):
    cats = meta["cats"]
    niveles = meta["niveles"]
    filas = []

    for t in range(1, T + 1):
        V = values[t]
        pol = policies[t]
        for iN, x in enumerate(niveles["N"]):
            for iP, y in enumerate(niveles["P"]):
                for iK, z in enumerate(niveles["K"]):
                    for jN, catN in enumerate(cats["N"]):
                        for jP, catP in enumerate(cats["P"]):
                            for jK, catK in enumerate(cats["K"]):
                                nxt_idx = pol[iN, iP, iK, jN, jP, jK]
                                xp = niveles["N"][nxt_idx[0]]
                                yp = niveles["P"][nxt_idx[1]]
                                zp = niveles["K"][nxt_idx[2]]
                                aN = accion_para_transicion_inv("N", x, xp)
                                aP = accion_para_transicion_inv("P", y, yp)
                                aK = accion_para_transicion_inv("K", z, zp)
                                filas.append({
                                    "epoca": t,
                                    "inv_N_ton": x,
                                    "inv_P_ton": y,
                                    "inv_K_ton": z,
                                    "cat_precio_N": catN,
                                    "cat_precio_P": catP,
                                    "cat_precio_K": catK,
                                    "compra_N_ton": aN,
                                    "compra_P_ton": aP,
                                    "compra_K_ton": aK,
                                    "inv_N_siguiente_ton": xp,
                                    "inv_P_siguiente_ton": yp,
                                    "inv_K_siguiente_ton": zp,
                                    "valor_optimo_desde_estado": V[iN, iP, iK, jN, jP, jK],
                                })

    df = pd.DataFrame(filas)
    out = TAB_DIR / "sdp_politica_optima_5_epocas.csv"
    df.to_csv(out, index=False)
    return df


def exportar_resumen(meta, politica_df):
    niveles = meta["niveles"]
    cats = meta["cats"]

    filas = []
    for c in ["N", "P", "K"]:
        filas.append({
            "commodity": c,
            "demanda_ton_mes": DEMANDA[c],
            "capacidad_ton": CAPACIDAD[c],
            "delta_ton": DELTA[c],
            "niveles_inventario": "; ".join(f"{v:.4f}" for v in niveles[c]),
            "categorias_precio_operativas": "; ".join(map(str, cats[c])),
            "categorias_sin_transicion": "; ".join(map(str, meta["invalidas"][c])) if meta["invalidas"][c] else "",
        })

    resumen = pd.DataFrame(filas)
    resumen["horizonte_meses"] = T
    resumen["rho_delta"] = RHO
    resumen["capacidad_total_ton"] = CAPACIDAD_TOTAL
    resumen["n_estados_operativos"] = (
        len(niveles["N"]) * len(niveles["P"]) * len(niveles["K"])
        * len(cats["N"]) * len(cats["P"]) * len(cats["K"])
    )
    resumen["n_filas_politica"] = len(politica_df)
    resumen.to_csv(TAB_DIR / "sdp_resumen_modelo.csv", index=False)


def exportar_ejemplo(meta, politica_df):
    """Ejemplo usando las categorías del último mes disponible: N=4, P=10, K=2."""
    ejemplo = politica_df[
        (politica_df["epoca"] == 1)
        & np.isclose(politica_df["inv_N_ton"], 0)
        & np.isclose(politica_df["inv_P_ton"], 0)
        & np.isclose(politica_df["inv_K_ton"], 0)
        & (politica_df["cat_precio_N"] == 4)
        & (politica_df["cat_precio_P"] == 10)
        & (politica_df["cat_precio_K"] == 2)
    ].copy()
    ejemplo.to_csv(TAB_DIR / "sdp_ejemplo_estado_inicial.csv", index=False)
    return ejemplo


def main():
    print("=" * 95)
    print("SDP DE HORIZONTE FINITO - COMPRA DE COMMODITIES")
    print("=" * 95)
    for c in ["N", "P", "K"]:
        print(
            f"{c}: demanda={DEMANDA[c]:,.2f} t/mes | capacidad={CAPACIDAD[c]:,.2f} t | "
            f"Delta={DELTA[c]:,.2f} t | niveles_inv={niveles_inventario(c)}"
        )

    values, policies, meta = resolver_sdp()
    politica_df = exportar_politica(values, policies, meta)
    exportar_resumen(meta, politica_df)
    ejemplo = exportar_ejemplo(meta, politica_df)

    n_estados = (
        len(meta["niveles"]["N"]) * len(meta["niveles"]["P"]) * len(meta["niveles"]["K"])
        * len(meta["cats"]["N"]) * len(meta["cats"]["P"]) * len(meta["cats"]["K"])
    )
    print("\n" + "=" * 95)
    print(f"Estados operativos por época: {n_estados:,}")
    print(f"Filas exportadas en la política (5 épocas): {len(politica_df):,}")
    print(f"Resultados guardados en: {TAB_DIR}")
    print("=" * 95)
    if not ejemplo.empty:
        cols = [
            "epoca", "cat_precio_N", "cat_precio_P", "cat_precio_K",
            "compra_N_ton", "compra_P_ton", "compra_K_ton",
            "inv_N_siguiente_ton", "inv_P_siguiente_ton", "inv_K_siguiente_ton",
            "valor_optimo_desde_estado",
        ]
        print("\nEjemplo: época 1, inventario inicial 0, categorías N=4, P=10, K=2")
        print(ejemplo[cols].to_string(index=False))


if __name__ == "__main__":
    main()

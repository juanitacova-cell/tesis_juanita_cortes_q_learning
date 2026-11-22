"""
05A_sdp_conjunto_jmarkov_EJECUTABLE.py
--------------------------------------
Modelo conjunto N-P-K con 5 categorías preparado para que su solución sea `jmarkov.sdp.dtsdp`.


IMPORTANTE
----------
- `dtsdp` usa matrices densas: este modelo sigue siendo computacionalmente grande.
- Para reducir memoria, las matrices se almacenan como float32 y se REUTILIZAN
  entre épocas (la dinámica es homogénea), en lugar de copiarlas cuatro veces.
- `dtsdp.solve()` crea la política con dtype=str (<U1), por eso cada acción conjunta
  se codifica con un carácter Unicode único de longitud 1.
- Se conserva el workaround de minimización usado en 05B:
      retorno = - costo / COST_SCALE
      solve(minimize=False)
  porque la implementación actual de dtsdp tiene una inicialización problemática
  cuando se usa `minimize=True` con costos grandes.
"""

from pathlib import Path
import itertools
import time
import traceback
import numpy as np
import pandas as pd
from jmarkov.sdp.dtsdp import dtsdp

# -----------------------------------------------------------------------------
# PARÁMETROS
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
TAB_DIR = BASE_DIR / "outputs" / "tables"
TAB_DIR.mkdir(parents=True, exist_ok=True)

T = 5
PERIODOS = np.arange(1, T + 1)
CONFIGURACION = "5_categorias"
METODO = "equal_width"
CAPACIDAD_TOTAL = 66_000.0
RHO = 0.05
BETA = 1.0
COST_SCALE = 1_000_000.0
RETORNO_INFACTIBLE = -1_000_000.0

# float32 reduce aproximadamente a la mitad la RAM de las matrices densas.
DTYPE_TRANSICIONES = np.float32

# Si False, hace un chequeo de RAM y detiene el intento con MemoryError controlado
# si la memoria parece insuficiente. Para forzar el intento real, usar True.
FORZAR_EJECUCION = False
FACTOR_SEGURIDAD_MEMORIA = 1.30

VENTAS_ICA_KG_ANIO = {
    "N": 456_865_757,
    "P": 56_474_803,
    "K": 200_075_415,
}
DEMANDA = {c: kg / 1000.0 / 12.0 for c, kg in VENTAS_ICA_KG_ANIO.items()}
DEMANDA_TOTAL = sum(DEMANDA.values())
CAPACIDAD = {c: CAPACIDAD_TOTAL * DEMANDA[c] / DEMANDA_TOTAL for c in DEMANDA}
DELTA = {c: RHO * DEMANDA[c] for c in DEMANDA}
PRECIO_VENTA = {"N": 0.0, "P": 0.0, "K": 0.0}

COMMODITY_ARCHIVO = {
    "N": "nitrogeno",
    "P": "fosfat",
    "K": "potash",
}

# -----------------------------------------------------------------------------
# LECTURA DE ARCHIVOS
# -----------------------------------------------------------------------------
def detectar_separador(path: Path) -> str:
    primera_linea = path.read_text(encoding="utf-8-sig").splitlines()[0]
    return ";" if primera_linea.count(";") > primera_linea.count(",") else ","


def leer_csv_auto(path: Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(
        path,
        sep=detectar_separador(path),
        encoding="utf-8-sig",
        **kwargs,
    )


def cargar_precios_representativos(c: str) -> pd.Series:
    path = TAB_DIR / "discretizacion_resumen.csv"
    df = leer_csv_auto(path)
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


def cargar_matriz_transicion(c: str) -> pd.DataFrame:
    path = TAB_DIR / f"transiciones_matriz_{COMMODITY_ARCHIVO[c]}_{CONFIGURACION}.csv"
    P = leer_csv_auto(path, index_col=0)
    P.index = P.index.astype(str).str.strip().astype(int)
    P.columns = P.columns.astype(str).str.strip().astype(int)
    return P.astype(float)


def preparar_cadena_operativa(P: pd.DataFrame, c: str, tol: float = 1e-12):
    row_sums = P.sum(axis=1, skipna=True)
    validas = row_sums[row_sums > tol].index.tolist()
    invalidas = row_sums[row_sums <= tol].index.tolist()

    if invalidas:
        incoming = P.loc[validas, invalidas].fillna(0).to_numpy(float)
        if np.any(incoming > tol):
            raise ValueError(
                f"{c}: categorías sin transición saliente pero alcanzables: {invalidas}"
            )
        print(f"ADVERTENCIA {c}: se excluyen categorías inalcanzables {invalidas}")

    P_op = P.loc[validas, validas].fillna(0).to_numpy(float)
    P_op = P_op / P_op.sum(axis=1, keepdims=True)
    return validas, P_op, invalidas

# -----------------------------------------------------------------------------
# INVENTARIOS Y ACCIONES
# -----------------------------------------------------------------------------
def niveles_inventario(c: str) -> np.ndarray:
    max_carry = CAPACIDAD[c] - DEMANDA[c]
    n = int(np.floor(max_carry / DELTA[c] + 1e-12))
    return np.arange(n + 1, dtype=float) * DELTA[c]


def acciones_globales_por_commodity(c: str, niveles: np.ndarray) -> np.ndarray:
    vals = {
        round(DEMANDA[c] + inv_sig - inv_act, 10)
        for inv_act in niveles
        for inv_sig in niveles
        if DEMANDA[c] + inv_sig - inv_act >= -1e-9
    }
    return np.array(sorted(vals), dtype=float)


def siguiente_inventario(c, inv_actual, compra, niveles, tol=1e-6):
    pre_demanda = inv_actual + compra
    inv_sig = pre_demanda - DEMANDA[c]

    if compra < -tol:
        return None
    if pre_demanda + tol < DEMANDA[c]:
        return None
    if pre_demanda > CAPACIDAD[c] + tol:
        return None

    j = int(np.argmin(np.abs(niveles - inv_sig)))
    if abs(niveles[j] - inv_sig) > tol:
        return None
    return float(niveles[j])

# -----------------------------------------------------------------------------
# MEMORIA
# -----------------------------------------------------------------------------
def ram_disponible_gib():
    try:
        import psutil
        return psutil.virtual_memory().available / 1024**3
    except Exception:
        return None


def estimar_memoria(n_states, n_actions):
    bytes_matrix = n_states * n_states * np.dtype(DTYPE_TRANSICIONES).itemsize
    bytes_matrices = bytes_matrix * n_actions
    bytes_returns = T * n_states * n_actions * 8
    total = bytes_matrices + bytes_returns
    return {
        "una_matriz_GiB": bytes_matrix / 1024**3,
        "matrices_unicas_GiB": bytes_matrices / 1024**3,
        "retornos_GiB": bytes_returns / 1024**3,
        "total_base_GiB": total / 1024**3,
        "total_con_margen_GiB": total * FACTOR_SEGURIDAD_MEMORIA / 1024**3,
    }


def verificar_memoria(n_states, n_actions):
    m = estimar_memoria(n_states, n_actions)
    disponible = ram_disponible_gib()

    print("\n--- PRE-CHEQUEO DE MEMORIA ---")
    print(f"Una matriz densa: {m['una_matriz_GiB']:.3f} GiB")
    print(f"{n_actions} matrices únicas: {m['matrices_unicas_GiB']:.2f} GiB")
    print(f"Retornos: {m['retornos_GiB']:.3f} GiB")
    print(f"Estimado con margen: {m['total_con_margen_GiB']:.2f} GiB")

    if disponible is not None:
        print(f"RAM disponible detectada: {disponible:.2f} GiB")
        if disponible < m["total_con_margen_GiB"] and not FORZAR_EJECUCION:
            raise MemoryError(
                "RAM insuficiente para materializar de forma segura el modelo denso.\n"
                f"Disponible: {disponible:.2f} GiB\n"
                f"Estimado requerido con margen: {m['total_con_margen_GiB']:.2f} GiB\n"
                "Si quieres forzar el intento, cambia FORZAR_EJECUCION = True."
            )
    return m

# -----------------------------------------------------------------------------
# METADATA CONJUNTA
# -----------------------------------------------------------------------------
def construir_metadata_conjunta():
    cadenas = {}
    precios = {}

    for c in ["N", "P", "K"]:
        cats, P, invalidas = preparar_cadena_operativa(cargar_matriz_transicion(c), c)
        cadenas[c] = {"cats": cats, "P": P, "invalidas": invalidas}
        precios[c] = cargar_precios_representativos(c).loc[cats]

    inv = {c: niveles_inventario(c) for c in ["N", "P", "K"]}
    acciones_c = {
        c: acciones_globales_por_commodity(c, inv[c])
        for c in ["N", "P", "K"]
    }

    inv_tuples = list(itertools.product(inv["N"], inv["P"], inv["K"]))
    price_tuples = list(
        itertools.product(
            cadenas["N"]["cats"],
            cadenas["P"]["cats"],
            cadenas["K"]["cats"],
        )
    )

    state_tuples = [
        (float(iN), float(iP), float(iK), int(pN), int(pP), int(pK))
        for iN, iP, iK in inv_tuples
        for pN, pP, pK in price_tuples
    ]

    states = np.array(
        [
            f"IN={iN:.6f}|IP={iP:.6f}|IK={iK:.6f}|pN={pN}|pP={pP}|pK={pK}"
            for iN, iP, iK, pN, pP, pK in state_tuples
        ],
        dtype=object,
    )

    action_tuples = list(
        itertools.product(acciones_c["N"], acciones_c["P"], acciones_c["K"])
    )

    # 125 etiquetas únicas de exactamente 1 carácter para evitar truncamiento <U1.
    action_labels = np.array(
        [chr(0x0100 + i) for i in range(len(action_tuples))],
        dtype=object,
    )

    return {
        "states": states,
        "state_tuples": state_tuples,
        "inv_tuples": inv_tuples,
        "price_tuples": price_tuples,
        "action_labels": action_labels,
        "action_tuples": action_tuples,
        "inv": inv,
        "acciones_c": acciones_c,
        "cadenas": cadenas,
        "precios": precios,
    }

# -----------------------------------------------------------------------------
# CONSTRUCCIÓN DEL MODELO DENSO
# -----------------------------------------------------------------------------
def matriz_precios_conjunta(cadenas):
    P_joint = np.kron(
        np.kron(cadenas["N"]["P"], cadenas["P"]["P"]),
        cadenas["K"]["P"],
    )
    return P_joint.astype(DTYPE_TRANSICIONES, copy=False)


def construir_modelo_denso(data):
    states = data["states"]
    inv_tuples = data["inv_tuples"]
    price_tuples = data["price_tuples"]
    action_labels = data["action_labels"]
    action_tuples = data["action_tuples"]

    nS = len(states)
    nA = len(action_labels)
    nP = len(price_tuples)

    P_joint = matriz_precios_conjunta(data["cadenas"])

    inv_index = {
        (round(float(iN), 10), round(float(iP), 10), round(float(iK), 10)): idx
        for idx, (iN, iP, iK) in enumerate(inv_tuples)
    }

    precioN = np.array([float(data["precios"]["N"].loc[p[0]]) for p in price_tuples])
    precioP = np.array([float(data["precios"]["P"].loc[p[1]]) for p in price_tuples])
    precioK = np.array([float(data["precios"]["K"].loc[p[2]]) for p in price_tuples])

    R = np.full((T, nS, nA), RETORNO_INFACTIBLE, dtype=np.float64)
    matrices_base = {}

    print(f"\nConstruyendo {nA} matrices densas de {nS}x{nS}...")
    inicio = time.perf_counter()

    for a_idx, (label, action) in enumerate(zip(action_labels, action_tuples)):
        aN, aP, aK = map(float, action)
        M = np.zeros((nS, nS), dtype=DTYPE_TRANSICIONES)

        for inv_idx, (iN, iP, iK) in enumerate(inv_tuples):
            iN_sig = siguiente_inventario("N", float(iN), aN, data["inv"]["N"])
            iP_sig = siguiente_inventario("P", float(iP), aP, data["inv"]["P"])
            iK_sig = siguiente_inventario("K", float(iK), aK, data["inv"]["K"])

            r0 = inv_idx * nP
            r1 = r0 + nP

            if iN_sig is None or iP_sig is None or iK_sig is None:
                idxs = np.arange(r0, r1)
                M[idxs, idxs] = 1.0
                continue

            next_inv_idx = inv_index[
                (round(iN_sig, 10), round(iP_sig, 10), round(iK_sig, 10))
            ]
            c0 = next_inv_idx * nP
            c1 = c0 + nP
            M[r0:r1, c0:c1] = P_joint

            costos = (
                precioN * aN
                + precioP * aP
                + precioK * aK
                - PRECIO_VENTA["N"] * DEMANDA["N"]
                - PRECIO_VENTA["P"] * DEMANDA["P"]
                - PRECIO_VENTA["K"] * DEMANDA["K"]
            )
            R[:, r0:r1, a_idx] = (-costos / COST_SCALE)[np.newaxis, :]

        row_sums = M.sum(axis=1)
        if not np.allclose(row_sums, 1.0, atol=1e-6):
            malos = np.where(~np.isclose(row_sums, 1.0, atol=1e-6))[0]
            raise ValueError(
                f"Acción {a_idx}: filas de transición no estocásticas: {malos[:10].tolist()}"
            )

        matrices_base[label] = M

        if (a_idx + 1) % 5 == 0 or a_idx == 0 or a_idx + 1 == nA:
            mins = (time.perf_counter() - inicio) / 60
            print(f"  matrices construidas: {a_idx + 1}/{nA} | {mins:.1f} min")

    # Reutilizamos las MISMAS matrices en los cuatro pasos del horizonte.
    transition_matrices = {t: matrices_base for t in range(1, T)}
    return transition_matrices, R

# -----------------------------------------------------------------------------
# EXPORTACIÓN
# -----------------------------------------------------------------------------
def exportar_resultados(data, valores, politica):
    label_to_action = {
        str(label): tuple(map(float, action))
        for label, action in zip(data["action_labels"], data["action_tuples"])
    }

    filas = []
    for s_idx, state in enumerate(data["state_tuples"]):
        iN, iP, iK, pN, pP, pK = state
        for t_idx, epoca in enumerate(PERIODOS):
            codigo = str(politica[s_idx, t_idx])
            aN, aP, aK = label_to_action[codigo]
            filas.append({
                "epoca": int(epoca),
                "I_N_ton": iN,
                "I_P_ton": iP,
                "I_K_ton": iK,
                "cat_N": pN,
                "cat_P": pP,
                "cat_K": pK,
                "accion_codigo": codigo,
                "compra_N_ton": aN,
                "compra_P_ton": aP,
                "compra_K_ton": aK,
                "valor_costo_conjunto_USD": -float(valores[s_idx, t_idx]) * COST_SCALE,
            })

    out = TAB_DIR / "sdp_jmarkov_conjunto_politica_5_epocas.csv"
    pd.DataFrame(filas).to_csv(out, index=False)
    print(f"Resultados exportados: {out}")

# -----------------------------------------------------------------------------
# EJECUCIÓN REAL
# -----------------------------------------------------------------------------
def ejecutar_modelo_conjunto():
    print("=" * 100)
    print("MODELO CONJUNTO N-P-K — INTENTO REAL CON jMarkov dtsdp")
    print("=" * 100)

    print("\n[1/5] Construyendo metadata...")
    data = construir_metadata_conjunta()
    nS = len(data["states"])
    nA = len(data["action_labels"])
    print(f"Estados conjuntos |S| = {nS:,}")
    print(f"Acciones conjuntas |A| = {nA:,}")

    print("\n[2/5] Verificando memoria...")
    verificar_memoria(nS, nA)

    print("\n[3/5] Materializando matrices y retornos...")
    transition_matrices, R = construir_modelo_denso(data)

    print("\n[4/5] Instanciando dtsdp...")
    modelo = dtsdp(
        PERIODOS,
        data["states"],
        data["action_labels"],
        transition_matrices,
        R,
        BETA,
    )
    print("Instancia dtsdp creada correctamente.")

    print("\n[5/5] Ejecutando backward iteration...")
    inicio = time.perf_counter()
    valores, politica = modelo.solve(minimize=False)
    mins = (time.perf_counter() - inicio) / 60
    print(f"SOLUCIÓN COMPLETADA en {mins:.2f} minutos")

    exportar_resultados(data, valores, politica)
    return valores, politica


def main():
    try:
        ejecutar_modelo_conjunto()
    except Exception as exc:
        print("\n" + "=" * 100)
        print("ERROR DURANTE EL MODELO CONJUNTO")
        print("=" * 100)
        print(f"{type(exc).__name__}: {exc}")
        print("\nTRACEBACK COMPLETO:")
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()

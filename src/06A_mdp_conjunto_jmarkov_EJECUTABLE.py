"""
06A_mdp_conjunto_jmarkov_EJECUTABLE.py
--------------------------------------
MODELO MDP CONJUNTO N-P-K de horizonte infinito usando DIRECTAMENTE
`jmarkov.mdp.dtmdp` del repositorio oficial de jMarkov.

Repositorio revisado:
    https://github.com/copa-uniandes/jmarkov

API oficial utilizada
=====================
    from jmarkov.mdp.dtmdp import dtmdp

    modelo = dtmdp(
        states,
        actions,
        transition_matrices,
        immediate_returns,
        discount_factor,
    )

    value_function, optimal_policy = modelo.solve(
        TOLERANCIA,
        minimize=True,
        method="value_iteration",
    )

    P_pi = modelo.policy_transition_matrix(optimal_policy)

    expected_policy_value = modelo.expected_policy_value(
        value_function,
        optimal_policy,
    )

Interpretación
==============
Estado conjunto:
    s = (I_N, I_P, I_K, p_N, p_P, p_K)

Acción conjunta:
    a = (a_N, a_P, a_K)

Costo inmediato:
    c(s,a) = p_N a_N + p_P a_P + p_K a_K
             - ingresos por venta (actualmente 0)

El MDP es estacionario e infinito. Por eso:
    0 <= BETA < 1

Se usan COSTOS POSITIVOS escalados y:
    minimize=True

IMPORTANTE SOBRE EL REPOSITORIO
===============================
El constructor `dtmdp` valida las matrices mediante `_check_transition_matrices`.
La implementación del repositorio hace una operación equivalente a apilar las
matrices para comprobar que las filas sumen 1. Para 125 matrices de
3375x3375 esto puede requerir varios GiB ADICIONALES temporalmente.

Además, `value_iteration` del repositorio recorre explícitamente:
    estados x acciones x estados_siguientes
en Python. Para el modelo conjunto este solver puede ser extremadamente lento.

Este script NO evita esas limitaciones: las deja visibles porque el objetivo
es intentar correr el modelo completo con la implementación real de jMarkov
y observar exactamente dónde aparece el cuello de botella.

Para proteger el computador, por defecto se hace un pre-chequeo de RAM.
Si quieres obligar el intento aun cuando el cálculo sugiera riesgo:
    FORZAR_EJECUCION = True

ADVERTENCIA:
Si el sistema operativo se queda sin RAM, puede cerrar Python antes de que
Python alcance a generar un MemoryError.
"""

from pathlib import Path
import itertools
import traceback
import time

import numpy as np
import pandas as pd

from jmarkov.mdp.dtmdp import dtmdp


# -----------------------------------------------------------------------------
# CONFIGURACIÓN
# -----------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]
TAB_DIR = BASE_DIR / "outputs" / "tables"
TAB_DIR.mkdir(parents=True, exist_ok=True)

CONFIGURACION = "5_categorias"
METODO = "equal_width"

CAPACIDAD_TOTAL = 66_000.0
RHO = 0.05

# dtmdp exige 0 <= beta < 1.
# Tasa nominal anual de costo de capital
# Damodaran, – Vale, Fertilizers. En el PDF cfpacket1spr24.pdf, la tabla “The Hurdle Rate” muestra para Fertilizers un Cost of capital (in US$) = 9.14%. 
R_NOMINAL_ANUAL = 0.0914

# Inflación esperada de largo plazo en USD
# 10-Year Breakeven Inflation Rate, enero 2026
INFLACION_ESPERADA = 0.0231

# Tasa real anual mediante ecuación de Fisher
R_REAL_ANUAL = (
    (1 + R_NOMINAL_ANUAL)
    / (1 + INFLACION_ESPERADA)
    - 1
)

# Conversión de tasa efectiva anual a efectiva mensual
R_REAL_MENSUAL = (
    (1 + R_REAL_ANUAL)**(1 / 12)
    - 1
)

# Factor de descuento mensual
BETA = 1 / (1 + R_REAL_MENSUAL)

print(f"Tasa nominal anual: {R_NOMINAL_ANUAL:.4%}")
print(f"Inflación esperada: {INFLACION_ESPERADA:.4%}")
print(f"Tasa real anual: {R_REAL_ANUAL:.4%}")
print(f"Tasa real mensual: {R_REAL_MENSUAL:.4%}")
print(f"Beta mensual: {BETA:.6f}")

# Tolerancia usada por value_iteration del repositorio.
TOLERANCIA = 1e-6

# Escalamiento puramente numérico.
COST_SCALE = 1_000_000.0

# Con minimize=True, las acciones infactibles reciben un costo muy grande.
COSTO_INFACTIBLE = 1_000_000.0

# float32 reduce la RAM de cada matriz a ~0.042 GiB.
DTYPE_TRANSICIONES = np.float32
DTYPE_COSTOS = np.float64

# Protección inicial.
FORZAR_EJECUCION = False

# Margen sobre matrices + temporal estimado del chequeo del constructor.
FACTOR_SEGURIDAD_MEMORIA = 1.20

VENTAS_ICA_KG_ANIO = {
    "N": 456_865_757,
    "P": 56_474_803,
    "K": 200_075_415,
}

DEMANDA = {
    c: kg / 1000.0 / 12.0
    for c, kg in VENTAS_ICA_KG_ANIO.items()
}

DEMANDA_TOTAL = sum(DEMANDA.values())

CAPACIDAD = {
    c: CAPACIDAD_TOTAL * DEMANDA[c] / DEMANDA_TOTAL
    for c in DEMANDA
}

DELTA = {
    c: RHO * DEMANDA[c]
    for c in DEMANDA
}

PRECIO_VENTA = {
    "N": 0.0,
    "P": 0.0,
    "K": 0.0,
}

COMMODITY_ARCHIVO = {
    "N": "nitrogeno",
    "P": "fosfat",
    "K": "potash",
}


# -----------------------------------------------------------------------------
# CSV
# -----------------------------------------------------------------------------

def detectar_separador(path: Path) -> str:
    primera_linea = path.read_text(
        encoding="utf-8-sig"
    ).splitlines()[0]

    return (
        ";"
        if primera_linea.count(";") > primera_linea.count(",")
        else ","
    )


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
        raise ValueError(
            f"{c}: se esperaban 5 categorías y se encontraron {len(df)}."
        )

    return pd.Series(
        df["precio_punto_medio"].to_numpy(float),
        index=df["categoria"].astype(int).to_numpy(),
    )


def cargar_matriz_transicion(c: str) -> pd.DataFrame:
    path = TAB_DIR / (
        f"transiciones_matriz_{COMMODITY_ARCHIVO[c]}_"
        f"{CONFIGURACION}.csv"
    )

    P = leer_csv_auto(path, index_col=0)

    P.index = (
        P.index.astype(str).str.strip().astype(int)
    )
    P.columns = (
        P.columns.astype(str).str.strip().astype(int)
    )

    return P.astype(float)


def preparar_cadena_operativa(
    P: pd.DataFrame,
    c: str,
    tol: float = 1e-12,
):
    row_sums = P.sum(axis=1, skipna=True)

    validas = row_sums[row_sums > tol].index.tolist()
    invalidas = row_sums[row_sums <= tol].index.tolist()

    if invalidas:
        incoming = (
            P.loc[validas, invalidas]
            .fillna(0)
            .to_numpy(float)
        )

        if np.any(incoming > tol):
            raise ValueError(
                f"{c}: categorías sin salida pero alcanzables: {invalidas}."
            )

        print(
            f"ADVERTENCIA {c}: se excluyen categorías "
            f"inalcanzables sin salida {invalidas}."
        )

    P_op = (
        P.loc[validas, validas]
        .fillna(0)
        .to_numpy(float)
    )

    P_op = P_op / P_op.sum(axis=1, keepdims=True)

    return validas, P_op, invalidas


# -----------------------------------------------------------------------------
# INVENTARIO Y ACCIONES
# -----------------------------------------------------------------------------

def niveles_inventario(c: str) -> np.ndarray:
    max_carry = CAPACIDAD[c] - DEMANDA[c]

    n = int(
        np.floor(
            max_carry / DELTA[c] + 1e-12
        )
    )

    return np.arange(n + 1, dtype=float) * DELTA[c]


def acciones_globales_por_commodity(
    c: str,
    niveles: np.ndarray,
) -> np.ndarray:
    valores = {
        round(
            DEMANDA[c] + inv_sig - inv_actual,
            10,
        )
        for inv_actual in niveles
        for inv_sig in niveles
        if (
            DEMANDA[c] + inv_sig - inv_actual
            >= -1e-9
        )
    }

    return np.array(sorted(valores), dtype=float)


def siguiente_inventario(
    c: str,
    inv_actual: float,
    compra: float,
    niveles: np.ndarray,
    tol: float = 1e-6,
):
    pre_demanda = inv_actual + compra
    inv_sig = pre_demanda - DEMANDA[c]

    if compra < -tol:
        return None

    if pre_demanda + tol < DEMANDA[c]:
        return None

    if pre_demanda > CAPACIDAD[c] + tol:
        return None

    diferencias = np.abs(niveles - inv_sig)
    idx = int(np.argmin(diferencias))

    if diferencias[idx] > tol:
        return None

    return float(niveles[idx])


# -----------------------------------------------------------------------------
# MEMORIA
# -----------------------------------------------------------------------------

def ram_disponible_gib():
    try:
        import psutil
        return psutil.virtual_memory().available / 1024**3
    except Exception:
        return None


def estimar_memoria(
    n_states: int,
    n_actions: int,
):
    item = np.dtype(DTYPE_TRANSICIONES).itemsize

    bytes_one = (
        n_states
        * n_states
        * item
    )

    bytes_matrices = (
        bytes_one
        * n_actions
    )

    bytes_costos = (
        n_states
        * n_actions
        * np.dtype(DTYPE_COSTOS).itemsize
    )

    # El chequeo del constructor del repo puede materializar temporalmente
    # una representación apilada de todas las matrices.
    bytes_temporal_check = bytes_matrices

    bytes_total_estimado = (
        bytes_matrices
        + bytes_costos
        + bytes_temporal_check
    )

    return {
        "una_matriz_GiB":
            bytes_one / 1024**3,
        "matrices_GiB":
            bytes_matrices / 1024**3,
        "costos_GiB":
            bytes_costos / 1024**3,
        "temporal_check_GiB":
            bytes_temporal_check / 1024**3,
        "total_con_margen_GiB":
            bytes_total_estimado
            * FACTOR_SEGURIDAD_MEMORIA
            / 1024**3,
    }


def verificar_memoria(
    n_states: int,
    n_actions: int,
):
    mem = estimar_memoria(
        n_states,
        n_actions,
    )

    disponible = ram_disponible_gib()

    print("\n--- PRE-CHEQUEO DE MEMORIA PARA dtmdp ---")
    print(
        f"Una matriz: {mem['una_matriz_GiB']:.3f} GiB"
    )
    print(
        f"{n_actions} matrices: {mem['matrices_GiB']:.2f} GiB"
    )
    print(
        f"Matriz de costos: {mem['costos_GiB']:.3f} GiB"
    )
    print(
        "Temporal estimado durante validación del constructor: "
        f"{mem['temporal_check_GiB']:.2f} GiB"
    )
    print(
        f"Estimado con margen: "
        f"{mem['total_con_margen_GiB']:.2f} GiB"
    )

    if disponible is not None:
        print(
            f"RAM disponible detectada: {disponible:.2f} GiB"
        )

        if (
            disponible < mem["total_con_margen_GiB"]
            and not FORZAR_EJECUCION
        ):
            raise MemoryError(
                "\nEl modelo puede exceder la RAM durante la validación "
                "interna de `dtmdp`.\n"
                f"Disponible: {disponible:.2f} GiB\n"
                f"Estimado con margen: "
                f"{mem['total_con_margen_GiB']:.2f} GiB\n\n"
                "Para intentar igualmente:\n"
                "    FORZAR_EJECUCION = True\n"
            )

    return mem


# -----------------------------------------------------------------------------
# METADATA CONJUNTA
# -----------------------------------------------------------------------------

def construir_metadata_conjunta():
    cadenas = {}
    precios = {}

    for c in ["N", "P", "K"]:
        cats, P, invalidas = preparar_cadena_operativa(
            cargar_matriz_transicion(c),
            c,
        )

        cadenas[c] = {
            "cats": cats,
            "P": P,
            "invalidas": invalidas,
        }

        precios[c] = (
            cargar_precios_representativos(c)
            .loc[cats]
        )

    inv = {
        c: niveles_inventario(c)
        for c in ["N", "P", "K"]
    }

    acciones_c = {
        c: acciones_globales_por_commodity(
            c,
            inv[c],
        )
        for c in ["N", "P", "K"]
    }

    inv_tuples = list(
        itertools.product(
            inv["N"],
            inv["P"],
            inv["K"],
        )
    )

    price_tuples = list(
        itertools.product(
            cadenas["N"]["cats"],
            cadenas["P"]["cats"],
            cadenas["K"]["cats"],
        )
    )

    state_tuples = [
        (
            float(iN),
            float(iP),
            float(iK),
            int(pN),
            int(pP),
            int(pK),
        )
        for (
            iN,
            iP,
            iK,
        ) in inv_tuples
        for (
            pN,
            pP,
            pK,
        ) in price_tuples
    ]

    states = np.array(
        [
            (
                f"IN={iN:.6f}|IP={iP:.6f}|IK={iK:.6f}|"
                f"pN={pN}|pP={pP}|pK={pK}"
            )
            for (
                iN,
                iP,
                iK,
                pN,
                pP,
                pK,
            ) in state_tuples
        ],
        dtype=object,
    )

    action_tuples = list(
        itertools.product(
            acciones_c["N"],
            acciones_c["P"],
            acciones_c["K"],
        )
    )

    # dtmdp NO presenta el truncamiento <U1 de dtsdp, así que aquí podemos
    # usar identificadores legibles A001,...,A125.
    actions = np.array(
        [
            f"A{i+1:03d}"
            for i in range(len(action_tuples))
        ],
        dtype=object,
    )

    return {
        "states": states,
        "state_tuples": state_tuples,
        "inv_tuples": inv_tuples,
        "price_tuples": price_tuples,
        "actions": actions,
        "action_tuples": action_tuples,
        "inv": inv,
        "acciones_c": acciones_c,
        "cadenas": cadenas,
        "precios": precios,
    }


# -----------------------------------------------------------------------------
# MATRICES DEL MDP
# -----------------------------------------------------------------------------

def matriz_precios_conjunta(cadenas) -> np.ndarray:
    P_joint = np.kron(
        np.kron(
            cadenas["N"]["P"],
            cadenas["P"]["P"],
        ),
        cadenas["K"]["P"],
    )

    return P_joint.astype(
        DTYPE_TRANSICIONES,
        copy=False,
    )


def construir_mdp_denso(data):
    states = data["states"]
    inv_tuples = data["inv_tuples"]
    price_tuples = data["price_tuples"]
    actions = data["actions"]
    action_tuples = data["action_tuples"]
    inv = data["inv"]
    precios = data["precios"]

    nS = len(states)
    nA = len(actions)
    nP = len(price_tuples)

    print(
        "\nConstruyendo matriz conjunta de transición de precios..."
    )

    P_joint = matriz_precios_conjunta(
        data["cadenas"]
    )

    if P_joint.shape != (nP, nP):
        raise ValueError(
            f"P_joint tiene shape {P_joint.shape}, "
            f"esperado {(nP, nP)}."
        )

    inv_index = {
        (
            round(float(iN), 10),
            round(float(iP), 10),
            round(float(iK), 10),
        ): idx
        for idx, (
            iN,
            iP,
            iK,
        ) in enumerate(inv_tuples)
    }

    price_vector = {
        "N": np.array(
            [
                float(precios["N"].loc[p[0]])
                for p in price_tuples
            ],
            dtype=float,
        ),
        "P": np.array(
            [
                float(precios["P"].loc[p[1]])
                for p in price_tuples
            ],
            dtype=float,
        ),
        "K": np.array(
            [
                float(precios["K"].loc[p[2]])
                for p in price_tuples
            ],
            dtype=float,
        ),
    }

    # dtmdp espera shape (|S|, |A|).
    immediate_costs = np.full(
        (nS, nA),
        COSTO_INFACTIBLE,
        dtype=DTYPE_COSTOS,
    )

    transition_matrices = {}

    print(
        f"Construyendo {nA} matrices densas "
        f"{nS}x{nS}..."
    )

    inicio = time.perf_counter()

    for a_idx, (
        label,
        accion,
    ) in enumerate(
        zip(actions, action_tuples)
    ):
        aN, aP, aK = map(float, accion)

        M = np.zeros(
            (nS, nS),
            dtype=DTYPE_TRANSICIONES,
        )

        for inv_idx, (
            iN,
            iP,
            iK,
        ) in enumerate(inv_tuples):

            iN_sig = siguiente_inventario(
                "N",
                float(iN),
                aN,
                inv["N"],
            )
            iP_sig = siguiente_inventario(
                "P",
                float(iP),
                aP,
                inv["P"],
            )
            iK_sig = siguiente_inventario(
                "K",
                float(iK),
                aK,
                inv["K"],
            )

            row_ini = inv_idx * nP
            row_fin = row_ini + nP

            if (
                iN_sig is None
                or iP_sig is None
                or iK_sig is None
            ):
                idxs = np.arange(
                    row_ini,
                    row_fin,
                )
                M[idxs, idxs] = 1.0
                continue

            next_inv_key = (
                round(iN_sig, 10),
                round(iP_sig, 10),
                round(iK_sig, 10),
            )

            next_inv_idx = inv_index[
                next_inv_key
            ]

            col_ini = next_inv_idx * nP
            col_fin = col_ini + nP

            M[
                row_ini:row_fin,
                col_ini:col_fin,
            ] = P_joint

            costos = (
                price_vector["N"] * aN
                + price_vector["P"] * aP
                + price_vector["K"] * aK
                - PRECIO_VENTA["N"] * DEMANDA["N"]
                - PRECIO_VENTA["P"] * DEMANDA["P"]
                - PRECIO_VENTA["K"] * DEMANDA["K"]
            )

            immediate_costs[
                row_ini:row_fin,
                a_idx,
            ] = (
                costos / COST_SCALE
            )

        row_sums = M.sum(axis=1)

        if not np.allclose(
            row_sums,
            1.0,
            atol=1e-6,
        ):
            malos = np.where(
                ~np.isclose(
                    row_sums,
                    1.0,
                    atol=1e-6,
                )
            )[0]

            raise ValueError(
                f"{label}: filas no estocásticas "
                f"{malos[:10].tolist()}."
            )

        transition_matrices[str(label)] = M

        if (
            a_idx == 0
            or (a_idx + 1) % 5 == 0
            or a_idx + 1 == nA
        ):
            transcurrido = (
                time.perf_counter()
                - inicio
            )

            print(
                f"  matrices: {a_idx+1}/{nA} "
                f"| {transcurrido/60:.1f} min"
            )

    return transition_matrices, immediate_costs


# -----------------------------------------------------------------------------
# EXPORTACIÓN
# -----------------------------------------------------------------------------

def exportar_mapa_acciones(data):
    filas = []

    for label, (
        aN,
        aP,
        aK,
    ) in zip(
        data["actions"],
        data["action_tuples"],
    ):
        filas.append({
            "accion_codigo": str(label),
            "compra_N_ton": float(aN),
            "compra_P_ton": float(aP),
            "compra_K_ton": float(aK),
        })

    df = pd.DataFrame(filas)

    path = (
        TAB_DIR
        / "mdp_conjunto_mapa_acciones.csv"
    )

    df.to_csv(
        path,
        index=False,
        encoding="utf-8-sig",
    )

    return path


def exportar_resultados(
    data,
    value_function,
    optimal_policy,
    expected_policy_value_scaled,
):
    action_map = {
        str(label): tuple(map(float, action))
        for label, action in zip(
            data["actions"],
            data["action_tuples"],
        )
    }

    filas = []

    for s_idx, state_tuple in enumerate(
        data["state_tuples"]
    ):
        (
            iN,
            iP,
            iK,
            pN,
            pP,
            pK,
        ) = state_tuple

        state_label = data["states"][s_idx]

        accion_codigo = str(
            optimal_policy[state_label]
        )

        aN, aP, aK = action_map[
            accion_codigo
        ]

        filas.append({
            "I_N_ton": iN,
            "I_P_ton": iP,
            "I_K_ton": iK,
            "cat_N": pN,
            "cat_P": pP,
            "cat_K": pK,
            "accion_codigo": accion_codigo,
            "compra_N_ton": aN,
            "compra_P_ton": aP,
            "compra_K_ton": aK,
            "valor_costo_descuento_USD":
                float(value_function[s_idx])
                * COST_SCALE,
        })

    df = pd.DataFrame(filas)

    path_pol = (
        TAB_DIR
        / "mdp_conjunto_value_iteration_politica.csv"
    )

    df.to_csv(
        path_pol,
        index=False,
        encoding="utf-8-sig",
    )

    resumen = pd.DataFrame([{
        "metodo_jmarkov":
            "dtmdp_value_iteration",
        "horizonte":
            "infinito",
        "n_estados":
            len(data["states"]),
        "n_acciones":
            len(data["actions"]),
        "beta":
            BETA,
        "tolerancia":
            TOLERANCIA,
        "cost_scale":
            COST_SCALE,
        "expected_policy_value_scaled":
            float(expected_policy_value_scaled),
        "expected_policy_value_USD":
            float(expected_policy_value_scaled)
            * COST_SCALE,
        "interpretacion":
            "producto punto entre distribución estacionaria "
            "de P_pi y vector V de costos descontados",
    }])

    path_resumen = (
        TAB_DIR
        / "mdp_conjunto_value_iteration_resumen.csv"
    )

    resumen.to_csv(
        path_resumen,
        index=False,
        encoding="utf-8-sig",
    )

    return path_pol, path_resumen


# -----------------------------------------------------------------------------
# EJECUCIÓN
# -----------------------------------------------------------------------------

def ejecutar_mdp_conjunto():
    print("=" * 100)
    print(
        "MDP CONJUNTO N-P-K — jMarkov dtmdp + VALUE ITERATION"
    )
    print("=" * 100)

    print("\n[1/7] Construyendo metadata...")
    data = construir_metadata_conjunta()

    nS = len(data["states"])
    nA = len(data["actions"])

    print(
        f"Estados conjuntos |S| = {nS:,}"
    )
    print(
        f"Acciones conjuntas |A| = {nA:,}"
    )

    mapa = exportar_mapa_acciones(data)
    print(
        f"Mapa de acciones: {mapa}"
    )

    print("\n[2/7] Verificando memoria...")
    verificar_memoria(
        nS,
        nA,
    )

    print("\n[3/7] Construyendo matrices y costos...")
    transition_matrices, immediate_costs = construir_mdp_denso(
        data
    )

    print("\n[4/7] Instanciando dtmdp del repositorio...")
    t0 = time.perf_counter()

    modelo = dtmdp(
        data["states"],
        data["actions"],
        transition_matrices,
        immediate_costs,
        BETA,
    )

    print(
        "Instancia dtmdp creada correctamente "
        f"en {(time.perf_counter()-t0)/60:.2f} min."
    )

    print("\n[5/7] Ejecutando VALUE ITERATION del repositorio...")
    print(
        "NOTA: la implementación oficial usa bucles Python "
        "estado × acción × estado_siguiente; esta etapa puede ser muy lenta."
    )

    t0 = time.perf_counter()
    

    inicio_vi = time.perf_counter()

    print("Iniciando value iteration...")
    value_function, optimal_policy = modelo.solve(
        TOLERANCIA,
        minimize=True,
        method="value_iteration",
    )
    fin_vi = time.perf_counter()

    print(
    f"Value iteration terminó en "
    f"{(fin_vi - inicio_vi)/60:.2f} minutos"
    )
    from jmarkov.dtmc import dtmc

# ---------------------------------------------------------
# 1. VECTOR V*
# ---------------------------------------------------------

    V = np.asarray(value_function, dtype=float)

    print("\nVECTOR V*:")
    print(V)

# ---------------------------------------------------------
# 2. MATRIZ DE TRANSICIÓN BAJO LA POLÍTICA ÓPTIMA
# ---------------------------------------------------------

    P_pi = modelo.policy_transition_matrix(
        optimal_policy
    )

# ---------------------------------------------------------
# 3. VECTOR DE PROBABILIDADES DE ESTADO ESTABLE
# ---------------------------------------------------------

    cadena_politica = dtmc(P_pi)

    steady_state = cadena_politica.steady_state()

    print("\nVECTOR DE PROBABILIDADES DE ESTADO ESTABLE:")
    print(steady_state)

# ---------------------------------------------------------
# 4. EXPECTED POLICY VALUE
# ---------------------------------------------------------

    expected_policy_value = modelo.expected_policy_value(
    V,
    optimal_policy,
    )

    print("\nEXPECTED POLICY VALUE:")
    print(expected_policy_value)

    print("\nEXPECTED POLICY VALUE EN USD:")
    print(expected_policy_value * COST_SCALE)
    print(
        "Value iteration completado en "
        f"{(time.perf_counter()-t0)/60:.2f} min."
    )

    value_function = np.asarray(
        value_function,
        dtype=float,
    )

    print("\n[6/7] Calculando P_pi y expected_policy_value...")

    P_pi = modelo.policy_transition_matrix(
        optimal_policy
    )

    path_Ppi = (
        TAB_DIR
        / "mdp_conjunto_policy_transition_matrix.csv"
    )

    pd.DataFrame(
        P_pi,
        index=data["states"],
        columns=data["states"],
    ).to_csv(
        path_Ppi,
        encoding="utf-8-sig",
    )

    expected_policy_value_scaled = (
        modelo.expected_policy_value(
            value_function,
            optimal_policy,
        )
    )

    print(
        "expected_policy_value = "
        f"USD "
        f"{float(expected_policy_value_scaled)*COST_SCALE:,.2f}"
    )

    print("\n[7/7] Exportando resultados...")

    path_pol, path_resumen = exportar_resultados(
        data,
        value_function,
        optimal_policy,
        expected_policy_value_scaled,
    )

    print(
        f"Política: {path_pol}"
    )
    print(
        f"Resumen: {path_resumen}"
    )
    print(
        f"P_pi: {path_Ppi}"
    )

    return (
        value_function,
        optimal_policy,
        expected_policy_value_scaled,
    )


# -----------------------------------------------------------------------------
# MAIN CON TRACEBACK
# -----------------------------------------------------------------------------

def main():
    try:
        ejecutar_mdp_conjunto()

    except MemoryError as exc:
        print("\n" + "=" * 100)
        print("MEMORY ERROR EN EL MDP CONJUNTO")
        print("=" * 100)
        print(str(exc))
        print("\nTRACEBACK COMPLETO:")
        traceback.print_exc()
        raise

    except Exception as exc:
        print("\n" + "=" * 100)
        print("ERROR EN EL MDP CONJUNTO")
        print("=" * 100)
        print(
            f"{type(exc).__name__}: {exc}"
        )
        print("\nTRACEBACK COMPLETO:")
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()

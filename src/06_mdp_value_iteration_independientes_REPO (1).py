"""
06_mdp_value_iteration_independientes.py
---------------------------------------
Resuelve N, P y K POR SEPARADO como MDP de horizonte infinito mediante
VALUE ITERATION usando `jmarkov.mdp.dtmdp`.

Este script NO reemplaza el SDP finito de 5 epocas del archivo 05B.
Su objetivo es cumplir la prueba con iteracion por valor y comparar:

    SDP finito (dtsdp + backward iteration)
    vs.
    MDP infinito (dtmdp + value iteration)

Se mantienen, en lo posible, los mismos elementos del benchmark 05B:
  - discretizacion de precio: 5 categorias, equal_width;
  - inventario discretizado en pasos de 5% de la demanda;
  - demanda fija y obligatoria;
  - capacidad distribuida proporcionalmente;
  - costo de almacenamiento = 0;
  - precio de venta = 0 por ahora;
  - acciones codificadas A, B, C, D, E;
  - costo transformado a retorno negativo para maximizar con jMarkov.

IMPORTANTE:
El MDP de horizonte infinito necesita beta < 1. Se usa BETA=0.95 como
valor de prueba. Este parametro SI tiene interpretacion economica en un
MDP infinito y debe validarse con el asesor o someterse a sensibilidad.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from jmarkov.mdp.dtmdp import dtmdp

# -----------------------------------------------------------------------------
# RUTAS Y PARAMETROS GENERALES
# -----------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]
TAB_DIR = BASE_DIR / "outputs" / "tables"
TAB_DIR.mkdir(parents=True, exist_ok=True)

CONFIGURACION = "5_categorias"
METODO = "equal_width"
CAPACIDAD_TOTAL = 66_000.0

# Misma resolucion de inventario/acciones usada en 05B.
RHO = 0.05

# MDP infinito: se requiere beta < 1.
# dtmdp exige 0 <= beta < 1.
# Tasa nominal anual de costo de capital
# Damodaran, Chemical (Basic), enero 2026
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


# Tolerancia (GAP) que usa solve() de dtmdp en la version del curso.
TOLERANCIA = 1e-6

# Escalamiento numerico: no cambia la politica optima.
COST_SCALE = 1_000_000.0

# Costo muy alto para acciones no factibles.
# Como usamos minimize=True, una accion no factible debe ser muy costosa.
COSTO_INFACTIBLE = 1_000_000.0

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

# Pendiente de definir con el asesor/fuente.
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
# FUNCIONES DE CARGA
# -----------------------------------------------------------------------------

def detectar_separador(path: Path) -> str:
    """Detecta si un CSV usa coma o punto y coma."""
    primera_linea = path.read_text(encoding="utf-8-sig").splitlines()[0]
    return ";" if primera_linea.count(";") > primera_linea.count(",") else ","


def leer_csv_auto(path: Path, **kwargs) -> pd.DataFrame:
    """Lee un CSV detectando automaticamente ',' o ';'."""
    sep = detectar_separador(path)
    return pd.read_csv(path, sep=sep, encoding="utf-8-sig", **kwargs)


def cargar_precios_representativos(c: str) -> pd.Series:
    """Carga el punto medio de cada categoria de precio del commodity c."""
    path = TAB_DIR / "discretizacion_resumen.csv"
    df = leer_csv_auto(path)

    df = df[
        (df["configuracion"] == CONFIGURACION)
        & (df["metodo"] == METODO)
        & (df["commodity"] == COMMODITY_ARCHIVO[c])
    ].sort_values("categoria")

    if len(df) != 5:
        raise ValueError(
            f"{c}: se esperaban 5 categorias en {CONFIGURACION} "
            f"y se encontraron {len(df)}."
        )

    return pd.Series(
        df["precio_punto_medio"].to_numpy(float),
        index=df["categoria"].astype(int).to_numpy(),
    )


def cargar_matriz_transicion(c: str) -> pd.DataFrame:
    """Carga la matriz MLE de transicion de precios de 5 categorias."""
    path = TAB_DIR / (
        f"transiciones_matriz_{COMMODITY_ARCHIVO[c]}_{CONFIGURACION}.csv"
    )

    P = leer_csv_auto(path, index_col=0)

    # Limpieza robusta de indices/columnas.
    P.index = P.index.astype(str).str.strip().astype(int)
    P.columns = P.columns.astype(str).str.strip().astype(int)

    return P.astype(float)


def preparar_cadena_operativa(P: pd.DataFrame, c: str, tol: float = 1e-12):
    """
    Elimina categorias sin transicion saliente solo si ademas son inalcanzables
    desde las categorias validas. Con 5 categorias esperamos que todas sean validas.
    """
    row_sums = P.sum(axis=1, skipna=True)
    validas = row_sums[row_sums > tol].index.tolist()
    invalidas = row_sums[row_sums <= tol].index.tolist()

    if invalidas:
        incoming = P.loc[validas, invalidas].fillna(0).to_numpy(float)
        if np.any(incoming > tol):
            raise ValueError(
                f"{c}: categorias sin salida pero alcanzables: {invalidas}. "
                "No se imputan probabilidades automaticamente."
            )
        print(
            f"ADVERTENCIA {c}: se excluyen categorias inalcanzables "
            f"sin salida {invalidas}."
        )

    P_op = P.loc[validas, validas].fillna(0).to_numpy(float)
    P_op = P_op / P_op.sum(axis=1, keepdims=True)

    return validas, P_op, invalidas


# -----------------------------------------------------------------------------
# INVENTARIOS Y ACCIONES
# -----------------------------------------------------------------------------

def niveles_inventario(c: str) -> np.ndarray:
    """Niveles de inventario que pueden mantenerse entre meses."""
    max_carry = CAPACIDAD[c] - DEMANDA[c]
    n = int(np.floor(max_carry / DELTA[c] + 1e-12))
    return np.arange(n + 1, dtype=float) * DELTA[c]


def acciones_globales(c: str, niveles: np.ndarray) -> np.ndarray:
    """
    Genera la union de compras que permiten pasar de un nivel de inventario
    de la malla a otro:

        compra = demanda + inventario_siguiente - inventario_actual
    """
    vals = {
        round(DEMANDA[c] + inv_sig - inv_act, 10)
        for inv_act in niveles
        for inv_sig in niveles
        if DEMANDA[c] + inv_sig - inv_act >= -1e-9
    }
    return np.array(sorted(vals), dtype=float)


def es_factible(
    c: str,
    inv_actual: float,
    compra: float,
    niveles: np.ndarray,
    tol: float = 1e-6,
) -> bool:
    """Comprueba demanda, capacidad y pertenencia a la malla de inventario."""
    pre_demanda = inv_actual + compra
    inv_sig = pre_demanda - DEMANDA[c]

    coincide_malla = np.any(
        np.isclose(inv_sig, niveles, atol=tol, rtol=0)
    )

    return (
        compra >= -tol
        and pre_demanda + tol >= DEMANDA[c]
        and pre_demanda <= CAPACIDAD[c] + tol
        and coincide_malla
    )


# -----------------------------------------------------------------------------
# CONSTRUCCION DEL MDP PARA UN COMMODITY
# -----------------------------------------------------------------------------

def construir_mdp(c: str):
    """Construye estados, acciones, P(a) y R(s,a) para un commodity."""

    precios = cargar_precios_representativos(c)

    cats, Pprecio, invalidas = preparar_cadena_operativa(
        cargar_matriz_transicion(c), c
    )
    precios = precios.loc[cats]

    niveles = niveles_inventario(c)
    compras = acciones_globales(c, niveles)

    # Estados: (inventario, categoria de precio)
    state_tuples = [
        (float(inv), int(cat))
        for inv in niveles
        for cat in cats
    ]

    states = np.array(
        [f"I={inv:.6f}|p={cat}" for inv, cat in state_tuples],
        dtype=object,
    )

    # Etiquetas cortas para evitar problemas de strings en jMarkov.
    letras = np.array(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"), dtype=object)
    if len(compras) > len(letras):
        raise ValueError("Hay mas acciones que letras disponibles.")
    actions = letras[:len(compras)].copy()

    state_index = {s: idx for idx, s in enumerate(state_tuples)}
    cat_index = {cat: idx for idx, cat in enumerate(cats)}

    nS = len(states)
    nA = len(actions)

    # dtmdp requiere immediate_returns con dimension estados x acciones.
    # Al usar minimize=True, almacenamos COSTOS POSITIVOS escalados.
    R = np.full((nS, nA), COSTO_INFACTIBLE, dtype=float)

    # dtmdp requiere un diccionario: accion -> matriz |S|x|S|.
    transition_matrices = {}

    for a_idx, (label, compra) in enumerate(zip(actions, compras)):
        M = np.zeros((nS, nS), dtype=float)

        for s_idx, (inv_actual, cat_actual) in enumerate(state_tuples):

            if not es_factible(c, inv_actual, compra, niveles):
                # Accion imposible: self-loop + costo muy alto.
                M[s_idx, s_idx] = 1.0
                continue

            # Inventario del siguiente periodo, deterministico.
            inv_sig = inv_actual + compra - DEMANDA[c]
            inv_sig = float(
                niveles[np.argmin(np.abs(niveles - inv_sig))]
            )

            # Precio siguiente, estocastico segun la matriz MLE.
            p_idx = cat_index[cat_actual]
            for next_p_idx, cat_sig in enumerate(cats):
                prob = Pprecio[p_idx, next_p_idx]
                next_state = (inv_sig, int(cat_sig))
                j = state_index[next_state]
                M[s_idx, j] += prob

            # Costo inmediato del periodo.
            precio_actual = float(precios.loc[cat_actual])
            costo = (
                precio_actual * compra
                - PRECIO_VENTA[c] * DEMANDA[c]
            )

            # Costo positivo escalado; dtmdp se encarga de minimizarlo.
            R[s_idx, a_idx] = costo / COST_SCALE

        # Validacion numerica de filas estocasticas.
        row_sums = M.sum(axis=1)
        if not np.allclose(row_sums, 1.0, atol=1e-10):
            malos = np.where(~np.isclose(row_sums, 1.0, atol=1e-10))[0]
            raise ValueError(
                f"{c}, accion {label}: hay filas que no suman 1: "
                f"{malos[:10].tolist()}"
            )

        transition_matrices[label] = M

    return {
        "commodity": c,
        "states": states,
        "state_tuples": state_tuples,
        "actions": actions,
        "compras": compras,
        "transition_matrices": transition_matrices,
        "returns": R,
        "niveles": niveles,
        "cats": cats,
        "invalidas": invalidas,
    }


# -----------------------------------------------------------------------------
# SOLUCION CON VALUE ITERATION
# -----------------------------------------------------------------------------

def resolver_value_iteration(c: str):
    data = construir_mdp(c)

    modelo = dtmdp(
        data["states"],
        data["actions"],
        data["transition_matrices"],
        data["returns"],
        BETA,
    )

    # API oficial de dtmdp: minimizamos directamente los costos positivos.
    value_functions, optimal_policy = modelo.solve(
        TOLERANCIA,
        minimize=True,
        method="value_iteration",
    )
    from jmarkov.dtmc import dtmc

# ---------------------------------------------------------
# 1. VECTOR V*
# ---------------------------------------------------------

    V = np.asarray(value_functions, dtype=float)

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
    print(np.sum(steady_state))
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


    if isinstance(value_functions, dict):
        valores_vec = np.array(
            [float(value_functions[state]) for state in data["states"]],
            dtype=float,
        )
    else:
        valores_vec = np.asarray(value_functions, dtype=float)

    # Metodos nativos del repositorio jMarkov.
    policy_transition = modelo.policy_transition_matrix(optimal_policy)
    expected_policy_value_scaled = modelo.expected_policy_value(
        valores_vec,
        optimal_policy,
    )

    # Volver a USD.
    valor_costo_desc_aprox = valores_vec * COST_SCALE
    expected_policy_value_usd = (
        float(expected_policy_value_scaled) * COST_SCALE
    )

    # Exportar P^pi, la matriz de transicion inducida por la politica optima.
    pd.DataFrame(
        policy_transition,
        index=data["states"],
        columns=data["states"],
    ).to_csv(
        TAB_DIR / f"mdp_value_iteration_{c}_policy_transition_matrix.csv",
        index=True,
    )

    action_to_ton = {
        str(label): float(compra)
        for label, compra in zip(data["actions"], data["compras"])
    }

    filas = []

    for s_idx, (inv, cat) in enumerate(data["state_tuples"]):
        state_label = data["states"][s_idx]

        if isinstance(optimal_policy, dict):
            accion = str(optimal_policy[state_label])
        else:
            accion = str(optimal_policy[s_idx])

        compra = action_to_ton[accion]
        inv_sig = inv + compra - DEMANDA[c]

        filas.append({
            "commodity": c,
            "inventario_ton": inv,
            "cat_precio": int(cat),
            "accion": accion,
            "compra_ton": compra,
            "inventario_siguiente_ton": inv_sig,
            "valor_costo_descuento_aprox_USD": valor_costo_desc_aprox[s_idx],
        })

    df_pol = pd.DataFrame(filas)

    out_csv = TAB_DIR / f"mdp_value_iteration_{c}_politica.csv"
    df_pol.to_csv(out_csv, index=False)

    resumen = pd.DataFrame([{
        "commodity": c,
        "metodo": "value_iteration",
        "horizonte": "infinito",
        "n_estados": len(data["states"]),
        "n_acciones_globales": len(data["actions"]),
        "niveles_inventario": ";".join(
            f"{x:.4f}" for x in data["niveles"]
        ),
        "categorias_precio_operativas": ";".join(
            map(str, data["cats"])
        ),
        "categorias_excluidas": ";".join(
            map(str, data["invalidas"])
        ),
        "rho": RHO,
        "beta": BETA,
        "tolerancia": TOLERANCIA,
        "cost_scale": COST_SCALE,
        "expected_policy_value_scaled": float(expected_policy_value_scaled),
        "expected_policy_value_USD": expected_policy_value_usd,
        "interpretacion_expected_policy_value": "promedio de V(s) ponderado por la distribucion estacionaria de P_pi",
    }])

    resumen.to_csv(
        TAB_DIR / f"mdp_value_iteration_{c}_resumen.csv",
        index=False,
    )

    return df_pol, resumen, data


# -----------------------------------------------------------------------------
# COMPARACION OPCIONAL CONTRA EL SDP FINITO DE 05B
# -----------------------------------------------------------------------------

def comparar_con_sdp_finito(df_vi: pd.DataFrame) -> pd.DataFrame | None:
    """
    Si existe el archivo consolidado del 05B, compara la accion estacionaria
    de value iteration contra la accion del SDP en cada una de las 5 epocas.
    """
    path_sdp = TAB_DIR / "sdp_jmarkov_independientes_politicas.csv"
    if not path_sdp.exists():
        return None

    sdp = leer_csv_auto(path_sdp)

    columnas_sdp = {
        "commodity",
        "epoca",
        "inventario_ton",
        "cat_precio",
        "accion",
        "compra_ton",
    }
    if not columnas_sdp.issubset(sdp.columns):
        print(
            "ADVERTENCIA: existe el archivo del SDP, pero sus columnas "
            "no coinciden con las esperadas. Se omite la comparacion."
        )
        return None

    vi = df_vi.rename(columns={
        "accion": "accion_value_iteration",
        "compra_ton": "compra_value_iteration_ton",
        "inventario_siguiente_ton": "inv_sig_value_iteration_ton",
    })

    sdp2 = sdp.rename(columns={
        "accion": "accion_sdp",
        "compra_ton": "compra_sdp_ton",
        "inventario_siguiente_ton": "inv_sig_sdp_ton",
    })
    # Redondear inventarios para evitar diferencias mínimas de punto flotante
    sdp2["inventario_ton"] = sdp2["inventario_ton"].round(4)
    vi["inventario_ton"] = vi["inventario_ton"].round(4)

    # Asegurar que las categorías sean enteros
    sdp2["cat_precio"] = sdp2["cat_precio"].astype(int)
    vi["cat_precio"] = vi["cat_precio"].astype(int)
    comp = sdp2.merge(
        vi[[
            "commodity",
            "inventario_ton",
            "cat_precio",
            "accion_value_iteration",
            "compra_value_iteration_ton",
            "inv_sig_value_iteration_ton",
        ]],
        on=["commodity", "inventario_ton", "cat_precio"],
        how="left",
    )

    comp["misma_accion"] = (
        comp["accion_sdp"] == comp["accion_value_iteration"]
    )

    return comp


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------

def main():
    print("=" * 90)
    print("MDP DE HORIZONTE INFINITO - VALUE ITERATION CON jMarkov")
    print("=" * 90)
    print(f"Configuracion de precios: {CONFIGURACION}")
    print(f"RHO inventario/accion: {RHO:.0%}")
    print(f"beta: {BETA}")
    print(f"tolerancia: {TOLERANCIA}")
    print(
        "Se pasan costos positivos a dtmdp y se usa "
        "solve(minimize=True, method='value_iteration').\n"
    )

    resultados = []
    resumenes = []

    for c in ["N", "P", "K"]:
        df, resumen, data = resolver_value_iteration(c)
        resultados.append(df)
        resumenes.append(resumen)

        print(
            f"{c}: {len(data['states'])} estados, "
            f"{len(data['actions'])} acciones globales, "
            f"niveles inv={[round(x, 2) for x in data['niveles']]}"
        )
        print(
            f"   salida: outputs/tables/mdp_value_iteration_{c}_politica.csv"
        )
        print(
            f"   expected_policy_value: USD "
            f"{float(resumen.loc[0, 'expected_policy_value_USD']):,.2f}"
        )
        print(
            f"   P_pi: outputs/tables/"
            f"mdp_value_iteration_{c}_policy_transition_matrix.csv"
        )

    combinado = pd.concat(resultados, ignore_index=True)
    combinado.to_csv(
        TAB_DIR / "mdp_value_iteration_independientes_politicas.csv",
        index=False,
    )

    resumen_combinado = pd.concat(resumenes, ignore_index=True)
    resumen_combinado.to_csv(
        TAB_DIR / "mdp_value_iteration_independientes_resumen.csv",
        index=False,
    )

    print(
        "\nArchivo combinado: "
        "outputs/tables/mdp_value_iteration_independientes_politicas.csv"
    )
    print(
        "Resumen con expected_policy_value: "
        "outputs/tables/mdp_value_iteration_independientes_resumen.csv"
    )

    comparacion = comparar_con_sdp_finito(combinado)
    if comparacion is not None:
        comparacion.to_csv(
            TAB_DIR / "comparacion_sdp_vs_value_iteration.csv",
            index=False,
        )

        pct = 100.0 * comparacion["misma_accion"].mean()
        print(
            "Comparacion generada: outputs/tables/"
            "comparacion_sdp_vs_value_iteration.csv"
        )
        print(
            f"Coincidencia global de acciones SDP vs VI: {pct:.2f}%"
        )


if __name__ == "__main__":
    main()

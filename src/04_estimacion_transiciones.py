"""
04_estimacion_transiciones.py
--------------------------------
Estima la matriz de transición de cada commodity por MÁXIMA VEROSIMILITUD
(conteo de transiciones observadas entre categorías consecutivas), a partir
de las categorías ya asignadas en 03_discretizacion.py.


--- Lógica del estimador de máxima verosimilitud ---
Para una cadena de Markov de primer orden, el estimador de máxima
verosimilitud de la probabilidad de transición del estado i al estado j es
simplemente la frecuencia relativa observada:

    P_hat(i -> j) = n_ij / n_i.

donde:
    n_ij = número de veces que se observó una transición de la categoría i
           en el mes t a la categoría j en el mes t+1
    n_i. = número total de veces que el sistema estuvo en la categoría i
           en un mes t que SÍ tiene un mes t+1 observable (= suma de la
           fila i de la matriz de conteos)

Solo se cuentan transiciones entre meses consecutivos en el calendario real
(t y t+1 = t + 1 mes). Si hay un mes faltante en medio (NaN), esa transición
NO se cuenta, para no mezclar saltos de 2+ meses con transiciones mensuales.

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "processed"
FIG_DIR = BASE_DIR / "outputs" / "figures"
TAB_DIR = BASE_DIR / "outputs" / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)

COMMODITIES = ["nitrogeno", "fosfat", "potash"]
NOMBRES = {"nitrogeno": "Nitrógeno (Urea)", "fosfat": "Fósforo (Phosphat)", "potash": "Potasio (Potash)"}
CONFIGURACIONES = ["5_categorias", "10_categorias"]
METODO = "equal_width"
UMBRAL_MIN_TRANSICIONES = 5  # filas con menos transiciones salientes que esto se marcan como poco confiables

plt.rcParams.update({"figure.dpi": 130})


def limpiar_salidas_previas():
    for patron in ["transiciones_conteos_*.csv", "transiciones_matriz_*.csv"]:
        for f in TAB_DIR.glob(patron):
            f.unlink()
    for f in FIG_DIR.glob("matriz_transicion_*.png"):
        f.unlink()


def cargar_categorias():
    df = pd.read_csv(DATA_DIR / "precios_discretizados.csv", parse_dates=["fecha"])
    return df[df["metodo"] == METODO]


def pares_transicion(df_cat: pd.DataFrame) -> pd.DataFrame:
    """A partir de la serie mensual de categorías, arma los pares (categoria_t,
    categoria_t+1) SOLO para meses estrictamente consecutivos y con ambos
    datos disponibles (sin NaN)."""
    d = df_cat.sort_values("fecha").reset_index(drop=True).copy()
    d["fecha_siguiente"] = d["fecha"].shift(-1)
    d["categoria_siguiente"] = d["categoria"].shift(-1)

    mes_actual = d["fecha"].dt.year * 12 + d["fecha"].dt.month
    mes_siguiente = d["fecha_siguiente"].dt.year * 12 + d["fecha_siguiente"].dt.month
    es_consecutivo = (mes_siguiente - mes_actual) == 1
    tiene_datos = d["categoria"].notna() & d["categoria_siguiente"].notna()

    pares = d[es_consecutivo & tiene_datos][["fecha", "categoria", "categoria_siguiente"]]
    pares = pares.rename(columns={"categoria": "origen", "categoria_siguiente": "destino"})
    pares["origen"] = pares["origen"].astype(int)
    pares["destino"] = pares["destino"].astype(int)
    return pares


def estimar_matriz(pares: pd.DataFrame, n_categorias: int):
    """Devuelve (conteos, probabilidades) como DataFrames n x n, índices y
    columnas = 1..n_categorias (todas las categorías posibles, aunque
    alguna quede en cero)."""
    idx = range(1, n_categorias + 1)
    conteos = pd.crosstab(pares["origen"], pares["destino"])
    conteos = conteos.reindex(index=idx, columns=idx, fill_value=0)

    totales_fila = conteos.sum(axis=1)
    probabilidades = conteos.div(totales_fila.replace(0, np.nan), axis=0)
    return conteos, probabilidades, totales_fila


def graficar_heatmap(probabilidades: pd.DataFrame, commodity: str, config: str):
    fig, ax = plt.subplots(figsize=(7, 6))
    datos = probabilidades.fillna(0).values
    im = ax.imshow(datos, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(probabilidades.columns)))
    ax.set_xticklabels(probabilidades.columns, fontsize=7)
    ax.set_yticks(range(len(probabilidades.index)))
    ax.set_yticklabels(probabilidades.index, fontsize=7)
    ax.set_xlabel("Categoría destino (t+1)")
    ax.set_ylabel("Categoría origen (t)")
    ax.set_title(f"Matriz de transición (MLE) - {NOMBRES[commodity]}\n{config.replace('_', ' ')}, método {METODO}",
                 fontsize=11, fontweight="bold")
    fig.colorbar(im, ax=ax, label="Probabilidad")
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"matriz_transicion_{commodity}_{config}.png", bbox_inches="tight")
    plt.close(fig)


def main():
    limpiar_salidas_previas()
    df_cat_all = cargar_categorias()

    filas_resumen = []

    for commodity in COMMODITIES:
        for config in CONFIGURACIONES:
            df_cat = df_cat_all[(df_cat_all["commodity"] == commodity) & (df_cat_all["configuracion"] == config)]
            n_categorias = int(df_cat["n_categorias"].iloc[0])

            pares = pares_transicion(df_cat)
            conteos, probabilidades, totales_fila = estimar_matriz(pares, n_categorias)

            conteos.to_csv(TAB_DIR / f"transiciones_conteos_{commodity}_{config}.csv")
            probabilidades.round(4).to_csv(TAB_DIR / f"transiciones_matriz_{commodity}_{config}.csv")
            graficar_heatmap(probabilidades, commodity, config)

            categorias_sin_datos = int((totales_fila == 0).sum())
            categorias_pocas_transiciones = int(((totales_fila > 0) & (totales_fila < UMBRAL_MIN_TRANSICIONES)).sum())

            filas_resumen.append({
                "commodity": commodity,
                "configuracion": config,
                "n_categorias": n_categorias,
                "n_transiciones_totales": int(len(pares)),
                "categorias_sin_transiciones_salientes": categorias_sin_datos,
                "categorias_con_pocas_transiciones (<5)": categorias_pocas_transiciones,
                "prob_promedio_diagonal": round(np.nanmean(np.diag(probabilidades.values)), 3),
            })

    df_resumen = pd.DataFrame(filas_resumen)
    df_resumen.to_csv(TAB_DIR / "transiciones_resumen.csv", index=False)

    print("=" * 100)
    print(f"RESUMEN DE ESTIMACIÓN POR MÁXIMA VEROSIMILITUD (método de discretización: {METODO})")
    print("=" * 100)
    print(df_resumen.to_string(index=False))
    print(f"\nTablas guardadas en: {TAB_DIR}")
    print(f"Figuras guardadas en: {FIG_DIR}")
    print("\nNota: 'categorias_sin_transiciones_salientes' son filas de la matriz donde nunca se")
    print("observó esa categoría como punto de partida -> esa fila queda con probabilidades NaN")
    print("(indefinidas), no se puede estimar hacia dónde se movería el precio desde ahí.")


if __name__ == "__main__":
    main()

"""
02_analisis_descriptivo.py
----------------------------
Análisis descriptivo de las series de precios mensuales (USD/tonelada) de
nitrógeno (Urea), fósforo (Phosphat) y potasio (Potash), previo a la
discretización de precios y la estimación de las matrices de transición por
máxima verosimilitud.

Requiere haber corrido antes: 01_limpieza_datos.py

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "processed"
FIG_DIR = BASE_DIR / "outputs" / "figures"
TAB_DIR = BASE_DIR / "outputs" / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)

COMMODITIES = ["nitrogeno", "fosfat", "potash"]
COLORES = {"nitrogeno": "#2E7D32", "fosfat": "#EF6C00", "potash": "#1565C0"}
NOMBRES = {"nitrogeno": "Nitrógeno (Urea)", "fosfat": "Fósforo (Phosphat)", "potash": "Potasio (Potash)"}
UNIDAD = "USD/tonelada"

plt.rcParams.update({"figure.dpi": 130, "axes.grid": True, "grid.alpha": 0.3})


def cargar_datos():
    df = pd.read_csv(DATA_DIR / "precios_commodities_wide.csv", parse_dates=["fecha"])
    df = df.set_index("fecha").sort_index()
    return df[COMMODITIES]


def calcular_retornos(df_wide: pd.DataFrame) -> pd.DataFrame:
    """Retorno simple mensual en % : (P_t - P_{t-1}) / P_{t-1} * 100"""
    return df_wide.pct_change().dropna(how="all") * 100


def tabla_descriptiva(df: pd.DataFrame) -> pd.DataFrame:
    filas = []
    for col in df.columns:
        serie = df[col].dropna()
        filas.append({
            "commodity": col,
            "n_obs": serie.shape[0],
            "media": serie.mean(),
            "mediana": serie.median(),
            "desv_std": serie.std(),
            "cv_%": (serie.std() / serie.mean() * 100) if serie.mean() != 0 else np.nan,
            "min": serie.min(),
            "p25": serie.quantile(0.25),
            "p75": serie.quantile(0.75),
            "max": serie.max(),
            "asimetria": stats.skew(serie),
            "curtosis": stats.kurtosis(serie),  # exceso de curtosis (normal = 0)
            "jarque_bera_p": stats.jarque_bera(serie).pvalue,
        })
    return pd.DataFrame(filas).set_index("commodity").round(4)

def graficar_series_tiempo(df_wide: pd.DataFrame):
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    for ax, c in zip(axes, COMMODITIES):
        ax.plot(df_wide.index, df_wide[c], color=COLORES[c], linewidth=1.3)
        ax.set_title(NOMBRES[c], fontsize=11, fontweight="bold", loc="left")
        ax.set_ylabel(UNIDAD)
    axes[-1].set_xlabel("Fecha")
    fig.suptitle("Series de tiempo mensuales de precios de commodities", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "series_tiempo.png", bbox_inches="tight")
    plt.close(fig)


def graficar_series_tiempo_conjunta(df_wide: pd.DataFrame):
    """Como ya las 3 series comparten unidad (USD/tonelada), se pueden ver en un solo eje."""
    fig, ax = plt.subplots(figsize=(10, 5))
    for c in COMMODITIES:
        ax.plot(df_wide.index, df_wide[c], label=NOMBRES[c], color=COLORES[c], linewidth=1.3)
    ax.set_ylabel(UNIDAD)
    ax.set_xlabel("Fecha")
    ax.set_title("Precios mensuales de los 3 commodities (USD/tonelada)", fontweight="bold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "series_tiempo_conjunta.png", bbox_inches="tight")
    plt.close(fig)


def graficar_series_normalizadas(df_wide: pd.DataFrame):
    """Indexa cada serie a 100 en el primer periodo en el que las 3 tienen dato, para comparar dinámicas relativas."""
    df_common = df_wide.dropna()
    base = df_common.iloc[0]
    df_idx = df_common.div(base) * 100

    fig, ax = plt.subplots(figsize=(10, 5))
    for c in COMMODITIES:
        ax.plot(df_idx.index, df_idx[c], label=NOMBRES[c], color=COLORES[c], linewidth=1.3)
    ax.axhline(100, color="gray", linestyle="--", linewidth=0.8)
    ax.set_title(f"Precios indexados a 100 (base: {df_common.index[0].strftime('%b-%Y')})", fontweight="bold")
    ax.set_ylabel("Índice (base=100)")
    ax.set_xlabel("Fecha")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "series_normalizadas.png", bbox_inches="tight")
    plt.close(fig)


def graficar_histogramas(df: pd.DataFrame, filename: str, titulo: str, xlabel: str):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, c in zip(axes, COMMODITIES):
        serie = df[c].dropna()
        ax.hist(serie, bins=20, color=COLORES[c], alpha=0.75, edgecolor="white")
        ax.axvline(serie.mean(), color="black", linestyle="--", linewidth=1, label=f"media={serie.mean():.1f}")
        ax.set_title(NOMBRES[c], fontsize=10, fontweight="bold")
        ax.set_xlabel(xlabel)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Frecuencia")
    fig.suptitle(titulo, fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / filename, bbox_inches="tight")
    plt.close(fig)


def graficar_boxplot_retornos(df_ret: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7, 5))
    data = [df_ret[c].dropna() for c in COMMODITIES]
    bp = ax.boxplot(data, labels=[NOMBRES[c] for c in COMMODITIES], patch_artist=True)
    for patch, c in zip(bp["boxes"], COMMODITIES):
        patch.set_facecolor(COLORES[c])
        patch.set_alpha(0.6)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_ylabel("Retorno mensual (%)")
    ax.set_title("Dispersión de los retornos mensuales por commodity", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "boxplot_retornos.png", bbox_inches="tight")
    plt.close(fig)


def main():
    df_wide = cargar_datos()
    df_ret = calcular_retornos(df_wide)

    resumen_niveles = tabla_descriptiva(df_wide)
    resumen_retornos = tabla_descriptiva(df_ret)
    corr_retornos = df_ret.corr().round(3)

    resumen_niveles.to_csv(TAB_DIR / "resumen_niveles.csv")
    resumen_retornos.to_csv(TAB_DIR / "resumen_retornos.csv")
    corr_retornos.to_csv(TAB_DIR / "correlacion_retornos.csv")

    graficar_series_tiempo(df_wide)
    graficar_series_tiempo_conjunta(df_wide)
    graficar_series_normalizadas(df_wide)
    graficar_histogramas(df_wide, "histogramas_niveles.png",
                          f"Distribución de los precios (niveles, {UNIDAD})", UNIDAD)
    graficar_histogramas(df_ret, "histogramas_retornos.png",
                          "Distribución de los retornos mensuales (%)", "Retorno (%)")
    graficar_boxplot_retornos(df_ret)

    print("=" * 70)
    print(f"RESUMEN DESCRIPTIVO - NIVELES DE PRECIO ({UNIDAD})")
    print("=" * 70)
    print(resumen_niveles.to_string())
    print("\n" + "=" * 70)
    print("RESUMEN DESCRIPTIVO - RETORNOS MENSUALES (%)")
    print("=" * 70)
    print(resumen_retornos.to_string())
    print("\n" + "=" * 70)
    print("CORRELACIÓN ENTRE RETORNOS")
    print("=" * 70)
    print(corr_retornos.to_string())
    print(f"\nTablas guardadas en: {TAB_DIR}")
    print(f"Figuras guardadas en: {FIG_DIR}")


if __name__ == "__main__":
    main()

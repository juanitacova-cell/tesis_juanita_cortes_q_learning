"""
03_discretizacion.py
----------------------
Discretiza los precios mensuales de cada commodity en un número finito de
"categorias" (niveles de precio), bajo 3 configuraciones de granularidad, como
insumo para estimar después las matrices de transición por máxima
verosimilitud (conteo de transiciones observadas entre categorias consecutivos).

Requiere haber corrido antes: 01_limpieza_datos.py

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
COLORES = {"nitrogeno": "#2E7D32", "fosfat": "#EF6C00", "potash": "#1565C0"}
NOMBRES = {"nitrogeno": "Nitrógeno (Urea)", "fosfat": "Fósforo (Phosphat)", "potash": "Potasio (Potash)"}
UNIDAD = "USD/tonelada"

CONFIGURACIONES = {"5_categorias": 5, "10_categorias": 10}
METODO_PRINCIPAL = "equal_width"
METODOS = ["equal_width"]  # quantile se calcula solo para comparar

plt.rcParams.update({"figure.dpi": 130, "axes.grid": True, "grid.alpha": 0.3})


def cargar_datos():
    df = pd.read_csv(DATA_DIR / "precios_commodities_wide.csv", parse_dates=["fecha"])
    return df.set_index("fecha").sort_index()[COMMODITIES]


def discretizar(serie: pd.Series, n_categorias: int, metodo: str):
    """Devuelve (categorias, edges) donde categorias es una Serie de enteros 1..n_categorias
    alineada con `serie` (NaN se conserva como NaN), y edges son los límites
    de los intervalos usados."""
    valores = serie.dropna()
    if metodo == "equal_width":
        categorias_validas, edges = pd.cut(
            valores, bins=n_categorias, labels=False, retbins=True, include_lowest=True
        )
    
    else:
        raise ValueError(f"Método desconocido: {metodo}")

    categorias_validas = categorias_validas + 1  # 1..k en vez de 0..k-1
    categorias = pd.Series(np.nan, index=serie.index)
    categorias.loc[valores.index] = categorias_validas
    return categorias, edges


def construir_tablas(df_wide: pd.DataFrame):
    filas_long = []
    filas_resumen = []

    for commodity in COMMODITIES:
        serie = df_wide[commodity]
        for metodo in METODOS:
            for config, n_categorias in CONFIGURACIONES.items():
                categorias, edges = discretizar(serie, n_categorias, metodo)
                n_categorias_reales = int(np.nanmax(categorias)) if categorias.notna().any() else 0

                # precio promedio observado por categoria (para el precio representativo)
                promedio_por_categoria = serie.groupby(categorias).mean()

                for fecha, valor, categoria in zip(serie.index, serie.values, categorias.values):
                    if pd.isna(categoria):
                        precio_pm = np.nan
                    else:
                        e = int(categoria)
                        precio_pm = round((edges[e - 1] + edges[e]) / 2, 2)
                    filas_long.append({
                        "commodity": commodity,
                        "fecha": fecha,
                        "valor": valor,
                        "metodo": metodo,
                        "configuracion": config,
                        "n_categorias": n_categorias_reales,
                        "categoria": categoria,
                        "precio_punto_medio": precio_pm,
                    })

                conteo = categorias.value_counts(dropna=True).sort_index()
                n_total = conteo.sum()
                for e in range(1, n_categorias_reales + 1):
                    n_obs = int(conteo.get(e, 0))
                    filas_resumen.append({
                        "commodity": commodity,
                        "metodo": metodo,
                        "configuracion": config,
                        "n_categorias": n_categorias_reales,
                        "categoria": e,
                        "limite_inferior": round(edges[e - 1], 2),
                        "limite_superior": round(edges[e], 2),
                        "precio_punto_medio": round((edges[e - 1] + edges[e]) / 2, 2),
                        "precio_promedio_obs": round(promedio_por_categoria.get(float(e), np.nan), 2),
                        "n_obs": n_obs,
                        "pct_obs": round(n_obs / n_total * 100, 1) if n_total else 0.0,
                    })

    df_long = pd.DataFrame(filas_long)
    df_resumen = pd.DataFrame(filas_resumen)
    return df_long, df_resumen


def graficar_cortes_por_commodity(df_wide: pd.DataFrame):
    for commodity in COMMODITIES:
        serie = df_wide[commodity].dropna()
        n_cfg = len(CONFIGURACIONES)
        fig, axes = plt.subplots(1, n_cfg, figsize=(5*n_cfg, 4.2), sharey=True)
        axes = np.atleast_1d(axes)
        for j, (config, n_categorias) in enumerate(CONFIGURACIONES.items()):
            ax = axes[j]
            _, edges = discretizar(df_wide[commodity], n_categorias, METODO_PRINCIPAL)
            ax.hist(serie, bins=40, color=COLORES[commodity], alpha=0.6, edgecolor="white")
            for e in edges:
                ax.axvline(e, color="black", linestyle="--", linewidth=0.7)
            ax.set_title(f"{config.replace('_', ' ')} ({len(edges)-1} cortes)", fontsize=10)
            ax.set_xlabel(UNIDAD)
        axes[0].set_ylabel("Frecuencia")
        fig.suptitle(f"Cortes de discretización (equal width) - {NOMBRES[commodity]}",
                     fontsize=13, fontweight="bold")
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"discretizacion_{commodity}.png", bbox_inches="tight")
        plt.close(fig)


def graficar_series_discretizadas(df_long: pd.DataFrame, metodo=METODO_PRINCIPAL, config="5_categorias"):
    subset = df_long[(df_long["metodo"] == metodo) & (df_long["configuracion"] == config)]
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    for ax, commodity in zip(axes, COMMODITIES):
        d = subset[subset["commodity"] == commodity].sort_values("fecha")
        ax2 = ax.twinx()
        ax.plot(d["fecha"], d["valor"], color=COLORES[commodity], linewidth=1.2, label="Precio")
        ax2.step(d["fecha"], d["categoria"], color="black", linewidth=1.0, alpha=0.6, where="mid", label="Categoría")
        ax.set_title(NOMBRES[commodity], fontsize=10, fontweight="bold", loc="left")
        ax.set_ylabel(UNIDAD)
        ax2.set_ylabel("Categoría")
    axes[-1].set_xlabel("Fecha")
    fig.suptitle(f"Precio y categoria discretizado en el tiempo (equal width, {config.replace('_', ' ')})",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"series_discretizadas_{config}.png", bbox_inches="tight")
    plt.close(fig)


def limpiar_figuras_previas():
    """Borra figuras de corridas anteriores de este script para evitar que queden
    archivos sueltos de configuraciones o nombres de columnas ya obsoletos."""
    patrones = ["discretizacion_*.png", "series_discretizadas_*.png"]
    for patron in patrones:
        for f in FIG_DIR.glob(patron):
            f.unlink()


def main():
    limpiar_figuras_previas()
    df_wide = cargar_datos()
    df_long, df_resumen = construir_tablas(df_wide)

    df_long.to_csv(DATA_DIR / "precios_discretizados.csv", index=False)
    df_resumen.to_csv(TAB_DIR / "discretizacion_resumen.csv", index=False)

    graficar_cortes_por_commodity(df_wide)
    for config in CONFIGURACIONES:
        graficar_series_discretizadas(df_long, metodo=METODO_PRINCIPAL, config=config)

    # --- Diagnóstico de sparsity para el método PRINCIPAL (equal_width) ---
    resumen_ew = df_resumen[df_resumen["metodo"] == METODO_PRINCIPAL]
    print("=" * 90)
    print(f"DIAGNÓSTICO DE CATEGORÍAS VACÍAS Y SPARSE — método principal: {METODO_PRINCIPAL}")
    print(f"(recuerda: solo hay ~168 meses de datos en total por commodity)")
    print("=" * 90)
    for commodity in COMMODITIES:
        for config, n_categorias in CONFIGURACIONES.items():
            sub = resumen_ew[(resumen_ew["commodity"] == commodity) & (resumen_ew["configuracion"] == config)]
            vacios = (sub["n_obs"] == 0).sum()
            pocos = ((sub["n_obs"] > 0) & (sub["n_obs"] < 5)).sum()
            promedio_obs = sub["n_obs"].mean()
            print(f"{commodity:10s} | {config:10s} ({n_categorias:2d} categorias) -> "
                  f"promedio {promedio_obs:5.1f} obs/categoria | "
                  f"{vacios:2d} categorias VACÍOS (0 obs) | {pocos:2d} categorias con 1-4 obs")

    print(f"\nTablas guardadas en: {TAB_DIR}")
    print(f"Figuras guardadas en: {FIG_DIR}")


if __name__ == "__main__":
    main()

"""
01_limpieza_datos.py
---------------------
Lee el Excel crudo y construye un único dataset mensual limpio para los tres
commodities. Para fósforo usa la columna final acordada después de los ajustes:

  Final Phosphate Price

Esta columna conserva los precios observados hasta octubre de 2023 y, desde
noviembre de 2023, usa el precio ajustado por el índice específico del fosfato
y posteriormente deflactado con CPI-U (octubre de 2023 como mes base).

Fuentes por commodity:
  - nitrógeno (Urea): hoja "precio urea", columna "Price"
  - fósforo: hoja "precio fosfat", columna "Final Phosphate Price"
  - potasio (Potash): hoja "precio potash", columna "Price"

Salidas:
  data/processed/precios_commodities_limpio.csv
  data/processed/precios_commodities_wide.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
RAW_PATH = BASE_DIR / "data" / "raw" / "mi_tesis_datos_mensuales_precios_de_ventas.xlsx"
OUT_DIR = BASE_DIR / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

UNIDAD = "USD/tonelada"

FUENTES = {
    "nitrogeno": ("precio urea", "Price"),
    "fosfat": ("precio fosfat", "Final Phosphate Price"),
    "potash": ("precio potash", "Price"),
}


def _parse_month(value):
    """Convierte fechas mixtas (datetime o texto) al primer día del mes."""
    if isinstance(value, str):
        ts = pd.to_datetime(value, format="%b %Y", errors="coerce")
        if pd.isna(ts):
            ts = pd.to_datetime(value, errors="coerce")
    else:
        ts = pd.to_datetime(value, errors="coerce")
    return ts.replace(day=1) if pd.notna(ts) else pd.NaT


def _parse_price(value):
    """Convierte precios numéricos y textos con separadores mal exportados."""
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return np.nan
        parts = value.split(",")
        if len(parts) >= 2:
            entero = "".join(parts[:-1])
            decimal = parts[-1]
            try:
                return float(f"{entero}.{decimal}")
            except ValueError:
                pass
        try:
            return float(value)
        except ValueError:
            return np.nan
    return np.nan


def load_precio(xls_path, sheet_name, commodity, price_col):
    df = pd.read_excel(xls_path, sheet_name=sheet_name)

    if "Month" not in df.columns:
        raise ValueError(f"La hoja '{sheet_name}' no contiene la columna 'Month'.")
    if price_col not in df.columns:
        raise ValueError(
            f"La hoja '{sheet_name}' no contiene la columna '{price_col}'. "
            f"Columnas disponibles: {list(df.columns)}"
        )

    out = pd.DataFrame({
        "commodity": commodity,
        "fecha": df["Month"].apply(_parse_month),
        "valor": df[price_col].apply(_parse_price),
        "unidad_original": UNIDAD,
        "columna_fuente": price_col,
    })
    return out.dropna(subset=["fecha", "valor"])


def main():
    frames = []

    for commodity, (hoja, columna_precio) in FUENTES.items():
        d = load_precio(RAW_PATH, hoja, commodity, columna_precio)

        dups = d["fecha"].duplicated().sum()
        if dups:
            print(f"[ADVERTENCIA] {commodity}: {dups} fechas duplicadas -> se promedian.")

        d = d.groupby(
            ["commodity", "fecha", "unidad_original", "columna_fuente"],
            as_index=False,
        )["valor"].mean()
        frames.append(d)

    df_long = pd.concat(frames, ignore_index=True).sort_values(["commodity", "fecha"])

    # Calendario mensual completo. Si todavía faltara algún mes, queda NaN
    # explícitamente; no se inventan observaciones de forma silenciosa.
    completos = []
    for commodity, g in df_long.groupby("commodity"):
        g = g.set_index("fecha").sort_index()
        full_idx = pd.date_range(g.index.min(), g.index.max(), freq="MS")
        g = g.reindex(full_idx)
        g["commodity"] = commodity
        g["unidad_original"] = UNIDAD
        g["columna_fuente"] = g["columna_fuente"].ffill().bfill()
        g.index.name = "fecha"
        completos.append(g.reset_index())

    df_long = pd.concat(completos, ignore_index=True)

    faltantes = df_long["valor"].isna()
    if faltantes.any():
        print(f"[INFO] {faltantes.sum()} meses siguen sin precio.")
        print(df_long.loc[faltantes, ["commodity", "fecha"]].to_string(index=False))

    df_long.to_csv(OUT_DIR / "precios_commodities_limpio.csv", index=False)

    df_wide = df_long.pivot(index="fecha", columns="commodity", values="valor").sort_index()
    df_wide.to_csv(OUT_DIR / "precios_commodities_wide.csv")

    print("\n" + "=" * 80)
    print("COBERTURA TEMPORAL FINAL")
    print("=" * 80)
    print(df_long.groupby("commodity").agg(
        fecha_min=("fecha", "min"),
        fecha_max=("fecha", "max"),
        n_obs=("valor", lambda s: s.notna().sum()),
        n_faltantes=("valor", lambda s: s.isna().sum()),
        fuente=("columna_fuente", "first"),
    ))

    print(f"\nArchivos guardados en: {OUT_DIR}")


if __name__ == "__main__":
    main()

"""
Por: Juanita Valentina Cortés Vargas

Gráficas de la sensibilidad ampliada de Q-learning.

Uso, desde la carpeta jmarkov (después de correr sensibilidad_ampliada.py):
    py examples/qlearning_fertilizantes/graficas_sensibilidad.py

Lee outputs/tables/qlearning_independientes/historico_sensibilidad.csv y, si
existe, outputs/tables/qlearning_conjunto/historico_sensibilidad_conjunto.csv.
Guarda en outputs/figures/qlearning/, para cada insumo c (N, P, K) y para el
modelo conjunto (c = NPK):

1. sensibilidad_alpha_c.png    Sobrecosto final frente a alpha, con IC 95 %.
2. sensibilidad_epsilon_c.png  Sobrecosto final frente a epsilon, con IC 95 %.
3. historico_alpha_c.png       Histórico: sobrecosto durante el entrenamiento
                               para tres valores de alpha (media e IC 95 %).
4. cardiograma_c.png           Histórico de cada semilla con la configuración
                               base (una línea por semilla) y su media.
5. rejilla_c.png               Mapa de calor de la rejilla alpha x epsilon:
                               ventaja capturada media de cada combinación.

Solo se usan las combinaciones de la rejilla actual y las semillas 1..N_SEMILLAS
del script de sensibilidad correspondiente (todas las celdas con el mismo n).

El sobrecosto es el costo de la política aprendida menos el costo mínimo del
MDP (USD); el costo de la política es costo mínimo + sobrecosto.
"""
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sensibilidad_ampliada as sa                                     # noqa: E402
from sensibilidad_ampliada import (CONFIG_BASE, NIVEL_CONFIANZA,         # noqa: E402
                                   ARCHIVO_HISTORICO, intervalo)

SALIDA = Path(__file__).resolve().parents[3] / "outputs" / "figures" / "qlearning"
NOMBRE = {"N": "Nitrógeno (urea)", "P": "Fósforo (roca fosfórica)",
          "K": "Potasio (cloruro de potasio)", "NPK": "Modelo conjunto N-P-K"}

# Paleta (tres primeros colores categóricos validados) y tinta.
AZUL, NARANJA, AGUA = "#2a78d6", "#eb6834", "#1baf7a"
TINTA, TINTA_2, GRILLA = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.family": "serif", "font.size": 10,
    "axes.edgecolor": TINTA_2, "axes.labelcolor": TINTA, "axes.linewidth": 0.6,
    "xtick.color": TINTA_2, "ytick.color": TINTA_2,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRILLA, "grid.linewidth": 0.6,
    "legend.frameon": False, "figure.dpi": 150, "savefig.bbox": "tight",
})
USD = FuncFormatter(lambda v, _: f"{v / 1e3:,.0f}".replace(",", "."))
PASOS = FuncFormatter(lambda v, _: f"{v / 1e6:g}".replace(".", ","))


def resumen_por(datos, columna, metrica="sobrecosto_USD"):
    filas = []
    for valor, g in datos.groupby(columna):
        n, media, _, li, ls = intervalo(g[metrica])
        filas.append((valor, media, li, ls, n))
    return pd.DataFrame(filas, columns=[columna, "media", "ic_inf", "ic_sup", "n"])


def titulo(ax, texto, subtitulo):
    ax.set_title(texto, loc="left", fontsize=11, fontweight="bold", color=TINTA, pad=18)
    ax.text(0, 1.02, subtitulo, transform=ax.transAxes, fontsize=8.5, color=TINTA_2)


def grafica_hiperparametro(final, c, hp):
    b = CONFIG_BASE[c]
    otro = "epsilon" if hp == "alpha" else "alpha"
    datos = final[np.isclose(final[otro], b[otro])]
    r = resumen_por(datos, hp)
    if r.empty:
        return
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.fill_between(r[hp], r["ic_inf"], r["ic_sup"], color=AZUL, alpha=0.18, linewidth=0,
                    label=f"IC {NIVEL_CONFIANZA:.0%}")
    ax.plot(r[hp], r["media"], color=AZUL, linewidth=2, marker="o", markersize=5,
            label="Media de las semillas")
    base = r[np.isclose(r[hp], b[hp])]
    if not base.empty:
        ax.scatter(base[hp], base["media"], s=90, facecolor="white", edgecolor=AZUL,
                   linewidth=2, zorder=5, label="Configuración base")
    ax.set_xscale("log")
    ax.set_xticks(r[hp])
    ax.set_xticklabels([f"{v:g}".replace(".", ",") for v in r[hp]],
                       rotation=45 if hp == "alpha" else 0,
                       ha="right" if hp == "alpha" else "center")
    ax.minorticks_off()
    ax.axhline(0, color=TINTA_2, linewidth=0.8)
    ax.yaxis.set_major_formatter(USD)
    simbolo = "α" if hp == "alpha" else "ε"
    ax.set_xlabel(f"{simbolo} (escala logarítmica)")
    ax.set_ylabel("Sobrecosto (miles de USD)")
    n = int(r["n"].max())
    fijo = (f"ε = {b['epsilon']:g}" if hp == "alpha" else f"α = {b['alpha']:g}").replace(".", ",")
    pasos = f"{int(final['paso'].max()):,}".replace(",", ".")
    titulo(ax, f"{NOMBRE[c]}: sobrecosto según {simbolo}",
           f"{fijo}; {pasos} pasos; {n} semillas; banda: IC {NIVEL_CONFIANZA:.0%} (bootstrap)")
    ax.legend(loc="best", fontsize=8.5)
    fig.savefig(SALIDA / f"sensibilidad_{hp}_{c}.png")
    plt.close(fig)


def alphas_para_historico(disponibles, base):
    """Tres alphas: el más cercano a base/10, la base y el más cercano a base*10."""
    disponibles = sorted(disponibles)
    cerca = lambda x: min(disponibles, key=lambda v: abs(np.log(v) - np.log(x)))
    return list(dict.fromkeys([cerca(base / 10), base, cerca(base * 10)]))


def grafica_historico_alpha(h, c):
    b = CONFIG_BASE[c]
    datos = h[np.isclose(h["epsilon"], b["epsilon"])]
    if datos.empty:
        return
    elegidos = alphas_para_historico(datos["alpha"].unique(), b["alpha"])
    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    for color, a in zip([NARANJA, AZUL, AGUA], elegidos):
        r = resumen_por(datos[np.isclose(datos["alpha"], a)], "paso")
        etiqueta = f"α = {a:g}".replace(".", ",") + (" (base)" if np.isclose(a, b["alpha"]) else "")
        ax.fill_between(r["paso"], r["ic_inf"], r["ic_sup"], color=color, alpha=0.15,
                        linewidth=0)
        ax.plot(r["paso"], r["media"], color=color, linewidth=2, label=etiqueta)
    ax.axhline(0, color=TINTA_2, linewidth=0.8)
    ax.xaxis.set_major_formatter(PASOS)
    ax.yaxis.set_major_formatter(USD)
    ax.set_xlabel("Pasos de entrenamiento (millones)")
    ax.set_ylabel("Sobrecosto (miles de USD)")
    ax.set_ylim(bottom=min(0, ax.get_ylim()[0]))
    n = datos["semilla"].nunique()
    titulo(ax, f"{NOMBRE[c]}: histórico del sobrecosto",
           f"Media e IC {NIVEL_CONFIANZA:.0%} (bootstrap) entre semillas; "
           f"ε = {b['epsilon']:g}; {n} semillas".replace(".", ","))
    ax.legend(loc="best", fontsize=8.5)
    fig.savefig(SALIDA / f"historico_alpha_{c}.png")
    plt.close(fig)


def grafica_cardiograma(h, c):
    b = CONFIG_BASE[c]
    datos = h[np.isclose(h["alpha"], b["alpha"]) & np.isclose(h["epsilon"], b["epsilon"])]
    if datos.empty:
        return
    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    for k, (_, g) in enumerate(datos.groupby("semilla")):
        g = g.sort_values("paso")
        ax.plot(g["paso"], g["sobrecosto_USD"], color=TINTA_2, linewidth=0.7, alpha=0.35,
                label="Cada semilla" if k == 0 else None)
    r = resumen_por(datos, "paso")
    ax.plot(r["paso"], r["media"], color=AZUL, linewidth=2.2, label="Media")
    ax.axhline(0, color=TINTA_2, linewidth=0.8)
    ax.xaxis.set_major_formatter(PASOS)
    ax.yaxis.set_major_formatter(USD)
    ax.set_xlabel("Pasos de entrenamiento (millones)")
    ax.set_ylabel("Sobrecosto (miles de USD)")
    titulo(ax, f"{NOMBRE[c]}: histórico por semilla",
           f"Configuración base: α = {b['alpha']:g}; ε = {b['epsilon']:g}; "
           f"{datos['semilla'].nunique()} semillas".replace(".", ","))
    ax.legend(loc="best", fontsize=8.5)
    fig.savefig(SALIDA / f"cardiograma_{c}.png")
    plt.close(fig)


def grafica_rejilla(final, c):
    """Mapa de calor alpha x epsilon de la ventaja capturada media (%)."""
    tabla = (final.groupby(["alpha", "epsilon"])["ventaja_capturada_%"].mean()
             .unstack("epsilon").sort_index(ascending=False))
    if tabla.empty or tabla.shape[0] < 2 or tabla.shape[1] < 2:
        return
    fig, ax = plt.subplots(figsize=(7.6, 5.6))
    im = ax.imshow(tabla.values, cmap="Blues", vmin=0, vmax=100, aspect="auto")
    for i in range(tabla.shape[0]):
        for j in range(tabla.shape[1]):
            v = tabla.values[i, j]
            if np.isnan(v):
                continue
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7.5,
                    color="white" if v > 60 else TINTA)
    # Mejor combinación (mayor ventaja media) con borde.
    i, j = np.unravel_index(np.nanargmax(tabla.values), tabla.shape)
    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                               edgecolor=NARANJA, linewidth=2.2))
    ax.set_xticks(range(tabla.shape[1]))
    ax.set_xticklabels([f"{v:g}".replace(".", ",") for v in tabla.columns])
    ax.set_yticks(range(tabla.shape[0]))
    ax.set_yticklabels([f"{v:g}".replace(".", ",") for v in tabla.index])
    ax.set_xlabel("ε")
    ax.set_ylabel("α")
    ax.grid(False)
    for lado in ax.spines.values():
        lado.set_visible(False)
    barra = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    barra.set_label("Ventaja capturada media (%)")
    barra.outline.set_visible(False)
    n = final.groupby(["alpha", "epsilon"])["semilla"].nunique().min()
    pasos = f"{int(final['paso'].max()):,}".replace(",", ".")
    mejor = f"mejor: α = {tabla.index[i]:g}, ε = {tabla.columns[j]:g}".replace(".", ",")
    titulo(ax, f"{NOMBRE[c]}: rejilla α × ε",
           f"Ventaja capturada media (%); {pasos} pasos; {n} semillas por celda; "
           f"recuadro: {mejor}")
    fig.savefig(SALIDA / f"rejilla_{c}.png")
    plt.close(fig)


def filtrar(h, valores_alpha, valores_epsilon, n_semillas):
    """Solo combinaciones de la rejilla actual y semillas 1..n_semillas."""
    a = h["alpha"].apply(lambda v: any(np.isclose(v, valores_alpha)))
    e = h["epsilon"].apply(lambda v: any(np.isclose(v, valores_epsilon)))
    return h[a & e & (h["semilla"] <= n_semillas)]


def graficar(h, c):
    final = h[h["paso"] == h["paso"].max()]
    grafica_hiperparametro(final, c, "alpha")
    grafica_hiperparametro(final, c, "epsilon")
    grafica_historico_alpha(h, c)
    grafica_cardiograma(h, c)
    grafica_rejilla(final, c)
    print(f"{c}: gráficas guardadas")


def main():
    SALIDA.mkdir(parents=True, exist_ok=True)
    hubo_datos = False

    # Modelos independientes (sensibilidad_ampliada.py).
    if ARCHIVO_HISTORICO.exists():
        h = filtrar(pd.read_csv(ARCHIVO_HISTORICO), sa.VALORES_ALPHA,
                    sa.VALORES_EPSILON, sa.N_SEMILLAS)
        for c in ["N", "P", "K"]:
            hc = h[h["commodity"] == c]
            if not hc.empty:
                graficar(hc, c)
                hubo_datos = True

    # Modelo conjunto (sensibilidad_conjunto.py), si ya se corrió.
    import sensibilidad_conjunto as sc
    if sc.ARCHIVO_HISTORICO.exists():
        CONFIG_BASE["NPK"] = sc.CONFIG_BASE
        graficar(filtrar(pd.read_csv(sc.ARCHIVO_HISTORICO), sc.VALORES_ALPHA,
                         sc.VALORES_EPSILON, sc.N_SEMILLAS), "NPK")
        hubo_datos = True

    if not hubo_datos:
        raise FileNotFoundError("Primero corre sensibilidad_ampliada.py o sensibilidad_conjunto.py")
    print(f"Carpeta: {SALIDA}")


if __name__ == "__main__":
    main()

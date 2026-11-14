"""Figuras do artigo a partir dos CSVs de uma rodada (padrão: a final).

Uso: python -m src.modelos.figuras [prefixo]   (ex.: rodada4c). Grava PDFs em artigo/figures/.
1. Desviância média por temporada de teste, por modelo, nos dois alvos.
2. Calibração de P(> 2,5 gols) e P(> 9,5 escanteios) do Random Forest com a binomial negativa, por decis.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd

from src.modelos.analise import deviance_jogo

RAIZ = Path(__file__).resolve().parents[2]
SAIDA = RAIZ / "artigo" / "figures"
# Paleta categórica validada (dataviz, modo claro); marcadores distintos para leitura em preto e branco.
SERIES = {"poisson_ridge": ("Poisson ridge", "#2a78d6", "o"), "random_forest": ("Random Forest", "#eb6834", "s"),
          "xgboost": ("XGBoost", "#1baf7a", "^"), "media_arvores": ("Média RF e XGBoost", "#eda100", "D")}
TITULOS = {"gols": "Gols", "escanteios": "Escanteios"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b", "xtick.color": "#52514e",
                     "ytick.color": "#52514e", "axes.grid": True, "grid.color": "#e5e4df", "grid.linewidth": 0.6,
                     "legend.frameon": False, "pdf.fonttype": 42})
VIRGULA = FuncFormatter(lambda v, _: f"{v:.2f}".replace(".", ","))  # decimal com vírgula (texto em português)


def deviance_por_temporada(prev: pd.DataFrame) -> pd.DataFrame:
    prev = prev.assign(media_arvores=(prev["random_forest"] + prev["xgboost"]) / 2)
    cols = ["ingenua", *SERIES]
    return prev.groupby(["alvo", "teste"]).apply(
        lambda g: pd.Series({c: deviance_jogo(g["y"], g[c]).mean() for c in cols}), include_groups=False)


def figura_temporadas(dev: pd.DataFrame, arq: Path):
    fig, eixos = plt.subplots(1, 2, figsize=(6.3, 2.8), sharex=True)
    for ax, alvo in zip(eixos, TITULOS):
        d = dev.loc[alvo]
        ax.plot(d.index, d["ingenua"], color="#52514e", ls="--", lw=1.5, label="Média da liga")
        for c, (rotulo, cor, m) in SERIES.items():
            ax.plot(d.index, d[c], color=cor, marker=m, ms=4, lw=1.5, label=rotulo)
        ax.set_title(TITULOS[alvo], fontsize=9)
        ax.set_xticks(d.index)
        ax.yaxis.set_major_formatter(VIRGULA)
        ax.set_xlabel("Temporada de teste")
    eixos[0].set_ylabel("Desviância média por partida")
    h, r = eixos[0].get_legend_handles_labels()
    fig.legend(h, r, loc="lower center", ncol=5, fontsize=7.5, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(arq)
    plt.close(fig)


def calibracao(prob: pd.DataFrame, col: str, n=10) -> pd.DataFrame:
    """Média prevista e frequência observada por decil da probabilidade prevista."""
    d = prob[[col, "acima"]].copy()
    d["faixa"] = pd.qcut(d[col], n, labels=False, duplicates="drop")
    return d.groupby("faixa").agg(prevista=(col, "mean"), observada=("acima", "mean"), n=("acima", "size"))


def figura_calibracao(prob: pd.DataFrame, arq: Path, col="random_forest_nb"):
    fig, eixos = plt.subplots(1, 2, figsize=(6.3, 3.0))
    for ax, (alvo, linha) in zip(eixos, (("gols", "2,5"), ("escanteios", "9,5"))):
        p = prob[prob["alvo"] == alvo]
        c = calibracao(p, col)
        lo = min(c["prevista"].min(), c["observada"].min()) - 0.03
        hi = max(c["prevista"].max(), c["observada"].max()) + 0.03
        ax.plot([lo, hi], [lo, hi], color="#52514e", ls="--", lw=1, label="Calibração perfeita")
        ax.axhline(p["frequencia"].mean(), color="#a3a29b", ls=":", lw=1.2, label="Frequência no treino")
        ax.plot(c["prevista"], c["observada"], color=SERIES["random_forest"][1], marker="s", ms=4, lw=1.5,
                label="Random Forest")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.xaxis.set_major_formatter(VIRGULA)
        ax.yaxis.set_major_formatter(VIRGULA)
        ax.set_title(f"Mais de {linha} {TITULOS[alvo].lower()}", fontsize=9)
        ax.set_xlabel("Probabilidade prevista (média do decil)")
    eixos[0].set_ylabel("Frequência observada")
    h, r = eixos[0].get_legend_handles_labels()
    fig.legend(h, r, loc="lower center", ncol=3, fontsize=7.5, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(arq)
    plt.close(fig)


def main(prefixo="rodada4c"):
    base = RAIZ / "results" / "modelos"
    prev = pd.read_csv(base / f"{prefixo}_previsoes.csv")
    prob = pd.read_csv(base / f"{prefixo}_probabilidades.csv")
    SAIDA.mkdir(parents=True, exist_ok=True)
    dev = deviance_por_temporada(prev)
    print(dev.round(4).to_string())
    figura_temporadas(dev, SAIDA / "deviance_temporadas.pdf")
    for alvo in TITULOS:
        print(alvo, calibracao(prob[prob["alvo"] == alvo], "random_forest_nb").round(3).to_string(), sep="\n")
    figura_calibracao(prob, SAIDA / "calibracao.pdf")


if __name__ == "__main__":
    main(*sys.argv[1:])

"""Referências mais fortes que a média da liga (revisão de 14/11/2026, achado 1).

Análise de robustez posterior aos resultados da 4c: o peso de 100 jogos foi fixado sem ajuste e não foi
planejado antes de ver o teste. Não roda modelo; lê `<etapa>_conjunto_previsoes.csv` e a canônica (SELECT).

- temporada anterior: média do total da temporada T−1, constante em T;
- corrente: média do total da temporada só com jogos de data estritamente anterior (nunca a mesma data),
  encolhida para a média da temporada anterior com peso `PESO` jogos: (n·média + PESO·anterior)/(n + PESO).
Compara cada modelo com cada referência no mesmo bootstrap por blocos de rodada de `analise.py`.

Uso: python -m src.modelos.referencias   (grava results/modelos/<etapa>_referencias.csv)
"""
import numpy as np
import pandas as pd

from src.modelos.analise import comparacao
from src.modelos.tabela import ALVOS
from src.modelos.walkforward import ETAPA, SAIDA

PESO = 100
MODELOS = {"Média da liga": "ingenua", "Poisson ridge": "poisson_ridge", "Random Forest": "random_forest",
           "XGBoost": "xgboost", "Média RF+XGB": "media_arvores"}


def media_corrente(data, y, anterior: float, peso: float = PESO) -> np.ndarray:
    """Para cada jogo, (n·média + peso·anterior)/(n + peso) com n e média só dos jogos de data < a sua.
    `data` e `y` são de uma única temporada, em qualquer ordem."""
    data, y = np.asarray(data), np.asarray(y, float)
    ordem = np.argsort(data, kind="stable")
    d, soma = data[ordem], np.r_[0.0, np.cumsum(y[ordem])]
    n = np.searchsorted(d, data, side="left")  # jogos com data estritamente anterior
    return (soma[n] + peso * anterior) / (n + peso)


def referencias(jogos: pd.DataFrame, alvo: str, t: int, peso: float = PESO) -> pd.DataFrame:
    """Referências (alinhadas a `jogos[jogos.temporada == t]`) do alvo total na temporada t."""
    y = jogos[ALVOS[alvo][0]] + jogos[ALVOS[alvo][1]]
    atual, anterior = jogos["temporada"] == t, y[jogos["temporada"] == t - 1].mean()
    return pd.DataFrame({"ref_anterior": np.full(int(atual.sum()), anterior),
                         "ref_corrente": media_corrente(jogos.loc[atual, "data"], y[atual], anterior, peso)})


def analisar(prev: pd.DataFrame, jogos: pd.DataFrame) -> pd.DataFrame:
    partes = []
    for (alvo, t), g in prev.groupby(["alvo", "teste"], sort=False):
        j = jogos[jogos["temporada"] == t]
        assert len(j) == len(g) and ((j[ALVOS[alvo][0]] + j[ALVOS[alvo][1]]).to_numpy() == g["y"].to_numpy()).all()
        partes.append(g.reset_index(drop=True).join(referencias(jogos, alvo, t)))
    juntas = pd.concat(partes)
    saida = []
    for ref, nome in (("ref_anterior", "média da temporada anterior"), ("ref_corrente", f"corrente (peso {PESO})")):
        c = comparacao(juntas, MODELOS, ref=ref).rename(columns={f"dif_{ref}": "dif", "ic_inf": "ic_inf", "ic_sup": "ic_sup"})
        base = comparacao(juntas.assign(zero=juntas[ref]), {nome: ref}, ref="ingenua")  # a referência contra a média da liga
        saida.append(c.assign(referencia=nome))
        saida.append(base.rename(columns={"dif_ingenua": "dif"}).assign(referencia=nome + " (contra a média da liga)"))
    return pd.concat(saida)[["alvo", "referencia", "modelo", "deviance", "dif", "ic_inf", "ic_sup"]]


def main():
    from src.features.historico import carregar_canonica

    prev = pd.read_csv(SAIDA / f"{ETAPA}_conjunto_previsoes.csv")
    res = analisar(prev, carregar_canonica())
    res.to_csv(SAIDA / f"{ETAPA}_referencias.csv", index=False)
    print(res.round(4).to_string(index=False))


if __name__ == "__main__":
    main()

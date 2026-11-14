"""Combinação das previsões da rodada 4 (decisão de 09/11/2026).

- conjunto: média ponderada das médias previstas (ingênua, ridge, RF, XGBoost), pesos numa grade do
  simplex com passo 0,1, escolhidos em cada dobra pela deviance média das validações 2018..T−1. Cada
  validação v é prevista pelo modelo escolhido para T reajustado em < v (como o α da NB na analise.py);
  a temporada de teste não entra. Ressalva: os hiperparâmetros de T foram escolhidos com essas mesmas
  validações, então as previsões de v são um pouco otimistas para os modelos ajustados.
- media_arvores: média simples de RF e XGBoost, fixada antes de olhar qualquer resultado (sem escolha).

Uso: python -m src.modelos.conjunto   (lê <ETAPA>_{previsoes,escolhas}.csv)
"""
import itertools

import numpy as np
import pandas as pd

from src.modelos.analise import MODELOS_R4, comparacao, deviance_jogo
from src.modelos.rodada2 import validacoes
from src.modelos.tabela import ALVOS
from src.modelos.walkforward import ETAPA, SAIDA

COMPONENTES = ("ingenua", *MODELOS_R4)


def grade_pesos(n: int, passo: float = 0.1) -> list[np.ndarray]:
    m = round(1 / passo)
    return [np.array(c) / m for c in itertools.product(range(m + 1), repeat=n) if sum(c) == m]


def melhores_pesos(val: list[tuple[np.ndarray, np.ndarray]], passo: float = 0.1) -> np.ndarray:
    """val: uma entrada por temporada de validação, (y, matriz jogos × componentes das médias previstas)."""
    return min(grade_pesos(val[0][1].shape[1], passo),
               key=lambda w: np.mean([deviance_jogo(y, M @ w).mean() for y, M in val]))


def validacao_fora(jogos, bases, esc_t: pd.DataFrame, alvo: str, t: int, ajustar_final):
    y = jogos[ALVOS[alvo][0]] + jogos[ALVOS[alvo][1]]
    temp = jogos["temporada"]
    val = []
    for v in validacoes(t):
        cols = [np.full(int((temp == v).sum()), y[temp < v].mean())]
        for m in MODELOS_R4:
            e = esc_t[esc_t.modelo == m].iloc[0].to_dict()
            prever, X, te = ajustar_final(m, {**e, "teste": v}, bases, jogos, alvo)[:3]
            assert (te == (temp == v)).all()
            cols.append(prever(X[te]))
        val.append((y[temp == v].to_numpy(float), np.column_stack(cols)))
    return val


def main():
    from src.modelos.rodada4 import ajustar_final, carregar_bases

    jogos, bases = carregar_bases()
    prev = pd.read_csv(SAIDA / f"{ETAPA}_previsoes.csv")
    esc = pd.read_csv(SAIDA / f"{ETAPA}_escolhas.csv")
    pesos, partes = [], []
    for (alvo, t), g in prev.groupby(["alvo", "teste"], sort=False):
        val = validacao_fora(jogos, bases, esc[(esc.alvo == alvo) & (esc.teste == t)], alvo, int(t), ajustar_final)
        w = melhores_pesos(val)
        nota = np.mean([deviance_jogo(y, M @ w).mean() for y, M in val])
        pesos.append({"alvo": alvo, "teste": t, **dict(zip(COMPONENTES, w)), "deviance_validacao": nota})
        partes.append(g.assign(conjunto=g[list(COMPONENTES)].to_numpy() @ w,
                               media_arvores=(g["random_forest"] + g["xgboost"]) / 2))
        print(alvo, t, dict(zip(COMPONENTES, w.round(1))), flush=True)
    pesos, juntas = pd.DataFrame(pesos), pd.concat(partes)
    rotulos = {"Poisson ridge": "poisson_ridge", "Random Forest": "random_forest", "XGBoost": "xgboost",
               "Média RF+XGB": "media_arvores", "Conjunto": "conjunto"}
    comp = comparacao(juntas, rotulos).merge(
        comparacao(juntas, rotulos, ref="random_forest")[["alvo", "modelo", "dif_random_forest", "ic_inf", "ic_sup"]]
        .rename(columns={"ic_inf": "ic_inf_rf", "ic_sup": "ic_sup_rf"}), on=["alvo", "modelo"])
    pesos.to_csv(SAIDA / f"{ETAPA}_conjunto_pesos.csv", index=False)
    juntas.to_csv(SAIDA / f"{ETAPA}_conjunto_previsoes.csv", index=False)
    comp.to_csv(SAIDA / f"{ETAPA}_conjunto_comparacao.csv", index=False)
    print(comp.round(4).to_string(index=False))


if __name__ == "__main__":
    main()

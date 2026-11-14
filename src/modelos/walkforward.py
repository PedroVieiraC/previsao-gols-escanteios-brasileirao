"""Rodada 1 dos modelos: referência ingênua e Poisson, com walk-forward por temporada (decisão de 08/11/2026).

Dobra T (2019 a 2024): treino = temporadas < T; teste = T. Escolhas só com o treino: ajusta em < T−1 e
valida em T−1 (Poisson deviance) a grade de k_enc × forma recente × fração dos pi-ratings × fração do GAP
(× versão da formação, nos escanteios); a melhor é reajustada no treino inteiro e prevista no teste.
Linhas com feature nula (início de 2017) saem do ajuste. Dispersão de Pearson (χ²/gl) no treino: acima de
DISPERSAO_LIMITE, a binomial negativa (NB2, statsmodels) também é ajustada com a mesma configuração.
Referência ingênua: média do alvo nas temporadas de treino.

Uso: python -m src.modelos.walkforward   (grava results/modelos/rodada1_metricas.csv)
"""
import itertools
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error, mean_poisson_deviance, mean_squared_error

from src.features.historico import carregar_canonica
from src.modelos.tabela import ALVOS, FORMACOES, FORMAS, FRACOES, calcular_base, enxuta, ler_tabela

KS = (0, 2, 5, 10)
TESTES = range(2019, 2025)
DISPERSAO_LIMITE = 1.1
SAIDA = Path(__file__).resolve().parents[2] / "results" / "modelos"
# Etapa da rodada 4 que o código reproduz: "rodada4c" (final, 240 combinações do XGBoost) ou "rodada4d"
# (sensibilidade, 480). Define a grade e o prefixo dos CSVs gravados e lidos; trocar com TCC_ETAPA=rodada4d.
ETAPA = os.environ.get("TCC_ETAPA", "rodada4c")


def dobra(temporadas: pd.Series, teste: int) -> dict:
    """Máscaras da dobra: treino (< teste), ajuste interno (< teste − 1), validação interna (teste − 1)."""
    return {"treino": temporadas < teste, "ajuste": temporadas < teste - 1,
            "validacao": temporadas == teste - 1, "teste": temporadas == teste}


def _poisson(X: pd.DataFrame, y: pd.Series):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sm.GLM(y, sm.add_constant(X, has_constant="add"), family=sm.families.Poisson()).fit()


def _binomial_negativa(X: pd.DataFrame, y: pd.Series, poisson):
    """NB2 por GLM, com α pela regressão auxiliar de Cameron e Trivedi: ((y − μ)² − y)/μ = α·μ + erro."""
    mu = poisson.fittedvalues
    alpha = max(float(np.sum(((y - mu) ** 2 - y) / mu * mu) / np.sum(mu ** 2)), 1e-8)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fam = sm.families.NegativeBinomial(alpha=alpha)
        return sm.GLM(y, sm.add_constant(X, has_constant="add"), family=fam).fit(), alpha


def _sem_constantes(X: pd.DataFrame, linhas: pd.Series) -> list[str]:
    """Colunas que variam nas linhas de ajuste (ex.: família D ausente no treino)."""
    return [c for c in X.columns if X.loc[linhas, c].nunique() > 1]


def _prever(modelo, X: pd.DataFrame) -> np.ndarray:
    return np.asarray(modelo.predict(sm.add_constant(X, has_constant="add")))


def metricas(y, p) -> dict:
    return {"mae": mean_absolute_error(y, p), "rmse": mean_squared_error(y, p) ** 0.5,
            "deviance": mean_poisson_deviance(y, p)}


def configuracoes(alvo: str):
    formacoes = FORMACOES if alvo == "escanteios" else ("sem",)
    return [dict(k=k, forma=fo, f_pi=fp, f_gap=fg, formacao=fm)
            for k, fo, fp, fg, fm in itertools.product(KS, FORMAS, FRACOES, FRACOES, formacoes)]


def avaliar(jogos: pd.DataFrame, bases: dict, alvo: str, testes=TESTES) -> tuple[pd.DataFrame, pd.DataFrame]:
    y = jogos[ALVOS[alvo][0]] + jogos[ALVOS[alvo][1]]
    tabelas = {}

    def tabela(c):
        chave = tuple(c.values())
        if chave not in tabelas:
            tabelas[chave] = enxuta(bases[c["k"]], alvo, c["forma"], c["f_pi"], c["f_gap"], c["formacao"])
        return tabelas[chave]

    linhas, escolhas = [], []
    for t in testes:
        m = dobra(jogos["temporada"], t)
        melhor, melhor_dev = None, np.inf
        for c in configuracoes(alvo):
            X = tabela(c)
            ok = X.notna().all(axis=1)
            aj, va = m["ajuste"] & ok, m["validacao"] & ok
            cols = _sem_constantes(X, aj)
            dev = mean_poisson_deviance(y[va], _prever(_poisson(X.loc[aj, cols], y[aj]), X.loc[va, cols]))
            if dev < melhor_dev:
                melhor, melhor_dev = c, dev
        X = tabela(melhor)
        ok = X.notna().all(axis=1)
        tr, te = m["treino"] & ok, m["teste"]
        assert ok[te].all(), f"feature nula no teste {t}"
        X = X[_sem_constantes(X, tr)]
        modelo = _poisson(X[tr], y[tr])
        dispersao = modelo.pearson_chi2 / modelo.df_resid
        previsoes = {"ingenua": np.full(int(te.sum()), y[m["treino"]].mean()), "poisson": _prever(modelo, X[te])}
        alpha = np.nan
        if dispersao > DISPERSAO_LIMITE:
            nb, alpha = _binomial_negativa(X[tr], y[tr], modelo)
            previsoes["binomial_negativa"] = _prever(nb, X[te])
        escolhas.append({"alvo": alvo, "teste": t, **melhor, "deviance_validacao": melhor_dev,
                         "dispersao": dispersao, "alpha_nb": alpha, "n_treino": int(tr.sum()), "n_features": X.shape[1]})
        for nome, p in previsoes.items():
            linhas.append({"alvo": alvo, "teste": t, "modelo": nome, **metricas(y[te], p)})
        print(f"{alvo} {t}: {melhor} dispersão {dispersao:.2f}", flush=True)
    return pd.DataFrame(linhas), pd.DataFrame(escolhas)


def main():
    jogos = carregar_canonica()
    copa, depara = ler_tabela("norm_copa_brasil"), ler_tabela("tb_tecnicos_depara")
    bases = {k: calcular_base(jogos, copa, depara, k) for k in KS}
    SAIDA.mkdir(parents=True, exist_ok=True)
    resultados, escolhas = zip(*(avaliar(jogos, bases, alvo) for alvo in ALVOS))
    res, esc = pd.concat(resultados), pd.concat(escolhas)
    res.to_csv(SAIDA / "rodada1_metricas.csv", index=False)
    esc.to_csv(SAIDA / "rodada1_escolhas.csv", index=False)
    print(res.groupby(["alvo", "modelo"])[["mae", "rmse", "deviance"]].mean().round(4).to_string())


if __name__ == "__main__":
    main()

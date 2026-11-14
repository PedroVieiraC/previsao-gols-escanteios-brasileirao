"""Rodada 2: seleção estável, Poisson regularizado e teste de significância (decisão de 08/11/2026).

Mudanças em relação à rodada 1 (walkforward.py):
- Seleção pela deviance média de várias temporadas de validação: para o teste T, cada temporada v de 2018
  a T−1 é prevista por um modelo ajustado em < v (walk-forward interno). Nada de T entra na escolha.
- k_enc fixo em 5 (padrão documentado, não escolhido pelos resultados da rodada 1, o que vazaria dados
  das dobras novas para as antigas); a grade fica com forma × frações (× formação, nos escanteios).
- Poisson ridge (sklearn PoissonRegressor, features padronizadas com média e desvio do ajuste), com a
  penalidade escolhida na mesma seleção. O Poisson sem penalidade é refeito com a seleção estável.
- Significância: bootstrap pareado da diferença de deviance por jogo contra a referência ingênua, nas
  temporadas de teste juntas (2.000 reamostragens).

Uso: python -m src.modelos.rodada2   (grava results/modelos/rodada2_*.csv)
"""
import itertools

import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_poisson_deviance

from src.features.historico import K_PADRAO, carregar_canonica
from src.modelos.tabela import ALVOS, FORMACOES, FORMAS, FRACOES, calcular_base, enxuta, ler_tabela
from src.modelos.walkforward import SAIDA, TESTES, _poisson, _prever, _sem_constantes, metricas

# A grade foi ampliada duas vezes porque a escolha caía no limite (1,0 e depois 30).
PENALIDADES = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0)
PRIMEIRA_VALIDACAO = 2018


def configuracoes(alvo: str):
    formacoes = FORMACOES if alvo == "escanteios" else ("sem",)
    return [dict(forma=fo, f_pi=fp, f_gap=fg, formacao=fm)
            for fo, fp, fg, fm in itertools.product(FORMAS, FRACOES, FRACOES, formacoes)]


def validacoes(teste: int) -> list[int]:
    return list(range(PRIMEIRA_VALIDACAO, teste))


def _ridge(X, y, penalidade):
    media, desvio = X.mean(), X.std().replace(0, 1)
    m = PoissonRegressor(alpha=penalidade, max_iter=1000).fit((X - media) / desvio, y)
    return lambda Z: m.predict((Z - media) / desvio)


def _glm(X, y, _=None):
    modelo = _poisson(X, y)
    return lambda Z: _prever(modelo, Z)


AJUSTES = {"poisson": (_glm, (None,)), "poisson_ridge": (_ridge, PENALIDADES)}


def bootstrap_diferenca(y, p_modelo, p_ref, n=2000, semente=0) -> tuple[float, float, float]:
    """Média e IC95% de (deviance do modelo − deviance da referência) por jogo; negativo = modelo melhor."""
    y, p_modelo, p_ref = map(np.asarray, (y, p_modelo, p_ref))

    def dev(p):
        return 2 * (np.where(y > 0, y * np.log(np.where(y > 0, y, 1) / p), 0) - (y - p))

    d = dev(p_modelo) - dev(p_ref)
    rng = np.random.default_rng(semente)
    medias = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)]
    return float(d.mean()), float(np.percentile(medias, 2.5)), float(np.percentile(medias, 97.5))


def avaliar(jogos, base, alvo, testes=TESTES):
    y = jogos[ALVOS[alvo][0]] + jogos[ALVOS[alvo][1]]
    temp = jogos["temporada"]
    tabelas = {tuple(c.values()): enxuta(base, alvo, **c) for c in configuracoes(alvo)}
    # Deviance de cada (configuração, modelo, penalidade) em cada temporada de validação, calculada uma vez.
    cache = {}
    for (chave, X), (nome, (ajuste, pens)), v in itertools.product(
            tabelas.items(), AJUSTES.items(), validacoes(max(testes))):
        ok = X.notna().all(axis=1)
        aj, va = (temp < v) & ok, temp == v
        cols = _sem_constantes(X, aj)
        for pen in pens:
            prever = ajuste(X.loc[aj, cols], y[aj], pen)
            cache[(chave, nome, pen, v)] = mean_poisson_deviance(y[va], prever(X.loc[va, cols]))
    linhas, escolhas, previsoes = [], [], []
    for t in testes:
        te = temp == t
        p_t = {"ingenua": np.full(int(te.sum()), y[temp < t].mean())}
        for nome, (ajuste, pens) in AJUSTES.items():
            notas = {(chave, pen): np.mean([cache[(chave, nome, pen, v)] for v in validacoes(t)])
                     for chave in tabelas for pen in pens}
            (chave, pen), nota = min(notas.items(), key=lambda kv: kv[1])
            X = tabelas[chave]
            ok = X.notna().all(axis=1)
            tr = (temp < t) & ok
            cols = _sem_constantes(X, tr)
            p_t[nome] = ajuste(X.loc[tr, cols], y[tr], pen)(X.loc[te, cols])
            escolhas.append({"alvo": alvo, "teste": t, "modelo": nome,
                             **dict(zip(("forma", "f_pi", "f_gap", "formacao"), chave)),
                             "penalidade": pen, "deviance_validacao": nota, "n_validacoes": len(validacoes(t))})
        for nome, p in p_t.items():
            linhas.append({"alvo": alvo, "teste": t, "modelo": nome, **metricas(y[te], p)})
        previsoes.append(pd.DataFrame({"alvo": alvo, "teste": t, "y": y[te].to_numpy(), **p_t}))
        print(f"{alvo} {t}: " + ", ".join(f"{e['modelo']} {e['forma']} pi{e['f_pi']} gap{e['f_gap']} "
                                           f"{e['formacao']} pen={e['penalidade']}"
                                           for e in escolhas[-len(AJUSTES):]), flush=True)
    prev = pd.concat(previsoes)
    sig = [{"alvo": alvo, "modelo": nome, **dict(zip(("dif_deviance", "ic_inf", "ic_sup"),
                                                     bootstrap_diferenca(prev.y, prev[nome], prev.ingenua)))}
           for nome in AJUSTES]
    return pd.DataFrame(linhas), pd.DataFrame(escolhas), pd.DataFrame(sig), prev


def main():
    jogos = carregar_canonica()
    base = calcular_base(jogos, ler_tabela("norm_copa_brasil"), ler_tabela("tb_tecnicos_depara"), K_PADRAO)
    SAIDA.mkdir(parents=True, exist_ok=True)
    partes = [avaliar(jogos, base, alvo) for alvo in ALVOS]
    for i, nome in enumerate(("metricas", "escolhas", "significancia", "previsoes")):
        pd.concat(p[i] for p in partes).to_csv(SAIDA / f"rodada2_{nome}.csv", index=False)
    res = pd.concat(p[0] for p in partes)
    print(res.groupby(["alvo", "modelo"])[["mae", "rmse", "deviance"]].mean().round(4).to_string())
    print(pd.concat(p[2] for p in partes).round(4).to_string(index=False))


if __name__ == "__main__":
    main()

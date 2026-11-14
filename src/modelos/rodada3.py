"""Rodada 3: Random Forest e XGBoost com a seleção estável da rodada 2 (decisão de 08/11/2026).

- Features: em cada dobra, a configuração (forma, frações, formação) que o Poisson ridge escolheu na rodada 2,
  que usou só o treino da dobra (results/modelos/rodada2_escolhas.csv). Assim a grade das árvores fica só
  nos hiperparâmetros e roda em minutos no computador local.
- XGBoost recebe também promovido e tecnico_jogos (NULL documentado; o XGBoost trata ausência). O RF usa
  o mesmo conjunto do Poisson (sem colunas com NULL).
- Hiperparâmetros escolhidos pela deviance média das temporadas de validação de 2018 a T−1, como na rodada 2.
- Significância: bootstrap pareado contra a média da liga e contra o Poisson ridge da rodada 2.

Uso: python -m src.modelos.rodada3   (grava results/modelos/rodada3_*.csv)
"""
import itertools

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_poisson_deviance
from xgboost import XGBRegressor

from src.features.historico import K_PADRAO, carregar_canonica
from src.modelos.rodada2 import bootstrap_diferenca, validacoes
from src.modelos.tabela import ALVOS, calcular_base, enxuta, ler_tabela
from src.modelos.walkforward import SAIDA, TESTES, metricas

GRADES = {
    "random_forest": [dict(min_samples_leaf=l, max_features=f) for l, f in itertools.product((10, 30, 100), (0.33, 1.0))],
    "xgboost": [dict(max_depth=d, min_child_weight=w) for d, w in itertools.product((2, 3, 4), (5, 20))],
}


def _modelo(nome: str, hp: dict):
    if nome == "random_forest":
        return RandomForestRegressor(n_estimators=300, criterion="poisson", n_jobs=-1, random_state=0, **hp)
    return XGBRegressor(objective="count:poisson", n_estimators=300, learning_rate=0.03, subsample=0.8,
                        colsample_bytree=0.8, n_jobs=-1, random_state=0, **hp)


def _ajustar_prever(nome, hp, X, y, linhas_aj, linhas_prev):
    m = _modelo(nome, hp).fit(X[linhas_aj], y[linhas_aj])
    return np.clip(m.predict(X[linhas_prev]), 1e-6, None)


def avaliar(jogos, base, alvo, escolhas_r2, testes=TESTES):
    y = jogos[ALVOS[alvo][0]] + jogos[ALVOS[alvo][1]]
    temp = jogos["temporada"]
    linhas, escolhas, previsoes = [], [], []
    for t in testes:
        c = escolhas_r2.query("alvo == @alvo and teste == @t and modelo == 'poisson_ridge'").iloc[0]
        cfg = dict(forma=c.forma, f_pi=int(c.f_pi), f_gap=int(c.f_gap), formacao=c.formacao)
        base_ok = enxuta(base, alvo, **cfg).notna().all(axis=1)  # mesmas linhas do Poisson
        te = temp == t
        p_t = {}
        for nome, grade in GRADES.items():
            X = enxuta(base, alvo, **cfg, com_nulos=(nome == "xgboost"))
            X = X if nome == "xgboost" else X.astype(float)
            notas = []
            for hp in grade:
                devs = [mean_poisson_deviance(y[temp == v], _ajustar_prever(nome, hp, X, y, (temp < v) & base_ok, temp == v))
                        for v in validacoes(t)]
                notas.append(np.mean(devs))
            hp = grade[int(np.argmin(notas))]
            p_t[nome] = _ajustar_prever(nome, hp, X, y, (temp < t) & base_ok, te)
            escolhas.append({"alvo": alvo, "teste": t, "modelo": nome, **cfg, **hp, "deviance_validacao": min(notas)})
        for nome, p in p_t.items():
            linhas.append({"alvo": alvo, "teste": t, "modelo": nome, **metricas(y[te], p)})
        previsoes.append(pd.DataFrame({"alvo": alvo, "teste": t, **p_t}))
        print(f"{alvo} {t}: " + "; ".join(f"{e['modelo']} {e}" for e in escolhas[-2:]), flush=True)
    return pd.DataFrame(linhas), pd.DataFrame(escolhas), pd.concat(previsoes)


def main():
    jogos = carregar_canonica()
    base = calcular_base(jogos, ler_tabela("norm_copa_brasil"), ler_tabela("tb_tecnicos_depara"), K_PADRAO)
    esc_r2 = pd.read_csv(SAIDA / "rodada2_escolhas.csv")
    prev_r2 = pd.read_csv(SAIDA / "rodada2_previsoes.csv")
    partes = [avaliar(jogos, base, alvo, esc_r2) for alvo in ALVOS]
    res, esc, prev = (pd.concat(p[i] for p in partes) for i in range(3))
    assert (prev_r2[["alvo", "teste"]].to_numpy() == prev[["alvo", "teste"]].to_numpy()).all(), "previsões desalinhadas"
    prev = pd.concat([prev_r2.reset_index(drop=True), prev.drop(columns=["alvo", "teste"]).reset_index(drop=True)], axis=1)
    sig = [{"alvo": a, "modelo": m, "contra": ref,
            **dict(zip(("dif_deviance", "ic_inf", "ic_sup"), bootstrap_diferenca(g.y, g[m], g[ref])))}
           for a, g in prev.groupby("alvo") for m in GRADES for ref in ("ingenua", "poisson_ridge")]
    res.to_csv(SAIDA / "rodada3_metricas.csv", index=False)
    esc.to_csv(SAIDA / "rodada3_escolhas.csv", index=False)
    pd.DataFrame(sig).to_csv(SAIDA / "rodada3_significancia.csv", index=False)
    prev.to_csv(SAIDA / "rodada3_previsoes.csv", index=False)
    todos = pd.concat([pd.read_csv(SAIDA / "rodada2_metricas.csv"), res])
    print(todos.groupby(["alvo", "modelo"])[["mae", "rmse", "deviance"]].mean().round(4).to_string())
    print(pd.DataFrame(sig).round(4).to_string(index=False))


if __name__ == "__main__":
    main()

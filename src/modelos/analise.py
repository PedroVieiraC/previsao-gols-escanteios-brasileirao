"""Análises sobre as previsões das rodadas 2 a 4 (decisões de 08/11/2026, itens 3, 4 e 7).

- Bootstrap por blocos de rodada: reamostra (temporada, rodada) com reposição, porque os jogos de uma
  mesma rodada não são independentes (mesmo calendário, mesma fase). Vale para todas as comparações,
  inclusive as refeitas das rodadas 2 e 3 (`comparacao`).
- Importância por permutação no teste de cada dobra: aumento da deviance quando uma coluna (ou um grupo
  de colunas, como a formação) é embaralhada dentro do teste; o modelo não é reajustado. No XGBoost também
  a contribuição média absoluta de TreeSHAP (pred_contribs do próprio xgboost, sem dependência nova).
- Previsão probabilística: P(total > linha) pela Poisson e pela binomial negativa com a média de cada
  modelo; α da binomial negativa estimado nas previsões fora da amostra das validações da dobra (< teste).
  Referência: frequência de jogos acima da linha nas temporadas de treino. Log loss e Brier.

Uso: python -m src.modelos.analise   (lê results/modelos/rodada3_*.csv e <ETAPA>_*.csv e grava os resultados abaixo)
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy import stats

from src.modelos.rodada2 import validacoes
from src.modelos.tabela import ALVOS
from src.modelos.walkforward import ETAPA, SAIDA

LINHAS = {"gols": 2.5, "escanteios": 9.5}
MODELOS_R4 = ("poisson_ridge", "random_forest", "xgboost")


def deviance_jogo(y, p) -> np.ndarray:
    y, p = np.asarray(y, float), np.asarray(p, float)
    return 2 * (np.where(y > 0, y * np.log(np.where(y > 0, y, 1) / p), 0) - (y - p))


def bootstrap_blocos(d, blocos, n=2000, semente=0) -> tuple[float, float, float]:
    """Média de d por jogo e IC95% reamostrando blocos inteiros (média ponderada pelo tamanho do bloco)."""
    g = pd.DataFrame({"d": np.asarray(d, float), "b": pd.Series(list(blocos)).astype(str).to_numpy()}
                     ).groupby("b")["d"].agg(["sum", "count"])
    soma, cont = g["sum"].to_numpy(), g["count"].to_numpy()
    idx = np.random.default_rng(semente).integers(0, len(g), (n, len(g)))
    medias = soma[idx].sum(axis=1) / cont[idx].sum(axis=1)
    return float(soma.sum() / cont.sum()), float(np.percentile(medias, 2.5)), float(np.percentile(medias, 97.5))


def comparacao(prev: pd.DataFrame, colunas: dict, ref="ingenua") -> pd.DataFrame:
    """Uma linha por (alvo, modelo): deviance, MAE, RMSE e Δdeviance contra `ref` com IC por rodada.
    prev: alvo, teste, rodada, y e uma coluna por modelo; colunas: {rótulo: coluna}."""
    linhas = []
    for alvo, g in prev.groupby("alvo", sort=False):
        blocos = g["teste"].astype(str) + "-" + g["rodada"].astype(str)
        for rotulo, c in colunas.items():
            dif = bootstrap_blocos(deviance_jogo(g.y, g[c]) - deviance_jogo(g.y, g[ref]), blocos)
            linhas.append({"alvo": alvo, "modelo": rotulo, "deviance": deviance_jogo(g.y, g[c]).mean(),
                           "mae": (g.y - g[c]).abs().mean(), "rmse": float(np.sqrt(((g.y - g[c]) ** 2).mean())),
                           **dict(zip((f"dif_{ref}", "ic_inf", "ic_sup"), dif))})
    return pd.DataFrame(linhas)


def com_rodada(prev: pd.DataFrame, jogos: pd.DataFrame) -> pd.DataFrame:
    """Acrescenta a rodada às previsões antigas, que seguem a ordem dos jogos de cada temporada de teste."""
    partes = []
    for (alvo, t), g in prev.groupby(["alvo", "teste"], sort=False):
        j = jogos[jogos["temporada"] == t]
        y = (j[ALVOS[alvo][0]] + j[ALVOS[alvo][1]]).to_numpy()
        assert len(j) == len(g) and (y == g["y"].to_numpy()).all(), f"previsões desalinhadas {alvo} {t}"
        partes.append(g.assign(rodada=j["rodada"].to_numpy()))
    return pd.concat(partes)


# ---------- importância ----------

def grupo(col: str) -> str:
    for prefixo, nome in (("hist_", "média da temporada"), ("pi", "pi-rating"), ("gap", "GAP"),
                          ("dias_descanso", "descanso"), ("tecnico", "técnico"), ("sem_publico", "sem público"),
                          ("familia_", "formação"), ("fam_mando", "formação"), ("meio_frente", "formação"),
                          ("promovido", "promovido")):
        if col.startswith(prefixo):
            return nome
    return "forma recente"  # j5_/j10_/d3_/d5_/d10_


def importancia_permutacao(prever, X: pd.DataFrame, y, n=20, semente=0) -> pd.DataFrame:
    """Δdeviance média ao embaralhar cada coluna e cada grupo de colunas (juntas, mesma permutação)."""
    y = np.asarray(y, float)
    base = deviance_jogo(y, prever(X)).mean()
    rng = np.random.default_rng(semente)
    alvos = [("coluna", c, [c]) for c in X.columns]
    grupos = pd.Series(X.columns, index=X.columns).map(grupo)
    alvos += [("grupo", g, list(cs.index)) for g, cs in grupos.groupby(grupos)]
    linhas = []
    for tipo, nome, cols in alvos:
        difs = []
        for _ in range(n):
            Z, perm = X.copy(), rng.permutation(len(X))
            for c in cols:  # .array mantém o tipo com NULL (Float64) que o XGBoost recebe
                Z[c] = X[c].array[perm]
            difs.append(deviance_jogo(y, prever(Z)).mean() - base)
        linhas.append({"tipo": tipo, "nome": nome, "dif_deviance": float(np.mean(difs)),
                       "dp": float(np.std(difs))})
    return pd.DataFrame(linhas)


def shap_xgb(modelo, X: pd.DataFrame, n_arvores) -> pd.Series:
    """Contribuição média absoluta (TreeSHAP, escala do log da média) por coluna."""
    contrib = modelo.get_booster().predict(xgb.DMatrix(X.astype(float)), pred_contribs=True,
                                           iteration_range=(0, int(n_arvores)))
    return pd.Series(np.abs(contrib[:, :-1]).mean(axis=0), index=X.columns)


# ---------- probabilidades ----------

def alpha_momentos(y, mu) -> float:
    """α da NB2 pela regressão auxiliar de Cameron e Trivedi, como em walkforward._binomial_negativa."""
    y, mu = np.asarray(y, float), np.asarray(mu, float)
    return max(float(np.sum((y - mu) ** 2 - y) / np.sum(mu ** 2)), 1e-8)


def prob_acima(mu, linha, alpha=None) -> np.ndarray:
    """P(Y > linha) pela Poisson (alpha None) ou pela binomial negativa NB2 com média mu."""
    mu, k = np.asarray(mu, float), np.floor(linha)
    if alpha is None:
        return stats.poisson.sf(k, mu)
    r = 1 / alpha
    return stats.nbinom.sf(k, r, r / (r + mu))


def log_loss_jogo(o, p) -> np.ndarray:
    p = np.clip(np.asarray(p, float), 1e-12, 1 - 1e-12)
    return -(o * np.log(p) + (1 - o) * np.log(1 - p))


def brier_jogo(o, p) -> np.ndarray:
    return (np.asarray(p, float) - o) ** 2


def probabilistico(jogos, bases, escolhas: pd.DataFrame, prev: pd.DataFrame, alvo: str):
    """Probabilidades por jogo de teste (rodada 4) e α de cada dobra; prev = rodada4_previsoes do alvo."""
    from src.modelos.rodada4 import ajustar_final

    y = jogos[ALVOS[alvo][0]] + jogos[ALVOS[alvo][1]]
    temp, linha = jogos["temporada"], LINHAS[alvo]
    partes, alphas = [], []
    for t, g in prev.groupby("teste"):
        o = (g["y"] > linha).astype(float).to_numpy()
        out = {"alvo": alvo, "teste": t, "rodada": g["rodada"].to_numpy(), "acima": o,
               "frequencia": np.full(len(g), float((y[temp < t] > linha).mean()))}
        mus = {"ingenua": g["ingenua"].to_numpy()}
        # α fora da amostra: cada validação v < t prevista pelo modelo escolhido para t, ajustado em < v.
        val_y = {"ingenua": [], **{m: [] for m in MODELOS_R4}}
        val_mu = {m: [] for m in val_y}
        for v in validacoes(t):
            val_y["ingenua"].append(y[temp == v]); val_mu["ingenua"].append(np.full(int((temp == v).sum()), y[temp < v].mean()))
        for m in MODELOS_R4:
            e = escolhas.query("teste == @t and modelo == @m").iloc[0].to_dict()
            mus[m] = g[m].to_numpy()
            for v in validacoes(t):
                prever, X, te = ajustar_final(m, {**e, "teste": v}, bases, jogos, alvo)[:3]
                val_y[m].append(y[te]); val_mu[m].append(prever(X[te]))
        for m, mu in mus.items():
            a = alpha_momentos(np.concatenate(val_y[m]), np.concatenate(val_mu[m]))
            alphas.append({"alvo": alvo, "teste": t, "modelo": m, "alpha": a})
            out[f"{m}_poisson"] = prob_acima(mu, linha)
            out[f"{m}_nb"] = prob_acima(mu, linha, a)
        partes.append(pd.DataFrame(out))
    return pd.concat(partes), pd.DataFrame(alphas)


def resumo_probabilistico(probs: pd.DataFrame) -> pd.DataFrame:
    linhas = []
    for alvo, g in probs.groupby("alvo", sort=False):
        blocos = g["teste"].astype(str) + "-" + g["rodada"].astype(str)
        for c in [c for c in g.columns if c.endswith(("_poisson", "_nb"))] + ["frequencia"]:
            r = {"alvo": alvo, "previsao": c}
            for nome, f in (("log_loss", log_loss_jogo), ("brier", brier_jogo)):
                d = f(g.acima, g[c]) - f(g.acima, g["frequencia"])
                r[nome] = f(g.acima, g[c]).mean()
                r.update(dict(zip((f"dif_{nome}", f"ic_inf_{nome}", f"ic_sup_{nome}"), bootstrap_blocos(d, blocos))))
            linhas.append(r)
    return pd.DataFrame(linhas)


# ---------- execução ----------

def importancias(jogos, bases, escolhas: pd.DataFrame):
    from src.modelos.rodada4 import ajustar_final

    perm, shap = [], []
    for _, e in escolhas[escolhas["modelo"].isin(MODELOS_R4)].iterrows():
        alvo, t, m = e["alvo"], int(e["teste"]), e["modelo"]
        y = jogos[ALVOS[alvo][0]] + jogos[ALVOS[alvo][1]]
        prever, X, te, modelo = ajustar_final(m, e.to_dict(), bases, jogos, alvo)
        perm.append(importancia_permutacao(prever, X[te], y[te]).assign(alvo=alvo, teste=t, modelo=m))
        if m == "xgboost":
            s = shap_xgb(modelo, X[te], e["n_arvores"])
            shap.append(pd.DataFrame({"alvo": alvo, "teste": t, "coluna": s.index, "grupo": s.index.map(grupo),
                                      "shap_abs": s.to_numpy()}))
        print(f"importância {alvo} {t} {m}", flush=True)
    return pd.concat(perm), pd.concat(shap)


def main():
    from src.modelos.rodada4 import carregar_bases

    jogos, bases = carregar_bases()
    r3 = com_rodada(pd.read_csv(SAIDA / "rodada3_previsoes.csv"), jogos)
    r4 = pd.read_csv(SAIDA / f"{ETAPA}_previsoes.csv")
    esc4 = pd.read_csv(SAIDA / f"{ETAPA}_escolhas.csv")
    # Tabela única: rodadas 2 e 3 (r3 traz as colunas da rodada 2) e rodada 4, mesmos jogos e mesma ordem.
    assert (r3[["alvo", "teste", "rodada", "y"]].to_numpy() == r4[["alvo", "teste", "rodada", "y"]].to_numpy()).all()
    juntas = r3.reset_index(drop=True).join(r4[list(MODELOS_R4)].add_suffix("_r4").reset_index(drop=True))
    rotulos = {"média da liga": "ingenua", "R2 Poisson": "poisson", "R2 Poisson ridge": "poisson_ridge",
               "R3 Random Forest": "random_forest", "R3 XGBoost": "xgboost",
               **{f"R4 {n}": f"{c}_r4" for n, c in zip(("Poisson ridge", "Random Forest", "XGBoost"), MODELOS_R4)}}
    comp = comparacao(juntas, rotulos)
    melhor = comparacao(juntas, {k: v for k, v in rotulos.items() if k.startswith("R4")}, ref="random_forest")
    comp = comp.merge(melhor[["alvo", "modelo", "dif_random_forest", "ic_inf", "ic_sup"]].rename(
        columns={"ic_inf": "ic_inf_r3rf", "ic_sup": "ic_sup_r3rf"}), on=["alvo", "modelo"], how="left")
    comp.to_csv(SAIDA / f"comparacao_rodadas_r{ETAPA[6:]}.csv", index=False)
    print(comp.round(4).to_string(index=False), flush=True)

    probs = [probabilistico(jogos, bases, esc4[esc4.alvo == a], r4[r4.alvo == a], a) for a in ALVOS]
    p, alphas = pd.concat(x[0] for x in probs), pd.concat(x[1] for x in probs)
    p.to_csv(SAIDA / f"{ETAPA}_probabilidades.csv", index=False)
    alphas.to_csv(SAIDA / f"{ETAPA}_alpha_nb.csv", index=False)
    resumo = resumo_probabilistico(p)
    resumo.to_csv(SAIDA / f"{ETAPA}_probabilistico.csv", index=False)
    print(resumo.round(4).to_string(index=False), flush=True)

    perm, shap = importancias(jogos, bases, esc4)
    perm.to_csv(SAIDA / f"{ETAPA}_importancia_permutacao.csv", index=False)
    shap.to_csv(SAIDA / f"{ETAPA}_shap_xgboost.csv", index=False)
    g = perm[perm.tipo == "grupo"].groupby(["alvo", "modelo", "nome"])["dif_deviance"].mean()
    print(g.unstack("modelo").round(4).to_string())


if __name__ == "__main__":
    main()

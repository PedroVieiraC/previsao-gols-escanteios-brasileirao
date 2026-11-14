"""Pi-ratings pré-jogo (decisão de 07/11/2026).

Versão simplificada de Constantinou (2022, Journal of Sports Analytics 8(3)), sobre Constantinou e Fenton (2013):
cada time tem um rating em casa (H) e um fora (A), na unidade de saldo de gols. No jogo X (casa) x Y (fora):
  saldo previsto = R_XH − R_YA;  e = saldo observado − saldo previsto
  R_XH += λ·e;   R_XA += γ·(variação de R_XH)
  R_YA −= λ·e;   R_YH += γ·(variação de R_YA)
com λ = 0,035 e γ = 0,7 (valores do artigo de 2013, confirmados em 2022). O Ψ(e) do original fica de fora.

Início de cada temporada (regra do autor): fração `f` do rating final da temporada anterior; o resto vai para
0 (média da liga, o sistema é centrado em zero). Promovido: a mesma fração da média dos ratings finais dos
promovidos das temporadas anteriores; sem promovidos anteriores, 0. 2017 começa com todos em 0 (aquecimento).
As frações 0,5, 0,7 e 1,0 são candidatas; a escolha é feita no treino de cada dobra.

Sem vazamento: o rating do jogo é lido antes de atualizar com o placar dele; jogos em ordem de data e hora.
Uso: python -m src.features.pi_ratings
"""
import pandas as pd

from src.features.historico import carregar_canonica

LAMBDA, GAMMA = 0.035, 0.7
FRACOES = (0.5, 0.7, 1.0)


def _uma_fracao(jogos: pd.DataFrame, f: float, lam: float, gam: float) -> pd.DataFrame:
    ordem = jogos.sort_values(["data", "hora"], kind="stable")
    temporadas = sorted(ordem["temporada"].unique())
    times_temp = {t: set(ordem.loc[ordem.temporada == t, "mandante"]) for t in temporadas}
    finais_promovidos = []  # (H, A) ao fim da temporada em que o time subiu
    r = {}
    saida = {}
    for i, t in enumerate(temporadas):
        anterior = r
        promovidos = times_temp[t] - times_temp[temporadas[i - 1]] if i else set()
        ref = ((sum(h for h, _ in finais_promovidos) / len(finais_promovidos),
                sum(a for _, a in finais_promovidos) / len(finais_promovidos)) if finais_promovidos else (0.0, 0.0))
        r = {}
        for time in times_temp[t]:
            h, a = (0.0, 0.0) if i == 0 else ref if time in promovidos else anterior[time]
            r[time] = [f * h, f * a]
        for idx, j in ordem[ordem.temporada == t].iterrows():
            x, y = r[j.mandante], r[j.visitante]
            saida[idx] = (x[0], y[1])  # antes do jogo
            e = (j.gols_m - j.gols_v) - (x[0] - y[1])
            x[0] += lam * e
            x[1] += gam * lam * e
            y[1] -= lam * e
            y[0] -= gam * lam * e
        finais_promovidos += [tuple(r[p]) for p in promovidos]
    return pd.DataFrame.from_dict(saida, orient="index", columns=["casa_m", "fora_v"])


def pi_ratings(jogos: pd.DataFrame, fracoes=FRACOES, lam: float = LAMBDA, gam: float = GAMMA) -> pd.DataFrame:
    """Devolve, no índice de `jogos`, pi<f>_casa_m (rating em casa do mandante), pi<f>_fora_v (rating fora do
    visitante) e pi<f>_dif (saldo de gols previsto), com f em porcentagem (pi70_...)."""
    partes = []
    for f in fracoes:
        p = _uma_fracao(jogos, f, lam, gam).reindex(jogos.index)
        p["dif"] = p["casa_m"] - p["fora_v"]
        partes.append(p.add_prefix(f"pi{round(f * 100)}_"))
    return pd.concat(partes, axis=1)


if __name__ == "__main__":
    j = carregar_canonica()
    p = pi_ratings(j)
    print(p.shape, "nulos:", int(p.isna().sum().sum()))
    print(p.describe().T[["mean", "std", "min", "max"]].round(3).to_string())
    alvo = j["gols_m"] - j["gols_v"]
    fora_2017 = j["temporada"] > 2017
    for f in FRACOES:
        c = f"pi{round(f * 100)}_dif"
        print(c, "correlação com o saldo (2018–2024):", round(p.loc[fora_2017, c].corr(alvo[fora_2017]), 3))

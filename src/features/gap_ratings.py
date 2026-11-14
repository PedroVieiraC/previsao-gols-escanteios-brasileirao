"""GAP ratings pré-jogo (Wheatcroft, 2020, International Journal of Forecasting 36(3); decisão de 07/11/2026).

Cada time tem 4 índices por medida de desempenho ofensivo S (gols, escanteios, chutes...): ataque e defesa,
em casa (Ha, Hd) e fora (Aa, Ad). No jogo i (casa) x j (fora), com S_h e S_a observados:
  esperado do mandante = (Ha_i + Ad_j)/2;  esperado do visitante = (Aa_j + Hd_i)/2
  Ha_i += λ·φ1·(S_h − esp_m);  Aa_i += λ·(1−φ1)·(S_h − esp_m)
  Hd_i += λ·φ1·(S_a − esp_v);  Ad_i += λ·(1−φ1)·(S_a − esp_v)
  Aa_j += λ·φ2·(S_a − esp_v);  Ha_j += λ·(1−φ2)·(S_a − esp_v)
  Ad_j += λ·φ2·(S_h − esp_m);  Hd_j += λ·(1−φ2)·(S_h − esp_m)
todos limitados a ≥ 0 (equações 1 e 2 do artigo).

Regras deste trabalho:
- λ, φ1 e φ2 por medida, calibrados só com 2017 (aquecimento, fora da avaliação), em `calibrar()`.
- Time que aparece pela 1ª vez começa na média da liga dos jogos já disputados; na 1ª rodada de 2017 não há
  jogos anteriores: a feature fica NULL e os índices começam com o valor observado.
- Virada de temporada (mesma regra dos pi-ratings): fração f do índice final + (1 − f) da média final da liga.
  Promovido: média final dos promovidos anteriores (sem eles, a da liga), com a mesma fração. f em 0,5, 0,7 e 1.
- Índices lidos antes de atualizar com o jogo; jogos em ordem de data e hora.
Uso: python -m src.features.gap_ratings            (features na canônica)
     python -m src.features.gap_ratings calibrar   (refaz a calibração em 2017)
"""
import itertools
import sys

import numpy as np
import pandas as pd

from src.features.historico import carregar_canonica

MEDIDAS = {
    "gols": (("gols_m",), ("gols_v",)),
    "escanteios": (("escanteios_m",), ("escanteios_v",)),
    "chutes": (("chutes_m",), ("chutes_v",)),
    "chutes_no_alvo": (("chutes_no_alvo_m",), ("chutes_no_alvo_v",)),
    "chutes_escanteios": (("chutes_m", "escanteios_m"), ("chutes_v", "escanteios_v")),
}
# Calibrados com calibrar() em 07/11/2026, só com 2017 (erro quadrático do esperado contra o observado).
# Na grade, o erro muda pouco (ex.: escanteios 8,869 com (0,15; 0,7; 0,7) e 8,856 com o escolhido).
PARAMS = {
    "gols": (0.05, 0.9, 0.9),
    "escanteios": (0.15, 0.9, 0.7),
    "chutes": (0.2, 0.7, 0.9),
    "chutes_no_alvo": (0.15, 0.9, 0.7),
    "chutes_escanteios": (0.2, 0.7, 0.9),
}
FRACOES = (0.5, 0.7, 1.0)
CHAVES = ("Ha", "Hd", "Aa", "Ad")


def _serie(jogos, cols):
    return jogos[list(cols)].sum(axis=1, min_count=len(cols))


def _uma(jogos: pd.DataFrame, medida: str, f: float, lam: float, phi1: float, phi2: float) -> pd.DataFrame:
    cm, cv = MEDIDAS[medida]
    ordem = jogos.assign(_sh=_serie(jogos, cm), _sa=_serie(jogos, cv)).sort_values(["data", "hora"], kind="stable")
    temporadas = sorted(ordem["temporada"].unique())
    times_temp = {t: set(ordem.loc[ordem.temporada == t, "mandante"]) for t in temporadas}
    soma, n = 0.0, 0  # média da liga dos jogos com início estritamente anterior (por time-jogo)
    pendente, instante = (0.0, 0), None  # jogos do mesmo horário só entram na média depois dele
    finais_promovidos, r, saida = [], {}, {}
    for i, t in enumerate(temporadas):
        if i:
            media = {c: np.mean([v[c] for v in r.values()]) for c in CHAVES}
            promovidos = times_temp[t] - times_temp[temporadas[i - 1]]
            ref = ({c: np.mean([p[c] for p in finais_promovidos]) for c in CHAVES} if finais_promovidos else media)
            anterior, r = r, {}
            for time in times_temp[t]:
                base = ref if time in promovidos else anterior[time]
                r[time] = {c: f * base[c] + (1 - f) * media[c] for c in CHAVES}
        else:
            promovidos = set()
        for idx, j in ordem[ordem.temporada == t].iterrows():
            sh, sa = j._sh, j._sa
            if (j.data, j.hora) != instante:
                soma, n, pendente, instante = soma + pendente[0], n + pendente[1], (0.0, 0), (j.data, j.hora)
            for time in (j.mandante, j.visitante):
                if time not in r:
                    m = soma / n if n else np.nan
                    r[time] = dict.fromkeys(CHAVES, m)
            x, y = r[j.mandante], r[j.visitante]
            esp_m, esp_v = (x["Ha"] + y["Ad"]) / 2, (y["Aa"] + x["Hd"]) / 2
            saida[idx] = (esp_m, esp_v)
            if pd.isna(sh) or pd.isna(sa):
                continue  # estatística ausente: não atualiza
            if np.isnan(esp_m) or np.isnan(esp_v):  # 1ª rodada de 2017: começa no observado
                for c, v in (("Ha", sh), ("Hd", sa), ("Aa", sh), ("Ad", sa)):
                    x[c] = v if np.isnan(x[c]) else x[c]
                for c, v in (("Aa", sa), ("Ad", sh), ("Ha", sa), ("Hd", sh)):
                    y[c] = v if np.isnan(y[c]) else y[c]
            else:
                em, ev = sh - esp_m, sa - esp_v
                novos_x = {"Ha": x["Ha"] + lam * phi1 * em, "Aa": x["Aa"] + lam * (1 - phi1) * em,
                           "Hd": x["Hd"] + lam * phi1 * ev, "Ad": x["Ad"] + lam * (1 - phi1) * ev}
                novos_y = {"Aa": y["Aa"] + lam * phi2 * ev, "Ha": y["Ha"] + lam * (1 - phi2) * ev,
                           "Ad": y["Ad"] + lam * phi2 * em, "Hd": y["Hd"] + lam * (1 - phi2) * em}
                x.update({c: max(v, 0.0) for c, v in novos_x.items()})
                y.update({c: max(v, 0.0) for c, v in novos_y.items()})
            pendente = (pendente[0] + sh + sa, pendente[1] + 2)
        finais_promovidos += [r[p] for p in promovidos]
    return pd.DataFrame.from_dict(saida, orient="index", columns=["m", "v"])


def gap_ratings(jogos: pd.DataFrame, fracoes=FRACOES, params: dict | None = None) -> pd.DataFrame:
    """Devolve, no índice de `jogos`, gap<f>_<medida>_m (esperado do mandante), _v e _total."""
    params = params or PARAMS
    partes = []
    for medida, (lam, phi1, phi2) in params.items():
        for f in fracoes:
            p = _uma(jogos, medida, f, lam, phi1, phi2).reindex(jogos.index)
            p["total"] = p["m"] + p["v"]
            partes.append(p.add_prefix(f"gap{round(f * 100)}_{medida}_"))
    return pd.concat(partes, axis=1)


def calibrar(jogos: pd.DataFrame) -> dict:
    """Escolhe (λ, φ1, φ2) por medida minimizando o erro quadrático em 2017 (sem a 1ª rodada)."""
    j17 = jogos[jogos["temporada"] == jogos["temporada"].min()]
    grade = list(itertools.product((0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5), (0.3, 0.5, 0.7, 0.9), (0.3, 0.5, 0.7, 0.9)))
    escolhidos = {}
    for medida, (cm, cv) in MEDIDAS.items():
        obs = pd.concat([_serie(j17, cm), _serie(j17, cv)], axis=1).to_numpy()
        erros = []
        for lam, p1, p2 in grade:
            esp = _uma(j17, medida, 1.0, lam, p1, p2).reindex(j17.index).to_numpy()
            ok = ~np.isnan(esp).any(axis=1) & ~np.isnan(obs).any(axis=1)
            erros.append(np.mean((esp[ok] - obs[ok]) ** 2))
        escolhidos[medida] = grade[int(np.argmin(erros))]
        print(medida, escolhidos[medida], round(min(erros), 3))
    return escolhidos


if __name__ == "__main__":
    j = carregar_canonica()
    if sys.argv[1:] == ["calibrar"]:
        calibrar(j)
    else:
        g = gap_ratings(j)
        print(g.shape, "nulos por coluna (máx.):", int(g.isna().sum().max()))
        print(g.filter(like="gap70_").filter(like="_total").describe().T[["mean", "min", "max"]].round(2).to_string())

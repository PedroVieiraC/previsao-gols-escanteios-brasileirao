"""Histórico pré-jogo de cada time: médias de gols e escanteios (a favor e contra) até a partida anterior.

Regra do início de temporada e dos promovidos (decisão de 03/11/2026):
- Média da temporada só com jogos anteriores (equivale ao shift(1)), em ordem de data e hora.
- Encolhimento para uma referência: (n * média + k * referência) / (n + k), n = jogos já disputados.
- Referência: média final do próprio time na temporada anterior; se ele não jogou a anterior
  (promovido), média dos promovidos nas temporadas anteriores; sem promovidos anteriores, média da liga
  nas temporadas anteriores. Na primeira temporada da base não há referência (fica só a média, NULL na 1ª rodada).
- `promovido`: indicador; NULL na primeira temporada da base, onde não se sabe.
As temporadas não se sobrepõem no tempo (conferido), então dados de temporadas anteriores são pré-jogo.
"""
from pathlib import Path

import pandas as pd

STATS = {"gols": ("gols_m", "gols_v"), "escanteios": ("escanteios_m", "escanteios_v")}
# Padrão da função; nos modelos o k é escolhido na validação cronológica de cada dobra (rodada4.KS).
K_PADRAO = 5.0


def formato_longo(jogos: pd.DataFrame, stats: dict = STATS) -> pd.DataFrame:
    """Uma linha por time por jogo: <stat>_pro (do time) e <stat>_contra (do adversário)."""
    partes = []
    for lado, time, outro in (("m", "mandante", "visitante"), ("v", "visitante", "mandante")):
        p = pd.DataFrame({"jogo": jogos.index, "lado": lado, "temporada": jogos["temporada"].values,
                          "data": jogos["data"].values, "hora": jogos["hora"].values, "time": jogos[time].values})
        for s, (cm, cv) in stats.items():
            pro, contra = (cm, cv) if lado == "m" else (cv, cm)
            p[f"{s}_pro"], p[f"{s}_contra"] = jogos[pro].values, jogos[contra].values
        partes.append(p)
    longo = pd.concat(partes, ignore_index=True).sort_values(["data", "hora", "jogo"], kind="stable")
    _conferir_temporadas(longo)
    return marcar_promovidos(longo.reset_index(drop=True))


def _conferir_temporadas(longo):
    lim = longo.groupby("temporada")["data"].agg(["min", "max"]).sort_index()
    if (lim["min"].iloc[1:].values <= lim["max"].iloc[:-1].values).any():
        raise ValueError("Temporadas se sobrepõem no tempo; a referência da temporada anterior vazaria.")


def marcar_promovidos(longo: pd.DataFrame) -> pd.DataFrame:
    """promovido = não jogou a temporada anterior; NULL na primeira temporada da base."""
    times_por_temp = longo.groupby("temporada")["time"].agg(set)
    equipes = longo[["time", "temporada"]].drop_duplicates()
    equipes["promovido"] = [
        pd.NA if t - 1 not in times_por_temp.index else time not in times_por_temp[t - 1]
        for time, t in zip(equipes["time"], equipes["temporada"])
    ]
    equipes["promovido"] = equipes["promovido"].astype("boolean")
    return longo.merge(equipes, on=["time", "temporada"], how="left")


def referencias(longo: pd.DataFrame, cols: list[str], extra: tuple = ()) -> pd.DataFrame:
    """ref_<col> por (time, temporada, *extra), só com temporadas anteriores.
    Com extra=("lado",), a referência do mandante vem só de jogos em casa e a do visitante só de jogos fora."""
    chave = ["time", "temporada", *extra]
    final = longo.groupby(chave)[cols].mean()
    linhas = []
    for _, e in longo[chave].drop_duplicates().iterrows():
        t, outros = e.temporada, tuple(e[c] for c in extra)
        anterior = (e.time, t - 1, *outros)
        if anterior in final.index:
            valores = final.loc[anterior]
        else:
            antes = longo[longo["temporada"] < t]
            for c, v in zip(extra, outros):
                antes = antes[antes[c] == v]
            promovidos = antes[antes["promovido"].fillna(False)]
            base = promovidos if len(promovidos) else antes
            valores = base[cols].mean() if len(base) else pd.Series(float("nan"), index=cols)
        linhas.append({**e.to_dict(), **{f"ref_{c}": valores[c] for c in cols}})
    return pd.DataFrame(linhas)


def encolher(media: pd.Series, n: pd.Series, ref: pd.Series, k: float) -> pd.Series:
    """(n·média + k·ref)/(n + k); sem jogos, a referência; sem referência, a média."""
    return ((n * media + k * ref) / (n + k)).where(n > 0, ref).where(ref.notna(), media)


def para_largo(longo: pd.DataFrame, colunas: list[str], indice) -> pd.DataFrame:
    """Uma linha por jogo, com <coluna>_m e <coluna>_v."""
    largo = longo.pivot(index="jogo", columns="lado", values=colunas)
    largo.columns = [f"{v}_{lado}" for v, lado in largo.columns]
    tipos = {}
    for c in largo.columns:
        tipos[c] = ("boolean" if c.startswith("promovido") else "int64" if c.startswith("hist_jogos")
                    else "float64")  # o pivot com tipos diferentes devolve object
    return largo.astype(tipos).reindex(indice)


def historico_times(jogos: pd.DataFrame, k: float = K_PADRAO) -> pd.DataFrame:
    """Devolve, no índice de `jogos`, hist_<stat>_<pro|contra>_<m|v>, hist_jogos_<m|v> e promovido_<m|v>."""
    longo = formato_longo(jogos)
    cols = [f"{s}_{t}" for s in STATS for t in ("pro", "contra")]
    longo = longo.merge(referencias(longo, cols), on=["time", "temporada"], how="left")
    g = longo.groupby(["time", "temporada"], sort=False)
    longo["hist_jogos"] = g.cumcount()
    for c in cols:
        anteriores = g[c].cumsum() - longo[c]          # soma só dos jogos anteriores
        n = g[c].transform(lambda x: x.notna().cumsum()) - longo[c].notna()
        longo[f"hist_{c}"] = encolher(anteriores / n.where(n > 0), n, longo[f"ref_{c}"], k)
    return para_largo(longo, [f"hist_{c}" for c in cols] + ["hist_jogos", "promovido"], jogos.index)


DADOS = Path(__file__).resolve().parents[2] / "data"


def ler_tabela(nome: str) -> pd.DataFrame:
    """Lê data/<nome>.csv; data vira datetime.date e hora datetime.time (nulo = None)."""
    df = pd.read_csv(DADOS / f"{nome}.csv")
    for c in df.columns.intersection(["data", "primeiro_jogo", "ultimo_jogo"]):
        df[c] = pd.Series(pd.to_datetime(df[c]).dt.date, dtype=object)
    if "hora" in df:
        df["hora"] = pd.Series(pd.to_datetime(df["hora"], format="%H:%M:%S").dt.time, dtype=object).where(df["hora"].notna(), None)
    return df


def carregar_canonica() -> pd.DataFrame:
    return ler_tabela("tb_main_brasileirao_tcc").sort_values(["data", "hora"], kind="stable", ignore_index=True)


if __name__ == "__main__":
    h = historico_times(carregar_canonica())
    print(h.describe().T[["count", "mean", "min", "max"]].round(2).to_string())

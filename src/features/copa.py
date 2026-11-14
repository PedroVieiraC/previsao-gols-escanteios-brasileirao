"""Feature "jogou a Copa do Brasil entre duas rodadas" (data/norm_copa_brasil.csv).

copa_entre_rodadas_<m|v>: o time teve jogo da Copa com início depois do seu jogo anterior da Série A e
antes do início do jogo-alvo (estritamente). NULL no 1º jogo de cada temporada, onde não há rodada anterior.
"""
import numpy as np
import pandas as pd


def inicio(df: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(df["data"].astype(str) + " " + df["hora"].astype(str))


def copa_entre_rodadas(jogos: pd.DataFrame, copa: pd.DataFrame) -> pd.DataFrame:
    """jogos: canônica (temporada, data, hora, mandante, visitante). copa: norm_copa_brasil."""
    copa = copa.assign(inicio=inicio(copa.fillna({"hora": "00:00:00"})))
    por_time = {}
    for lado in ("mandante", "visitante"):
        for time_, g in copa.dropna(subset=[lado]).groupby(lado):
            por_time.setdefault(time_, []).extend(g["inicio"])
    por_time = {t: np.sort(np.array(v, dtype="datetime64[ns]")) for t, v in por_time.items()}

    saida = pd.DataFrame(index=jogos.index)
    longo = pd.concat([
        pd.DataFrame({"jogo": jogos.index, "lado": l, "temporada": jogos["temporada"].values,
                      "time": jogos[c].values, "inicio": inicio(jogos).values})
        for l, c in (("m", "mandante"), ("v", "visitante"))
    ]).sort_values(["time", "inicio"])
    longo["anterior"] = longo.groupby(["time", "temporada"])["inicio"].shift(1)
    valores = []
    for r in longo.itertuples():
        if pd.isna(r.anterior):
            valores.append(pd.NA)
            continue
        datas = por_time.get(r.time, np.array([], dtype="datetime64[ns]"))
        ini, fim = np.datetime64(r.anterior, "ns"), np.datetime64(r.inicio, "ns")
        valores.append(bool(np.searchsorted(datas, fim, "left") > np.searchsorted(datas, ini, "right")))
    longo["copa"] = pd.array(valores, dtype="boolean")
    for lado in ("m", "v"):
        parte = longo[longo["lado"] == lado].set_index("jogo")["copa"]
        saida[f"copa_entre_rodadas_{lado}"] = parte.reindex(jogos.index)
    return saida


def main():
    from src.features.historico import carregar_canonica, ler_tabela

    copa = ler_tabela("norm_copa_brasil")
    jogos = carregar_canonica()
    c = copa_entre_rodadas(jogos, copa).join(jogos["temporada"])
    print(c.groupby("temporada")[["copa_entre_rodadas_m", "copa_entre_rodadas_v"]].sum().to_string())


if __name__ == "__main__":
    main()

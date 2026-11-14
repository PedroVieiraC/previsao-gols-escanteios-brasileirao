"""Médias com decaimento exponencial pré-jogo (decisão de 07/11/2026).

Alternativa às janelas de 5 e 10 jogos (janelas.py): todos os jogos anteriores da temporada contam, com peso
que cai à metade a cada `h` jogos (meia-vida). As meias-vidas 3, 5 e 10 são candidatas; a escolha é feita
no treino de cada dobra da validação.
- Mesmas séries, recortes e estatísticas de janelas.py: shift(1) dentro de (time, temporada[, lado]),
  ordenado por data e hora; o jogo-alvo nunca entra.
- Não atravessa temporadas. Encolhe sempre para a referência de historico.py com n = jogos anteriores
  na temporada: (n·média + k·ref)/(n + k); o peso da referência some conforme a temporada avança.

Uso: python -m src.features.decaimento   (resumo na canônica)
"""
import pandas as pd

from src.features.historico import K_PADRAO, carregar_canonica, encolher, formato_longo, para_largo, referencias
from src.features.janelas import RECORTES, STATS

MEIAS_VIDAS = (3, 5, 10)


def decaimento_times(jogos: pd.DataFrame, meias_vidas=MEIAS_VIDAS, k: float = K_PADRAO) -> pd.DataFrame:
    """Devolve, no índice de `jogos`, d<h>_<recorte>_<stat>_<pro|contra>_<m|v>."""
    longo = formato_longo(jogos, STATS).drop(columns="posse_contra")
    cols = [c for c in longo.columns if c.endswith(("_pro", "_contra"))]
    saida = []
    for recorte, extra in RECORTES.items():
        chave = ["time", "temporada", *extra]
        ref = longo[chave].merge(referencias(longo, cols, extra), on=chave, how="left")
        g = longo.groupby(chave, sort=False)
        for c in cols:
            anterior = g[c].shift(1)  # mesmo índice de longo, dentro de cada série
            series = [longo[k_] for k_ in chave]
            n = anterior.notna().groupby(series, sort=False).cumsum()
            for h in meias_vidas:
                media = (anterior.groupby(series, sort=False).ewm(halflife=h, ignore_na=True).mean()
                         .reset_index(level=list(range(len(chave))), drop=True).reindex(longo.index))
                longo[f"d{h}_{recorte}_{c}"] = encolher(media, n, ref[f"ref_{c}"], k)
                saida.append(f"d{h}_{recorte}_{c}")
    return para_largo(longo, saida, jogos.index)


if __name__ == "__main__":
    d = decaimento_times(carregar_canonica())
    print(d.shape)
    print(d.isna().sum().sort_values().tail(5).to_string())
    print(d.filter(like="geral_escanteios_pro").describe().T[["mean", "min", "max"]].round(2).to_string())

"""Janelas pré-jogo: médias dos últimos w jogos de cada time (decisão de 03/11/2026).

- w = 5 e 10; recortes "geral" (casa e fora) e "mando" (mandante só em casa, visitante só fora).
- shift(1) dentro de cada série (time, temporada[, lado]), ordenada por data e hora: o jogo-alvo nunca entra.
  No recorte por mando, a série é filtrada pelo mando antes do shift.
- A janela não atravessa temporadas. Com menos de w jogos, encolhe para a referência de historico.py
  (temporada anterior do time; promovidos; liga), no mesmo recorte.
- Posse: só a favor (a contra é 100 menos ela).

Uso: python -m src.features.janelas   (resumo das 72 colunas na canônica)
"""
import pandas as pd

from src.features.historico import K_PADRAO, carregar_canonica, encolher, formato_longo, para_largo, referencias

STATS = {
    "gols": ("gols_m", "gols_v"),
    "escanteios": ("escanteios_m", "escanteios_v"),
    "chutes": ("chutes_m", "chutes_v"),
    "chutes_no_alvo": ("chutes_no_alvo_m", "chutes_no_alvo_v"),
    "posse": ("posse_m", "posse_v"),
}
JANELAS = (5, 10)
RECORTES = {"geral": (), "mando": ("lado",)}


def janelas_times(jogos: pd.DataFrame, janelas=JANELAS, k: float = K_PADRAO) -> pd.DataFrame:
    """Devolve, no índice de `jogos`, j<w>_<recorte>_<stat>_<pro|contra>_<m|v>."""
    longo = formato_longo(jogos, STATS).drop(columns="posse_contra")
    cols = [c for c in longo.columns if c.endswith(("_pro", "_contra"))]
    saida = []
    for recorte, extra in RECORTES.items():
        chave = ["time", "temporada", *extra]
        ref = longo[chave].merge(referencias(longo, cols, extra), on=chave, how="left")
        g = longo.groupby(chave, sort=False)
        for c in cols:
            anterior = g[c].shift(1)  # mesmo índice de longo, dentro de cada série
            for w in janelas:
                rol = anterior.groupby([longo[k_] for k_ in chave], sort=False).rolling(w, min_periods=1)
                media = rol.mean().reset_index(level=list(range(len(chave))), drop=True).reindex(longo.index)
                n = rol.count().reset_index(level=list(range(len(chave))), drop=True).reindex(longo.index).fillna(0)
                valor = encolher(media, n, ref[f"ref_{c}"], k).where(n < w, media)
                longo[f"j{w}_{recorte}_{c}"] = valor
                saida.append(f"j{w}_{recorte}_{c}")
    return para_largo(longo, saida, jogos.index)


if __name__ == "__main__":
    j = janelas_times(carregar_canonica())
    print(j.shape)
    print(j.isna().sum().sort_values().tail(5).to_string())
    print(j.filter(like="gols_pro").describe().T[["mean", "min", "max"]].round(2).to_string())

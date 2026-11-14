"""Features de formação tática (decisões de 07/11/2026).

A formação do jogo-alvo é conhecida antes do jogo (escalação divulgada) e a do adaoduque é a inicial.
- familia_<m|v>: nível de amplitude (A alta, B média, C estreita, D cinco atrás), critério tático do autor.
- meio_frente_<m|v>: jogadores do meio para a frente = 10 − nº de defensores.
- fam_<recorte>_escanteios_<pro|contra>_<m|v>: escanteios médios do time nos jogos anteriores com a mesma
  família (todas as temporadas, porque as famílias raras têm poucos jogos), geral ou só no mesmo mando;
  encolhidos (n·média + k·ref)/(n + k) para a média da liga na família (e no mando), calculada só com jogos
  de início estritamente anterior. Sem jogo anterior na família, vale a referência; se nem a liga tem jogo
  anterior na família (no mando), vale a média da liga (no mando).

Uso: python -m src.features.formacao
"""
import pandas as pd

from src.features.historico import K_PADRAO, carregar_canonica, encolher, formato_longo, para_largo

FAMILIAS = {
    "A": ("3-5-2", "3-4-3", "3-4-2-1", "3-4-1-2", "3-1-4-2", "3-2-4-1", "3-5-1-1", "3-3-3-1",
          "4-4-2", "4-4-1-1", "4-2-4", "4-5-1"),
    "B": ("4-3-3", "4-2-3-1", "4-1-4-1"),
    "C": ("4-1-2-1-2", "4-2-2-2", "4-3-1-2", "4-1-3-2", "4-3-2-1"),
    "D": ("5-4-1", "5-3-2"),
}
FAMILIA = {f: fam for fam, fs in FAMILIAS.items() for f in fs}
RECORTES = {"geral": (), "mando": ("lado",)}


def familia(formacao: pd.Series) -> pd.Series:
    fam = formacao.map(FAMILIA)
    faltando = sorted(set(formacao[fam.isna()].dropna()))
    if faltando:
        raise ValueError(f"Formações sem família (definir com o autor): {faltando}")
    return fam


def _media_liga_anterior(longo: pd.DataFrame, chave: list[str], col: str) -> pd.Series:
    """Média de `col` por `chave`, só com jogos de início estritamente anterior ao da linha."""
    instante = ["data", "hora"]
    por = longo.groupby(chave + instante, sort=False)[col].agg(["sum", "count"]).reset_index()
    por = por.sort_values(instante, kind="stable")
    g = por.groupby(chave, sort=False)
    por["ref"] = (g["sum"].cumsum() - por["sum"]) / (g["count"].cumsum() - por["count"]).replace(0, float("nan"))
    return longo[chave + instante].merge(por[chave + instante + ["ref"]], how="left")["ref"].set_axis(longo.index)


def formacao_features(jogos: pd.DataFrame, k: float = K_PADRAO) -> pd.DataFrame:
    longo = formato_longo(jogos, {"escanteios": ("escanteios_m", "escanteios_v")})
    lado_form = {"m": "formacao_m", "v": "formacao_v"}
    longo["familia"] = familia(pd.Series([jogos.at[j, lado_form[l]] for j, l in zip(longo["jogo"], longo["lado"])],
                                         index=longo.index))
    saida = []
    for recorte, extra in RECORTES.items():
        chave = ["time", "familia", *extra]
        g = longo.groupby(chave, sort=False)
        for c in ("escanteios_pro", "escanteios_contra"):
            anterior = g[c].shift(1)  # longo já está em ordem de data e hora
            series = [longo[k_] for k_ in chave]
            n = anterior.notna().groupby(series, sort=False).cumsum()
            media = anterior.groupby(series, sort=False).expanding().mean() \
                .reset_index(level=list(range(len(chave))), drop=True).reindex(longo.index)
            # Família ainda sem jogo anterior (no mando): vale a média da liga (no mando).
            ref = _media_liga_anterior(longo, ["familia", *extra], c).fillna(
                _media_liga_anterior(longo.assign(_liga=1), ["_liga", *extra], c))
            longo[f"fam_{recorte}_{c}"] = encolher(media, n, ref, k)
            saida.append(f"fam_{recorte}_{c}")
    out = para_largo(longo, saida, jogos.index)
    for lado in ("m", "v"):
        out[f"familia_{lado}"] = familia(jogos[f"formacao_{lado}"])
        out[f"meio_frente_{lado}"] = 10 - jogos[f"formacao_{lado}"].str[0].astype(int)
    return out


if __name__ == "__main__":
    f = formacao_features(carregar_canonica())
    print(f.shape, "nulos:", f.isna().sum()[lambda s: s > 0].to_dict())
    print(f["familia_m"].value_counts().to_dict(), f["meio_frente_m"].value_counts().to_dict())
    print(f.filter(like="fam_").describe().T[["mean", "min", "max"]].round(2).to_string())

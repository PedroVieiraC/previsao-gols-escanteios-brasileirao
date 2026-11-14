"""Features de técnico, conhecidas antes do jogo (o técnico do jogo-alvo é público antes do início).

Identidade: pessoa_id da data/tb_tecnicos_depara.csv ou, sem ID, o nome_padrao (join por tecnico_original e clube).
Regra do interino (decisão de 03/11/2026): técnico diferente por 1 jogo vale como o anterior; com 2 ou mais
jogos conta como técnico. Sem vazamento, a troca só é reconhecida no 2º jogo seguido do técnico novo
(saber que o interino ficou 1 jogo exigiria ver o jogo seguinte). A-B-A: B nunca vira troca.
Esse atraso de 1 jogo é a forma de aplicar a regra sem usar o futuro.

tecnico_jogos_<m|v>: jogos anteriores do clube na Série A sob o técnico efetivo, seguidos, atravessando
  temporadas. NULL quando o início do trabalho não está na base: 1ª aparição do clube (2017) ou volta depois
  de temporada fora da Série A, até a primeira troca observada (ausência não é zero).
tecnico_trocas_temp_<m|v>: trocas efetivas do clube na temporada até o jogo-alvo, inclusive.
"""
import pandas as pd

from src.features.copa import inicio


def identidades(jogos: pd.DataFrame, depara: pd.DataFrame) -> pd.DataFrame:
    """Uma linha por (jogo, lado) com a identidade do técnico, ordenada por clube, data e hora."""
    ident = depara.assign(ident=depara["pessoa_id"].astype("Int64").astype("string")
                          .fillna("nome:" + depara["nome_padrao"])).set_index(["tecnico_original", "clube"])["ident"]
    longo = pd.concat([
        pd.DataFrame({"jogo": jogos.index, "lado": l, "temporada": jogos["temporada"].to_numpy(),
                      "inicio": inicio(jogos).to_numpy(), "clube": jogos[c].to_numpy(), "tecnico": jogos[t].to_numpy()})
        for l, c, t in (("m", "mandante", "tecnico_m"), ("v", "visitante", "tecnico_v"))
    ])
    longo["ident"] = ident.reindex(pd.MultiIndex.from_arrays([longo.tecnico, longo.clube])).to_numpy()
    if longo["ident"].isna().any():
        faltam = longo.loc[longo["ident"].isna(), ["tecnico", "clube"]].drop_duplicates().to_records(index=False)
        raise ValueError(f"técnico sem de-para: {list(faltam)[:5]}")
    return longo.sort_values(["clube", "inicio"], kind="stable")


def tecnico_features(jogos: pd.DataFrame, depara: pd.DataFrame) -> pd.DataFrame:
    """jogos: canônica (temporada, data, hora, mandante, visitante, tecnico_m, tecnico_v). depara: data/tb_tecnicos_depara.csv."""
    linhas = []
    for _, g in identidades(jogos, depara).groupby("clube", sort=False):
        efetivo = cand = None
        inicio_run = cand_inicio = n_cand = 0
        conhecido, temp_ant, trocas = False, None, 0
        for k, r in enumerate(g.itertuples()):
            if temp_ant is None or r.temporada > temp_ant + 1:  # 1ª aparição ou volta da Série B
                efetivo, inicio_run, conhecido, cand = r.ident, k, False, None
            if r.temporada != temp_ant:
                trocas = 0
            if r.ident == efetivo:
                cand = None
            elif r.ident == cand:
                n_cand += 1
            else:
                cand, cand_inicio, n_cand = r.ident, k, 1
            if cand is not None and n_cand >= 2:  # 2º jogo seguido do técnico novo: troca efetiva
                efetivo, inicio_run, conhecido, cand = cand, cand_inicio, True, None
                trocas += 1
            linhas.append((r.jogo, r.lado, k - inicio_run if conhecido else pd.NA, trocas))
            temp_ant = r.temporada
    t = pd.DataFrame(linhas, columns=["jogo", "lado", "jogos", "trocas"])
    saida = pd.DataFrame(index=jogos.index)
    for lado in ("m", "v"):
        p = t[t["lado"] == lado].set_index("jogo")
        saida[f"tecnico_jogos_{lado}"] = p["jogos"].astype("Int64").reindex(jogos.index)
        saida[f"tecnico_trocas_temp_{lado}"] = p["trocas"].astype("Int64").reindex(jogos.index)
    return saida

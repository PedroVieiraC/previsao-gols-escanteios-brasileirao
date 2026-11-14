"""Tabela de modelagem: uma linha por jogo, conjunto enxuto por alvo (decisão de 08/11/2026).

Todas as features são calculadas uma vez (`calcular_base`); `enxuta` escolhe uma versão de cada grupo:
- esperado do confronto: <s>_esp_m = (s_pro_m + s_contra_v)/2 e <s>_esp_v = (s_pro_v + s_contra_m)/2,
  na média da temporada (só o alvo) e na forma recente (`forma`: j5/j10/d3/d5/d10 × geral/mando);
- pi<f>_dif; GAP<f> mandante e visitante (o total é a soma, redundante num modelo linear);
- contexto: dias de descanso, sem público, trocas de técnico na temporada; `com_nulos=True` acrescenta
  promovido e tecnico_jogos, que têm NULL documentado (para modelos que tratam ausência, como o XGBoost);
- formação, só para escanteios: "sem", "familia" (indicadores A, C, D contra B + esperado por família no
  mando) ou "meio_frente". A posse fica de fora.
"""
import pandas as pd

from src.features.calendario import dias_descanso, sem_publico
from src.features.decaimento import decaimento_times
from src.features.formacao import formacao_features
from src.features.gap_ratings import gap_ratings
from src.features.historico import K_PADRAO, carregar_canonica, historico_times, ler_tabela
from src.features.janelas import janelas_times
from src.features.pi_ratings import pi_ratings
from src.features.tecnicos import tecnico_features

ALVOS = {"gols": ("gols_m", "gols_v"), "escanteios": ("escanteios_m", "escanteios_v")}
STATS_ALVO = {"gols": ("gols", "chutes_no_alvo"), "escanteios": ("escanteios", "chutes")}
GAP_ALVO = {"gols": ("gols", "chutes_no_alvo"), "escanteios": ("escanteios", "chutes_escanteios")}
FORMAS = tuple(f"{p}_{r}" for p in ("j5", "j10", "d3", "d5", "d10") for r in ("geral", "mando"))
FRACOES = (50, 70, 100)
FORMACOES = ("sem", "familia", "meio_frente")


def calcular_base(jogos: pd.DataFrame, copa: pd.DataFrame, depara: pd.DataFrame, k: float = K_PADRAO) -> pd.DataFrame:
    partes = [historico_times(jogos, k), janelas_times(jogos, k=k), decaimento_times(jogos, k=k), pi_ratings(jogos),
              gap_ratings(jogos), formacao_features(jogos, k), dias_descanso(jogos, copa), sem_publico(jogos),
              tecnico_features(jogos, depara)]
    return pd.concat(partes, axis=1)


def _esperado(base: pd.DataFrame, prefixo: str, s: str) -> dict:
    return {f"{prefixo}_{s}_esp_m": (base[f"{prefixo}_{s}_pro_m"] + base[f"{prefixo}_{s}_contra_v"]) / 2,
            f"{prefixo}_{s}_esp_v": (base[f"{prefixo}_{s}_pro_v"] + base[f"{prefixo}_{s}_contra_m"]) / 2}


def enxuta(base: pd.DataFrame, alvo: str, forma: str = "d5_geral", f_pi: int = 70, f_gap: int = 70,
           formacao: str = "sem", com_nulos: bool = False) -> pd.DataFrame:
    s_alvo = STATS_ALVO[alvo][0]
    cols = dict(_esperado(base, "hist", s_alvo))
    for s in STATS_ALVO[alvo]:
        cols.update(_esperado(base, forma, s))
    cols[f"pi{f_pi}_dif"] = base[f"pi{f_pi}_dif"]
    for g in GAP_ALVO[alvo]:
        for lado in ("m", "v"):
            cols[f"gap{f_gap}_{g}_{lado}"] = base[f"gap{f_gap}_{g}_{lado}"]
    for lado in ("m", "v"):
        cols[f"dias_descanso_{lado}"] = base[f"dias_descanso_{lado}"]
        cols[f"tecnico_trocas_temp_{lado}"] = base[f"tecnico_trocas_temp_{lado}"].astype(float)
    cols["sem_publico"] = base["sem_publico"].astype(float)
    if alvo == "escanteios" and formacao == "familia":
        for lado in ("m", "v"):
            for fam in ("A", "C", "D"):
                cols[f"familia_{fam}_{lado}"] = base[f"familia_{lado}"].eq(fam).astype(float)
        cols.update(_esperado(base, "fam_mando", "escanteios"))
    elif alvo == "escanteios" and formacao == "meio_frente":
        for lado in ("m", "v"):
            cols[f"meio_frente_{lado}"] = base[f"meio_frente_{lado}"].astype(float)
    if com_nulos:
        for lado in ("m", "v"):
            cols[f"promovido_{lado}"] = base[f"promovido_{lado}"].astype("Float64")
            cols[f"tecnico_jogos_{lado}"] = base[f"tecnico_jogos_{lado}"].astype("Float64")
    return pd.DataFrame(cols, index=base.index)


def carregar() -> tuple[pd.DataFrame, pd.DataFrame]:
    """(jogos da canônica, base com todas as features)."""
    jogos = carregar_canonica()
    return jogos, calcular_base(jogos, ler_tabela("norm_copa_brasil"), ler_tabela("tb_tecnicos_depara"))


if __name__ == "__main__":
    jogos, base = carregar()
    print("base:", base.shape)
    for alvo in ALVOS:
        t = enxuta(base, alvo, formacao="familia")
        print(alvo, t.shape, "nulos a partir de 2018:", int(t[jogos["temporada"] >= 2018].isna().sum().sum()))

"""Features de calendário conhecidas antes do jogo (decisões de 03 e 04/11/2026).

dias_descanso_<m|v>: dias (com fração) entre o início do jogo anterior do time, na Série A ou na Copa do Brasil
  (norm_copa_brasil, só jogos realizados), e o início do jogo-alvo; truncado em DESCANSO_MAX. Sem jogo anterior
  na base (ou só muito antes, como no 1º jogo da temporada) vale DESCANSO_MAX: nunca é nulo.
  Estaduais, Libertadores e Sul-Americana não estão na base, então o descanso é superestimado.
sem_publico: jogo com portões fechados por causa da pandemia: temporada 2020 inteira e 2021 antes de 02/10/2021,
  quando a Série A voltou a ter público (Conselho Técnico de 28/09/2021, a partir da 23ª rodada).
"""
import numpy as np
import pandas as pd

from src.features.copa import inicio

DESCANSO_MAX = 14.0  # p99 dos intervalos fora do 1º jogo da temporada é 14,5 dias
VOLTA_PUBLICO = pd.Timestamp("2021-10-02")


def dias_descanso(jogos: pd.DataFrame, copa: pd.DataFrame) -> pd.DataFrame:
    """jogos: canônica (data, hora, mandante, visitante). copa: norm_copa_brasil. Mesmo índice de jogos."""
    copa = copa.assign(inicio=inicio(copa.fillna({"hora": "00:00:00"})))
    serie = inicio(jogos)
    eventos = pd.concat(
        [pd.DataFrame({"time": copa[c], "inicio": copa["inicio"]}) for c in ("mandante", "visitante")]
        + [pd.DataFrame({"time": jogos[c], "inicio": serie}) for c in ("mandante", "visitante")]
    ).dropna(subset=["time"])
    por_time = {t: np.unique(g["inicio"].to_numpy("datetime64[ns]")) for t, g in eventos.groupby("time")}

    saida = pd.DataFrame(index=jogos.index)
    for lado, col in (("m", "mandante"), ("v", "visitante")):
        valores = []
        for time_, ini in zip(jogos[col], serie.to_numpy("datetime64[ns]")):
            datas = por_time[time_]
            k = np.searchsorted(datas, ini, "left")  # só jogos com início estritamente anterior
            dias = (ini - datas[k - 1]) / np.timedelta64(1, "D") if k else DESCANSO_MAX
            valores.append(min(float(dias), DESCANSO_MAX))
        saida[f"dias_descanso_{lado}"] = valores
    return saida


def sem_publico(jogos: pd.DataFrame) -> pd.Series:
    return (jogos["temporada"].eq(2020) | (jogos["temporada"].eq(2021) & inicio(jogos).lt(VOLTA_PUBLICO))
            ).rename("sem_publico")

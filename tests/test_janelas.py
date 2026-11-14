import unittest
from datetime import date, time, timedelta

import pandas as pd

from src.features.janelas import janelas_times


def jogo(temp, d, m, v, gm, gv):
    return {"temporada": temp, "data": d, "hora": time(16), "mandante": m, "visitante": v,
            "gols_m": gm, "gols_v": gv, "escanteios_m": 5, "escanteios_v": 3,
            "chutes_m": 10, "chutes_v": 8, "chutes_no_alvo_m": 4, "chutes_no_alvo_v": 3,
            "posse_m": 55, "posse_v": 45}


def temporada(ano, inicio, rodadas):
    return [jogo(ano, inicio + timedelta(days=7 * i), m, v, gm, gv) for i, (m, v, gm, gv) in enumerate(rodadas)]


# A alterna casa e fora contra B. Em 2018, A marca 1, 2, 3, 4, 5, 6 gols.
J = pd.DataFrame(
    temporada(2017, date(2017, 5, 1), [("A", "B", 2, 0), ("B", "A", 1, 1), ("A", "B", 2, 2), ("B", "A", 0, 0)])
    + temporada(2018, date(2018, 5, 1), [("A", "B", 1, 0), ("B", "A", 0, 2), ("A", "B", 3, 0),
                                          ("B", "A", 0, 4), ("A", "B", 5, 0), ("B", "A", 0, 6)])
)


class JanelasTest(unittest.TestCase):
    def setUp(self):
        self.j = janelas_times(J, janelas=(2,), k=2)

    def test_janela_geral_cheia(self):
        # 2018, 3º jogo (índice 6), A mandante: últimos 2 jogos de A = 1 e 2 gols
        self.assertEqual(self.j.loc[6, "j2_geral_gols_pro_m"], 1.5)

    def test_janela_por_mando(self):
        # 2018, 5º jogo (índice 8), A em casa: jogos anteriores em casa = 1 e 3 gols
        self.assertEqual(self.j.loc[8, "j2_mando_gols_pro_m"], 2.0)
        # 6º jogo (índice 9), A fora: jogos anteriores fora = 2 e 4
        self.assertEqual(self.j.loc[9, "j2_mando_gols_pro_v"], 3.0)

    def test_encolhe_com_janela_incompleta(self):
        # 2018, 1º jogo (índice 4), A em casa: sem jogos, vale a referência = A em casa em 2017 (2 e 2)
        self.assertEqual(self.j.loc[4, "j2_mando_gols_pro_m"], 2.0)
        # 2º jogo (índice 5), A fora, recorte geral: 1 jogo (1 gol), ref geral de A em 2017 = (2+1+2+0)/4; k=2
        self.assertAlmostEqual(self.j.loc[5, "j2_geral_gols_pro_v"], (1 + 2 * 1.25) / 3)

    def test_janela_nao_atravessa_temporada(self):
        # 1º jogo de 2018 não usa os jogos de 2017 como janela (só como referência)
        self.assertEqual(self.j.loc[4, "j2_geral_gols_pro_m"], 1.25)

    def test_sem_vazamento(self):
        alterado = J.copy()
        alterado.loc[9, ["gols_m", "gols_v", "escanteios_m", "chutes_m", "posse_m"]] = [9, 9, 20, 30, 90]
        pd.testing.assert_frame_equal(janelas_times(alterado, janelas=(2,), k=2), self.j)

    def test_posse_so_a_favor(self):
        self.assertFalse(any("posse_contra" in c for c in self.j.columns))


if __name__ == "__main__":
    unittest.main()

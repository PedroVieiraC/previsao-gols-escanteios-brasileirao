import math
import unittest
from datetime import date, time

import pandas as pd

from src.features.historico import historico_times


def jogo(temp, d, m, v, gm, gv, em=4, ev=4):
    return {"temporada": temp, "data": d, "hora": time(16), "mandante": m, "visitante": v,
            "gols_m": gm, "gols_v": gv, "escanteios_m": em, "escanteios_v": ev}


# 2017: A, B, C. 2018: A e B continuam, D é promovido.
JOGOS = pd.DataFrame([
    jogo(2017, date(2017, 5, 1), "A", "B", 2, 0),
    jogo(2017, date(2017, 5, 8), "B", "C", 1, 1),
    jogo(2017, date(2017, 5, 15), "C", "A", 0, 3),
    jogo(2018, date(2018, 5, 1), "A", "D", 1, 1),
    jogo(2018, date(2018, 5, 8), "D", "B", 0, 2),
])


class HistoricoTest(unittest.TestCase):
    def setUp(self):
        self.h = historico_times(JOGOS, k=2)

    def test_primeira_rodada_da_base_sem_historico(self):
        self.assertTrue(math.isnan(self.h.loc[0, "hist_gols_pro_m"]))
        self.assertTrue(pd.isna(self.h.loc[0, "promovido_m"]))  # 2017: não se sabe

    def test_media_so_com_jogos_anteriores(self):
        # A no 3º jogo de 2017: 1 jogo antes (2 gols); 1ª temporada sem referência
        self.assertEqual(self.h.loc[2, "hist_gols_pro_v"], 2.0)
        self.assertEqual(self.h.loc[2, "hist_jogos_v"], 1)

    def test_promovido_e_referencia(self):
        # D promovido em 2018, sem promovidos antes: referência = média da liga em 2017 (7 gols / 6)
        self.assertTrue(self.h.loc[3, "promovido_v"])
        self.assertAlmostEqual(self.h.loc[3, "hist_gols_pro_v"], 7 / 6)
        self.assertFalse(self.h.loc[3, "promovido_m"])
        # A na 1ª de 2018: média final de A em 2017 = 2,5
        self.assertEqual(self.h.loc[3, "hist_gols_pro_m"], 2.5)

    def test_encolhimento(self):
        # D no 2º jogo de 2018: 1 jogo (1 gol), ref 7/6, k=2 -> (1*1 + 2*7/6)/3
        self.assertAlmostEqual(self.h.loc[4, "hist_gols_pro_m"], (1 + 2 * 7 / 6) / 3)

    def test_sem_vazamento(self):
        alterado = JOGOS.copy()
        alterado.loc[4, ["gols_m", "gols_v", "escanteios_m"]] = [9, 9, 20]
        pd.testing.assert_frame_equal(historico_times(alterado, k=2), self.h)


if __name__ == "__main__":
    unittest.main()

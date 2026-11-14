import unittest
from datetime import date, time

import pandas as pd

from src.features.gap_ratings import gap_ratings


def jogo(temp, d, m, v, em, ev):
    return {"temporada": temp, "data": d, "hora": time(16), "mandante": m, "visitante": v,
            "escanteios_m": em, "escanteios_v": ev}


# 2017: A x B (6 x 2), C x D (4 x 4), depois B x A (3 x 5). 2018: D cai, E sobe.
J = pd.DataFrame([
    jogo(2017, date(2017, 5, 1), "A", "B", 6, 2),
    jogo(2017, date(2017, 5, 1), "C", "D", 4, 4),
    jogo(2017, date(2017, 5, 8), "B", "A", 3, 5),
    jogo(2018, date(2018, 5, 1), "A", "E", 4, 4),
    jogo(2018, date(2018, 5, 8), "E", "C", 9, 1),
])
P = {"escanteios": (0.5, 0.5, 0.5)}


class GapRatingsTest(unittest.TestCase):
    def setUp(self):
        self.g = gap_ratings(J, fracoes=(1.0,), params=P)

    def test_primeira_rodada_sem_historico(self):
        self.assertTrue(pd.isna(self.g.loc[0, "gap100_escanteios_m"]))

    def test_esperado_apos_primeira_rodada(self):
        # Começo no observado: A (Ha=6, Hd=2, Aa=6, Ad=2), B (Aa=2, Ad=6, Ha=2, Hd=6).
        # Jogo 2, B em casa x A fora: esp_m = (Ha_B + Ad_A)/2 = (2 + 2)/2; esp_v = (Aa_A + Hd_B)/2 = (6 + 6)/2.
        self.assertEqual(self.g.loc[2, "gap100_escanteios_m"], 2.0)
        self.assertEqual(self.g.loc[2, "gap100_escanteios_v"], 6.0)
        self.assertEqual(self.g.loc[2, "gap100_escanteios_total"], 8.0)

    def test_atualizacao_e_promovido(self):
        # Jogo 2: em = 3 − 2 = 1; ev = 5 − 6 = −1. Com λ = φ = 0,5:
        # A (visitante): Aa = 6 + 0,25·(−1) = 5,75; Ha = 5,75; Ad = 2 + 0,25 = 2,25; Hd = 2,25.
        # B (mandante): Hd = 6 + 0,25·(−1) = 5,75; Ad = 5,75.
        # 2018, f = 1: E é promovido sem promovidos anteriores: média final da liga em cada índice.
        # Médias de 2017: Ad = (A 2,25 + B 5,75 + C 4 + D 4)/4 = 4. Jogo 3: esp_m = (Ha_A 5,75 + Ad_E 4)/2.
        self.assertAlmostEqual(self.g.loc[3, "gap100_escanteios_m"], (5.75 + 4) / 2)

    def test_jogo_simultaneo_nao_entra_na_media(self):
        # C x D começa no mesmo horário de A x B: sem jogos anteriores, fica NULL.
        self.assertTrue(pd.isna(self.g.loc[1, "gap100_escanteios_m"]))

    def test_sem_vazamento(self):
        alterado = J.copy()
        alterado.loc[4, ["escanteios_m", "escanteios_v"]] = [20, 0]
        pd.testing.assert_frame_equal(gap_ratings(alterado, fracoes=(1.0,), params=P), self.g)


if __name__ == "__main__":
    unittest.main()

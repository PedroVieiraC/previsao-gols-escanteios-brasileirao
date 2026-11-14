import unittest
from datetime import date, time

import pandas as pd

from src.features.pi_ratings import pi_ratings


def jogo(temp, d, m, v, gm, gv):
    return {"temporada": temp, "data": d, "hora": time(16), "mandante": m, "visitante": v, "gols_m": gm, "gols_v": gv}


# 2017: A e B (A vence 3x0 e depois 1x1 fora). 2018: B cai, C sobe.
J = pd.DataFrame([
    jogo(2017, date(2017, 5, 1), "A", "B", 3, 0),
    jogo(2017, date(2017, 5, 8), "B", "A", 1, 1),
    jogo(2018, date(2018, 5, 1), "A", "C", 0, 0),
    jogo(2018, date(2018, 5, 8), "C", "A", 2, 0),
])
LAM, GAM = 0.1, 0.5


class PiRatingsTest(unittest.TestCase):
    def setUp(self):
        self.p = pi_ratings(J, fracoes=(1.0,), lam=LAM, gam=GAM)

    def test_primeiro_jogo_comeca_em_zero(self):
        self.assertEqual(self.p.loc[0, "pi100_casa_m"], 0.0)
        self.assertEqual(self.p.loc[0, "pi100_fora_v"], 0.0)

    def test_atualizacao(self):
        # Jogo 0: e = 3. A: H = 0,3; A = 0,15. B: A = −0,3; H = −0,15.
        # Jogo 1 (B em casa, A fora): lê B_H = −0,15 e A_A = 0,15.
        self.assertAlmostEqual(self.p.loc[1, "pi100_casa_m"], -0.15)
        self.assertAlmostEqual(self.p.loc[1, "pi100_fora_v"], 0.15)
        self.assertAlmostEqual(self.p.loc[1, "pi100_dif"], -0.30)

    def test_fracao_na_virada_e_promovido(self):
        # Jogo 1: e = 0 − (−0,3) = 0,3. A_A = 0,15 − 0,03 = 0,12; A_H = 0,3 − 0,015 = 0,285.
        # 2018 com f = 0,5: A começa com H = 0,1425. C é promovido sem promovidos anteriores: 0.
        p = pi_ratings(J, fracoes=(0.5,), lam=LAM, gam=GAM)
        self.assertAlmostEqual(p.loc[2, "pi50_casa_m"], 0.1425)
        self.assertEqual(p.loc[2, "pi50_fora_v"], 0.0)

    def test_sem_vazamento(self):
        alterado = J.copy()
        alterado.loc[3, ["gols_m", "gols_v"]] = [9, 0]
        pd.testing.assert_frame_equal(pi_ratings(alterado, fracoes=(1.0,), lam=LAM, gam=GAM), self.p)


if __name__ == "__main__":
    unittest.main()

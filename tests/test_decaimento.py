import unittest

import pandas as pd

from src.features.decaimento import decaimento_times
from tests.test_janelas import J  # A alterna casa e fora contra B; em 2018, A marca 1, 2, 3, 4, 5, 6 gols

# Meia-vida 1: o peso cai à metade a cada jogo. Média de [x1, x2] = (x2 + 0,5·x1) / 1,5.


class DecaimentoTest(unittest.TestCase):
    def setUp(self):
        self.d = decaimento_times(J, meias_vidas=(1,), k=0)

    def test_peso_maior_no_jogo_recente(self):
        # 2018, 3º jogo (índice 6), A mandante: jogos anteriores de A = 1 e 2 gols
        self.assertAlmostEqual(self.d.loc[6, "d1_geral_gols_pro_m"], (2 + 0.5 * 1) / 1.5)

    def test_por_mando(self):
        # 2018, 5º jogo (índice 8), A em casa: jogos anteriores em casa = 1 e 3 gols
        self.assertAlmostEqual(self.d.loc[8, "d1_mando_gols_pro_m"], (3 + 0.5 * 1) / 1.5)

    def test_sem_jogos_vale_a_referencia(self):
        # 1º jogo de 2018 (índice 4): não usa 2017 como média, só como referência (A em 2017: 2, 1, 2, 0)
        self.assertEqual(self.d.loc[4, "d1_geral_gols_pro_m"], 1.25)

    def test_encolhe_para_a_referencia(self):
        d = decaimento_times(J, meias_vidas=(1,), k=2)
        media = (2 + 0.5 * 1) / 1.5
        self.assertAlmostEqual(d.loc[6, "d1_geral_gols_pro_m"], (2 * media + 2 * 1.25) / 4)

    def test_sem_vazamento(self):
        alterado = J.copy()
        alterado.loc[9, ["gols_m", "gols_v", "escanteios_m", "chutes_m", "posse_m"]] = [9, 9, 20, 30, 90]
        pd.testing.assert_frame_equal(decaimento_times(alterado, meias_vidas=(1,), k=0), self.d)


if __name__ == "__main__":
    unittest.main()

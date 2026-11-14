import unittest

import numpy as np
import pandas as pd

from src.modelos.referencias import analisar, media_corrente, referencias


class MediaCorrenteTest(unittest.TestCase):
    def test_nao_usa_a_mesma_data_nem_posteriores(self):
        data = np.array([1, 2, 2, 3])
        y = np.array([4.0, 10.0, 100.0, 1000.0])
        r = media_corrente(data, y, anterior=0.0, peso=1)
        # n = jogos de data < a sua: 0, 1, 1, 3 (os dois jogos do dia 2 não se veem).
        esperado = [0.0, 4.0 / 2, 4.0 / 2, (4 + 10 + 100) / 4]
        np.testing.assert_allclose(r, esperado)

    def test_perturbar_o_futuro_nao_muda(self):
        data, y = np.array([1, 2, 3, 3, 4]), np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        base = media_corrente(data, y, 2.0)
        y2 = y.copy()
        y2[2:] = 999  # tudo de data >= 3
        mudou = media_corrente(data, y2, 2.0)
        np.testing.assert_allclose(base[:2], mudou[:2])

    def test_ordem_das_linhas_nao_importa(self):
        data, y = np.array([3, 1, 2, 2]), np.array([9.0, 1.0, 5.0, 7.0])
        i = np.array([1, 2, 3, 0])
        np.testing.assert_allclose(media_corrente(data, y, 2.0)[i], media_corrente(data[i], y[i], 2.0))

    def test_sem_jogos_e_a_temporada_anterior(self):
        self.assertAlmostEqual(media_corrente(np.array([5]), np.array([3.0]), 2.7)[0], 2.7)


class AnalisarTest(unittest.TestCase):
    def test_referencias_e_alinhamento(self):
        jogos = pd.DataFrame({"temporada": [2018] * 2 + [2019] * 3, "data": [1, 2, 10, 11, 11],
                              "rodada": [1, 2, 1, 2, 2], "gols_m": [1, 3, 1, 0, 2], "gols_v": [1, 1, 0, 0, 2]})
        r = referencias(jogos, "gols", 2019, peso=2)
        self.assertEqual(list(r.ref_anterior), [3.0] * 3)  # (2 + 4)/2
        np.testing.assert_allclose(r.ref_corrente, [3.0, (1 + 2 * 3.0) / 3, (1 + 2 * 3.0) / 3])
        prev = pd.DataFrame({"alvo": "gols", "teste": 2019, "rodada": [1, 2, 2], "y": [1, 0, 4],
                             **{c: 2.0 for c in ("ingenua", "poisson_ridge", "random_forest", "xgboost", "media_arvores")}})
        self.assertEqual(len(analisar(prev, jogos)), 2 * (5 + 1))


if __name__ == "__main__":
    unittest.main()

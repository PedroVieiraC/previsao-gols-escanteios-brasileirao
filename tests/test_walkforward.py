import unittest

import numpy as np
import pandas as pd

from src.modelos.walkforward import _binomial_negativa, _poisson, configuracoes, dobra


class WalkForwardTest(unittest.TestCase):
    def test_dobra_nao_ve_o_teste(self):
        temporadas = pd.Series([2017, 2018, 2019, 2020, 2021])
        m = dobra(temporadas, 2020)
        self.assertEqual(list(temporadas[m["treino"]]), [2017, 2018, 2019])
        self.assertEqual(list(temporadas[m["ajuste"]]), [2017, 2018])
        self.assertEqual(list(temporadas[m["validacao"]]), [2019])
        self.assertEqual(list(temporadas[m["teste"]]), [2020])
        self.assertFalse((m["treino"] & m["teste"]).any())

    def test_alpha_da_binomial_negativa(self):
        # Dados NB2 simulados com α = 0,1 e média 10: a regressão auxiliar recupera α.
        rng = np.random.default_rng(0)
        x = pd.DataFrame({"x": rng.normal(size=20000)})
        mu = np.exp(np.log(10) + 0.2 * x["x"])
        y = pd.Series(rng.negative_binomial(n=1 / 0.1, p=1 / (1 + 0.1 * mu)))
        _, alpha = _binomial_negativa(x, y, _poisson(x, y))
        self.assertAlmostEqual(alpha, 0.1, delta=0.01)

    def test_validacoes_da_rodada2_param_antes_do_teste(self):
        from src.modelos.rodada2 import validacoes
        self.assertEqual(validacoes(2019), [2018])
        self.assertEqual(validacoes(2022), [2018, 2019, 2020, 2021])

    def test_bootstrap_da_diferenca(self):
        from src.modelos.rodada2 import bootstrap_diferenca
        y = np.array([1, 2, 3, 2, 1, 4] * 50)
        media, inf, sup = bootstrap_diferenca(y, y + 0.01, np.full(len(y), y.mean()))
        self.assertLess(sup, 0)  # prever quase o próprio valor é melhor que a média
        self.assertLess(inf, media)

    def test_formacao_so_nos_escanteios(self):
        self.assertEqual({c["formacao"] for c in configuracoes("gols")}, {"sem"})
        self.assertEqual({c["formacao"] for c in configuracoes("escanteios")}, {"sem", "familia", "meio_frente"})


if __name__ == "__main__":
    unittest.main()

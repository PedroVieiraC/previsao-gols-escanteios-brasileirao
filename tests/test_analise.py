import unittest

import numpy as np
import pandas as pd
from scipy import stats

from src.modelos.analise import (alpha_momentos, bootstrap_blocos, brier_jogo, com_rodada, comparacao, deviance_jogo,
                                 grupo, importancia_permutacao, log_loss_jogo, prob_acima)


class BootstrapBlocosTest(unittest.TestCase):
    def test_media_e_ic(self):
        d = np.r_[np.full(50, -1.0), np.full(50, -0.5)]
        media, inf, sup = bootstrap_blocos(d, np.repeat(np.arange(10), 10))
        self.assertAlmostEqual(media, -0.75)
        self.assertTrue(-1 <= inf <= media <= sup < 0)

    def test_blocos_correlacionados_alargam_o_ic(self):
        # Mesmo d por jogo; com blocos que agrupam valores parecidos, o IC fica mais largo que por jogo.
        rng = np.random.default_rng(1)
        d = np.repeat(rng.normal(size=40), 10)
        _, i1, s1 = bootstrap_blocos(d, np.arange(400))
        _, i2, s2 = bootstrap_blocos(d, np.repeat(np.arange(40), 10))
        self.assertGreater(s2 - i2, 2 * (s1 - i1))

    def test_comparacao_contra_referencia(self):
        prev = pd.DataFrame({"alvo": "gols", "teste": 2019, "rodada": np.repeat(np.arange(20), 10),
                             "y": np.tile([1, 2, 3, 2, 1], 40)})
        prev["ingenua"] = prev.y.mean()
        prev["bom"] = prev.y + 0.01
        c = comparacao(prev, {"bom": "bom"}).iloc[0]
        self.assertLess(c.ic_sup, 0)

    def test_com_rodada_confere_y(self):
        jogos = pd.DataFrame({"temporada": [2019] * 3, "rodada": [1, 1, 2], "gols_m": [1, 0, 2], "gols_v": [0, 0, 1]})
        prev = pd.DataFrame({"alvo": "gols", "teste": 2019, "y": [1, 0, 3]})
        self.assertEqual(list(com_rodada(prev, jogos).rodada), [1, 1, 2])
        with self.assertRaises(AssertionError):
            com_rodada(prev.assign(y=[1, 1, 3]), jogos)


class ImportanciaTest(unittest.TestCase):
    def test_so_a_coluna_usada_importa(self):
        rng = np.random.default_rng(0)
        X = pd.DataFrame({"pi70_dif": rng.normal(size=500), "dias_descanso_m": rng.normal(size=500)})
        y = rng.poisson(np.exp(1 + 0.5 * X.pi70_dif))
        imp = importancia_permutacao(lambda Z: np.exp(1 + 0.5 * Z.pi70_dif.to_numpy()), X, y, n=5)
        col = imp[imp.tipo == "coluna"].set_index("nome").dif_deviance
        self.assertGreater(col["pi70_dif"], 0.05)
        self.assertEqual(col["dias_descanso_m"], 0)
        self.assertEqual(set(imp[imp.tipo == "grupo"].nome), {"pi-rating", "descanso"})

    def test_grupos(self):
        self.assertEqual(grupo("familia_A_m"), "formação")
        self.assertEqual(grupo("fam_mando_escanteios_esp_m"), "formação")
        self.assertEqual(grupo("meio_frente_v"), "formação")
        self.assertEqual(grupo("d5_geral_gols_esp_m"), "forma recente")
        self.assertEqual(grupo("hist_gols_esp_v"), "média da temporada")
        self.assertEqual(grupo("gap70_gols_m"), "GAP")


class ProbabilidadeTest(unittest.TestCase):
    def test_poisson_e_nb(self):
        self.assertAlmostEqual(prob_acima([2.5], 2.5)[0], 1 - stats.poisson.cdf(2, 2.5))
        # α → 0: a NB vira Poisson; α > 0 põe mais massa nas caudas.
        self.assertAlmostEqual(prob_acima([10.0], 9.5, 1e-8)[0], prob_acima([10.0], 9.5)[0], places=5)
        self.assertGreater(prob_acima([10.0], 15.5, 0.1)[0], prob_acima([10.0], 15.5)[0])

    def test_alpha_momentos(self):
        rng = np.random.default_rng(0)
        mu = np.full(20000, 10.0)
        y = rng.negative_binomial(10, 0.5, size=20000)  # média 10, α = 1/10
        self.assertAlmostEqual(alpha_momentos(y, mu), 0.1, delta=0.01)
        self.assertEqual(alpha_momentos(rng.poisson(mu) * 0 + 10, mu), 1e-8)  # subdisperso: piso

    def test_perdas(self):
        o = np.array([1.0, 0.0])
        self.assertAlmostEqual(log_loss_jogo(o, [0.5, 0.5]).mean(), np.log(2))
        self.assertAlmostEqual(brier_jogo(o, [1.0, 0.0]).sum(), 0)
        self.assertGreater(deviance_jogo([0, 3], [1, 1]).min(), 0)


if __name__ == "__main__":
    unittest.main()

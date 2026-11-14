import unittest

import numpy as np

from src.modelos import conjunto as cj


class ConjuntoTest(unittest.TestCase):
    def test_grade_no_simplex(self):
        g = cj.grade_pesos(4)
        self.assertEqual(len(g), 286)
        self.assertTrue(all(abs(w.sum() - 1) < 1e-9 and (w >= 0).all() for w in g))

    def test_acha_o_componente_certo(self):
        rng = np.random.default_rng(0)
        mu = rng.uniform(1, 4, 500)
        y = rng.poisson(mu).astype(float)
        M = np.column_stack([np.full(500, y.mean()), mu, mu * 1.5])
        w = cj.melhores_pesos([(y, M)])
        self.assertGreaterEqual(w[1], 0.7)


if __name__ == "__main__":
    unittest.main()

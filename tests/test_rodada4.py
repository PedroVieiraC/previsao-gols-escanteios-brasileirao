import unittest

import pandas as pd

from src.modelos import rodada4 as r4


class Rodada4Test(unittest.TestCase):
    def test_grades(self):
        # Padrão: a 4c (240 combinações, sementes 0 e 1); a 4d (480) só com TCC_ETAPA=rodada4d.
        self.assertEqual(r4.ETAPA, "rodada4c")
        self.assertEqual(len(r4.GRADE_XGB), 240)
        self.assertEqual(len({tuple(g.values()) for g in r4.GRADE_XGB}), 240)
        self.assertEqual(len(r4._amostra_xgb()), 480)
        # As 240 da 4d têm ao menos um valor fora do espaço das rodadas anteriores.
        self.assertTrue(all(c[4] == 0.1 or c[5] == 30.0 or c[6] == 0.3 for c in r4._amostra_xgb()[240:]))
        # As 120 primeiras são as das rodadas 4 e 4b, que o cache reaproveita.
        self.assertEqual(r4.GRADE_XGB[:120], [dict(zip(r4.GRADE_XGB[0], c)) for c in
                                              __import__("random").Random(0).sample(r4._ESPACO_XGB, 120)])
        self.assertLessEqual(max(r4.ARVORES_XGB), r4.modelo_arvore("xgboost", r4.GRADE_XGB[0]).n_estimators)
        self.assertEqual(set(r4.KS), {0, 2, 5, 10, 20, 40, 80, 160, 320, 1000})
        self.assertEqual(len(r4.GRADE_RF), 7 * 6 * 6)

    def test_ajuste_nunca_ve_a_temporada_proibida(self):
        temp = pd.Series([2017, 2018, 2019])
        r4._checar_sem_teste(temp, temp < 2019, 2019)
        with self.assertRaises(AssertionError):
            r4._checar_sem_teste(temp, temp <= 2019, 2019)

    def test_no_limite(self):
        esc = pd.DataFrame([{"alvo": "gols", "teste": 2019, "modelo": "random_forest", "min_samples_leaf": 300,
                             "max_features": 0.2, "max_depth": None},
                            {"alvo": "gols", "teste": 2019, "modelo": "poisson_ridge", "penalidade": 1.0}])
        self.assertEqual(r4.no_limite(esc), ["gols 2019 random_forest: min_samples_leaf = 300.0 no limite"])

    def test_executar_grava_e_retoma(self):
        import math
        import tempfile
        from pathlib import Path
        from unittest import mock

        with tempfile.TemporaryDirectory() as d, mock.patch.object(r4, "CACHE", Path(d)), \
                mock.patch.object(r4, "CONTROLE", Path(d) / "processos.txt"):
            tarefas = {i: (lambda i=i: (math.sqrt, (i,))) for i in range(12)}
            self.assertEqual(r4.executar(tarefas, "t"), {i: math.sqrt(i) for i in range(12)})
            # Na retomada nada é recalculado: uma tarefa que falharia não chega a ser enviada.
            tarefas[0] = lambda: (math.sqrt, (-1,))
            self.assertEqual(r4.executar(tarefas, "t")[0], 0.0)


if __name__ == "__main__":
    unittest.main()

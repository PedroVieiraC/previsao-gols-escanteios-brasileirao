import unittest
from datetime import date, time, timedelta

import pandas as pd

from src.features.formacao import familia, formacao_features


def jogo(i, m, v, fm, fv, em, ev):
    return {"temporada": 2018, "data": date(2018, 5, 1) + timedelta(days=7 * i), "hora": time(16),
            "mandante": m, "visitante": v, "formacao_m": fm, "formacao_v": fv, "escanteios_m": em, "escanteios_v": ev}


# A em casa: 4-2-3-1 (B) com 6 e depois 8 escanteios; no 3º jogo, 3-5-2 (A). C sempre 4-4-2 (A).
J = pd.DataFrame([
    jogo(0, "A", "C", "4-2-3-1", "4-4-2", 6, 2),
    jogo(1, "C", "A", "4-4-2", "4-3-3", 4, 3),
    jogo(2, "A", "C", "4-2-3-1", "4-4-2", 8, 1),
    jogo(3, "A", "C", "3-5-2", "4-4-2", 5, 5),
])


class FormacaoTest(unittest.TestCase):
    def setUp(self):
        self.f = formacao_features(J, k=0)

    def test_familia_e_meio_frente(self):
        self.assertEqual(self.f.loc[1, "familia_v"], "B")
        self.assertEqual(self.f.loc[3, "familia_m"], "A")
        self.assertEqual(self.f.loc[3, "meio_frente_m"], 7)
        self.assertEqual(self.f.loc[0, "meio_frente_m"], 6)

    def test_historico_na_mesma_familia(self):
        # Jogo 2: A na família B; jogos anteriores de A na família B: 6 (jogo 0) e 3 (jogo 1, 4-3-3 fora).
        self.assertEqual(self.f.loc[2, "fam_geral_escanteios_pro_m"], 4.5)
        # Só em casa: 6.
        self.assertEqual(self.f.loc[2, "fam_mando_escanteios_pro_m"], 6.0)

    def test_sem_historico_na_familia_vale_a_referencia(self):
        # Jogo 3: A estreia na família A; referência = média da liga na família A antes do jogo
        # (C: 2, 4, 1) = 7/3.
        self.assertAlmostEqual(self.f.loc[3, "fam_geral_escanteios_pro_m"], 7 / 3)

    def test_familia_inedita_na_liga_vale_a_media_da_liga(self):
        # Jogo 1: A fora em 4-3-3 (família B); no mando "fora", a liga só tem C no jogo 0, família A (2).
        # Sem B fora antes: referência = média da liga fora = 2.
        self.assertEqual(self.f.loc[1, "fam_mando_escanteios_pro_v"], 2.0)

    def test_sem_vazamento(self):
        alterado = J.copy()
        alterado.loc[3, ["escanteios_m", "escanteios_v"]] = [20, 0]
        pd.testing.assert_frame_equal(formacao_features(alterado, k=0), self.f)

    def test_formacao_desconhecida_falha(self):
        with self.assertRaises(ValueError):
            familia(pd.Series(["4-2-3-1", "2-3-5"]))


if __name__ == "__main__":
    unittest.main()

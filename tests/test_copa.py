import unittest
from datetime import date, time

import pandas as pd

from src.features.copa import copa_entre_rodadas

JOGOS = pd.DataFrame([
    {"temporada": 2024, "data": date(2024, 5, 1), "hora": time(16), "mandante": "A", "visitante": "B"},
    {"temporada": 2024, "data": date(2024, 5, 8), "hora": time(16), "mandante": "B", "visitante": "A"},
    {"temporada": 2024, "data": date(2024, 5, 15), "hora": time(16), "mandante": "A", "visitante": "B"},
])
COPA = pd.DataFrame([
    # A joga a Copa no meio de semana entre a 1ª e a 2ª rodada
    {"data": date(2024, 5, 4), "hora": time(21), "mandante": "A", "visitante": None},
    # B joga a Copa no mesmo dia da 3ª rodada, mas depois: não conta para a 3ª
    {"data": date(2024, 5, 15), "hora": time(21), "mandante": None, "visitante": "B"},
])


class CopaFeatureTest(unittest.TestCase):
    def setUp(self):
        self.c = copa_entre_rodadas(JOGOS, COPA)

    def test_primeiro_jogo_da_temporada_e_nulo(self):
        self.assertTrue(pd.isna(self.c.loc[0, "copa_entre_rodadas_m"]))

    def test_copa_entre_rodadas(self):
        self.assertTrue(self.c.loc[1, "copa_entre_rodadas_v"])   # A, 2ª rodada
        self.assertFalse(self.c.loc[1, "copa_entre_rodadas_m"])  # B não jogou
        self.assertFalse(self.c.loc[2, "copa_entre_rodadas_m"])  # A, 3ª rodada: Copa foi antes da 2ª

    def test_copa_depois_do_jogo_alvo_nao_conta(self):
        self.assertFalse(self.c.loc[2, "copa_entre_rodadas_v"])


if __name__ == "__main__":
    unittest.main()

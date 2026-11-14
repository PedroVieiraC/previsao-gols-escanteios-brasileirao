import unittest
from datetime import date, time

import pandas as pd

from src.features.calendario import DESCANSO_MAX, dias_descanso, sem_publico

JOGOS = pd.DataFrame([
    {"temporada": 2024, "data": date(2024, 5, 1), "hora": time(16), "mandante": "A", "visitante": "B"},
    {"temporada": 2024, "data": date(2024, 5, 8), "hora": time(16), "mandante": "B", "visitante": "A"},
    {"temporada": 2024, "data": date(2024, 6, 15), "hora": time(16), "mandante": "A", "visitante": "B"},
])
COPA = pd.DataFrame([
    {"data": date(2024, 4, 25), "hora": time(21), "mandante": "A", "visitante": None},  # 5,8 dias antes do 1º
    {"data": date(2024, 5, 4), "hora": time(21), "mandante": None, "visitante": "B"},
    {"data": date(2024, 6, 15), "hora": time(21), "mandante": "A", "visitante": None},  # depois do jogo-alvo
])


class CalendarioTest(unittest.TestCase):
    def setUp(self):
        self.d = dias_descanso(JOGOS, COPA)

    def test_descanso_conta_a_copa(self):
        self.assertAlmostEqual(self.d.loc[1, "dias_descanso_m"], 3 + 19 / 24)  # B: Copa 04/05 21h -> 08/05 16h
        self.assertAlmostEqual(self.d.loc[1, "dias_descanso_v"], 7.0)          # A: Série A 01/05 -> 08/05

    def test_primeiro_jogo_e_truncagem(self):
        self.assertAlmostEqual(self.d.loc[0, "dias_descanso_m"], 5 + 19 / 24)  # A teve Copa antes do 1º jogo
        self.assertEqual(self.d.loc[0, "dias_descanso_v"], DESCANSO_MAX)       # B sem jogo anterior
        self.assertEqual(self.d.loc[2, "dias_descanso_m"], DESCANSO_MAX)       # 38 dias: truncado
        self.assertFalse(self.d.isna().any().any())

    def test_sem_vazamento(self):
        # jogo da Copa no mesmo dia, depois do jogo-alvo, e jogos futuros não mudam o valor
        futuro = pd.concat([JOGOS, pd.DataFrame([{"temporada": 2024, "data": date(2024, 6, 14), "hora": time(23),
                                                   "mandante": "C", "visitante": "D"}])], ignore_index=True)
        self.assertTrue(dias_descanso(futuro, COPA).loc[:2].equals(self.d))
        sem_copa_futura = dias_descanso(JOGOS, COPA.iloc[:2])
        self.assertTrue(sem_copa_futura.equals(self.d))

    def test_sem_publico(self):
        j = pd.DataFrame({"temporada": [2019, 2020, 2020, 2021, 2021, 2022],
                          "data": [date(2019, 9, 1), date(2020, 8, 8), date(2021, 2, 25), date(2021, 10, 1),
                                   date(2021, 10, 2), date(2022, 4, 9)],
                          "hora": [time(16)] * 6})
        self.assertEqual(sem_publico(j).tolist(), [False, True, True, True, False, False])


if __name__ == "__main__":
    unittest.main()

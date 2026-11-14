import unittest
from datetime import date, time, timedelta

import pandas as pd

from src.features.tecnicos import tecnico_features

DEPARA = pd.DataFrame({
    "tecnico_original": ["A", "B", "C", "X", "Y"],
    "clube": ["T", "T", "T", "U", "U"],
    "pessoa_id": [1, 2, 3, 9, pd.NA],
    "nome_padrao": ["a", "b", "c", "x", "y"],
})


def jogos(tecnicos_t, temporadas=None, inicio=date(2018, 5, 1)):
    """Clube T (mandante) contra U; um jogo por semana."""
    temporadas = temporadas or [2018] * len(tecnicos_t)
    return pd.DataFrame({"temporada": temporadas,
                         "data": [inicio + timedelta(weeks=i) for i in range(len(tecnicos_t))],
                         "hora": [time(16)] * len(tecnicos_t), "mandante": "T", "visitante": "U",
                         "tecnico_m": tecnicos_t, "tecnico_v": "X"})


class TecnicoFeaturesTest(unittest.TestCase):
    def test_interino_de_um_jogo_nao_e_troca(self):
        f = tecnico_features(jogos(list("AABAA")), DEPARA)
        self.assertTrue(f["tecnico_jogos_m"].isna().all())  # início do trabalho fora da base
        self.assertEqual(f["tecnico_trocas_temp_m"].tolist(), [0, 0, 0, 0, 0])

    def test_troca_reconhecida_no_segundo_jogo(self):
        f = tecnico_features(jogos(list("AABCCC")), DEPARA)  # A-B-C: B interino, C técnico
        self.assertEqual(f["tecnico_trocas_temp_m"].tolist(), [0, 0, 0, 0, 1, 1])
        self.assertEqual(f["tecnico_jogos_m"].tolist()[4:], [1, 2])  # C desde o jogo 3
        self.assertTrue(f["tecnico_jogos_m"].iloc[:4].isna().all())

    def test_sem_vazamento(self):
        base = tecnico_features(jogos(list("AABB")), DEPARA)
        futuro = tecnico_features(jogos(list("AABBCCA")), DEPARA)
        pd.testing.assert_frame_equal(futuro.iloc[:4], base)
        # o jogo de estreia do técnico novo não vê a troca: só o 2º jogo seguido
        self.assertEqual(base["tecnico_trocas_temp_m"].tolist(), [0, 0, 0, 1])

    def test_temporada_zera_trocas_e_volta_da_serie_b_zera_o_historico(self):
        f = tecnico_features(jogos(list("ABBBBC"), [2018, 2018, 2018, 2019, 2021, 2021]), DEPARA)
        self.assertEqual(f["tecnico_trocas_temp_m"].tolist(), [0, 0, 1, 0, 0, 0])
        self.assertEqual(f["tecnico_jogos_m"].tolist()[:4], [pd.NA, pd.NA, 1, 2])  # continua entre 2018 e 2019
        self.assertTrue(pd.isna(f["tecnico_jogos_m"].iloc[4]))  # 2020 fora: início desconhecido

    def test_sem_id_usa_nome_padrao(self):
        j = jogos(list("AAAA")).assign(tecnico_v=["X", "X", "Y", "Y"])
        self.assertEqual(tecnico_features(j, DEPARA)["tecnico_trocas_temp_v"].tolist(), [0, 0, 0, 1])

    def test_tecnico_sem_de_para_falha(self):
        with self.assertRaises(ValueError):
            tecnico_features(jogos(["A", "Z"]), DEPARA)


if __name__ == "__main__":
    unittest.main()

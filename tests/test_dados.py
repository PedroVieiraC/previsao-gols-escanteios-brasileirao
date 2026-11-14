import unittest

from src.features.historico import carregar_canonica, ler_tabela


class DadosTest(unittest.TestCase):
    def test_canonica(self):
        j = carregar_canonica()
        self.assertEqual(len(j), 3040)
        self.assertTrue((j.groupby("temporada").size() == 380).all())
        self.assertFalse(j.duplicated(["data", "mandante", "visitante"]).any())

    def test_copa_e_depara_carregam(self):
        self.assertGreater(len(ler_tabela("norm_copa_brasil")), 0)
        self.assertEqual(len(ler_tabela("tb_tecnicos_depara")), 395)


if __name__ == "__main__":
    unittest.main()

import os
import unittest

from src.features.historico import carregar_canonica, ler_tabela

DATA = os.path.join(os.path.dirname(__file__), "..", "data")


class DadosTest(unittest.TestCase):
    def test_canonica(self):
        j = carregar_canonica()
        self.assertEqual(len(j), 3040)
        self.assertTrue((j.groupby("temporada").size() == 380).all())
        self.assertFalse(j.duplicated(["data", "mandante", "visitante"]).any())

    @unittest.skipUnless(os.path.exists(os.path.join(DATA, "norm_copa_brasil.csv")), "CSV local não versionado")
    def test_copa_carrega(self):
        self.assertGreater(len(ler_tabela("norm_copa_brasil")), 0)

    @unittest.skipUnless(os.path.exists(os.path.join(DATA, "tb_tecnicos_depara.csv")), "CSV local não versionado")
    def test_depara_carrega(self):
        self.assertEqual(len(ler_tabela("tb_tecnicos_depara")), 395)


if __name__ == "__main__":
    unittest.main()

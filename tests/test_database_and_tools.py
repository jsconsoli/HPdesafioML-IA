"""Testes sem dependência de Ollama, rede ou Streamlit."""

import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from gpu_assistant.database import GpuDatabase
from gpu_assistant.tools import GpuQueryTools


class DatabaseToolsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        db_path = Path(self.temp.name) / "gpus.sqlite3"
        self.database = GpuDatabase(db_path)
        self.database.initialize_db()
        self.query_tools = GpuQueryTools(self.database)

    def test_initial_data(self):
        self.assertEqual(len(self.database.list_gpus()), 5)
        self.assertEqual(self.database.get_summary(), {"modelos": 5, "total_disponivel": 19})
        self.assertEqual(self.database.get_gpu("A100")["memoria_gb"], 80)
        self.assertEqual(self.database.get_gpu("4090")["quantidade_disponivel"], 3)

    def test_names_and_multiple_gpu_queries(self):
        result = self.query_tools.consultar_gpus(["a100", "RTX 4090", "nvidia a100", "V100"])
        self.assertEqual(len(result["gpus"]), 2)
        self.assertEqual(result["modelos_nao_encontrados"], ["V100"])
        self.assertEqual(len(self.query_tools.consultar_gpus()["gpus"]), 5)
        self.assertEqual(
            self.query_tools.consultar_gpus(modelo="L40S"), self.query_tools.consultar_gpus(["L40S"])
        )
        self.assertIn("erro", self.query_tools.consultar_gpus(["A100"], modelo="L40S"))

    def test_availability_boolean_and_unknown(self):
        self.assertTrue(self.query_tools.verificar_disponibilidade("A100", 2)["atende_solicitacao"])
        self.assertFalse(self.query_tools.verificar_disponibilidade("H100", 3)["atende_solicitacao"])
        self.assertEqual(self.query_tools.verificar_disponibilidade("V100", 1), {"modelo_nao_encontrado": "V100"})
        self.assertEqual(self.query_tools.verificar_disponibilidade("48 GB", 1), {"modelo_nao_encontrado": "48 GB"})
        self.assertEqual(
            self.query_tools.verificar_disponibilidade("A100", 0),
            {"erro": "O argumento quantidade deve ser maior ou igual a 1."},
        )

    def test_memory_filters(self):
        exact = self.query_tools.consultar_gpus(memoria_exata_gb=48)
        self.assertEqual([gpu["modelo"] for gpu in exact["gpus"]], ["NVIDIA A40", "NVIDIA L40S"])
        minimum = self.query_tools.consultar_gpus(memoria_minima_gb=48)
        self.assertEqual(
            [gpu["modelo"] for gpu in minimum["gpus"]],
            ["NVIDIA A100", "NVIDIA A40", "NVIDIA H100", "NVIDIA L40S"],
        )
        combined = self.query_tools.consultar_gpus(["A100", "L40S"], memoria_exata_gb=48)
        self.assertEqual([gpu["modelo"] for gpu in combined["gpus"]], ["NVIDIA L40S"])
        self.assertEqual(self.query_tools.consultar_gpus(memoria_exata_gb=47), {"gpus": []})
        self.assertEqual(
            self.query_tools.consultar_gpus(["48 GB"]), {"gpus": [], "modelos_nao_encontrados": ["48 GB"]}
        )

    def test_quantity_comparison_includes_ties_and_respects_availability_scope(self):
        minimum = self.query_tools.comparar_gpus("menos_disponiveis")
        self.assertEqual(
            [(gpu["modelo"], gpu["quantidade_disponivel"]) for gpu in minimum["gpus"]],
            [("NVIDIA H100", 2)],
        )
        maximum = self.query_tools.comparar_gpus("mais_disponiveis")
        self.assertEqual(
            [(gpu["modelo"], gpu["quantidade_disponivel"]) for gpu in maximum["gpus"]],
            [("NVIDIA L40S", 6)],
        )

        self.database.update_availability("A100", 0)
        self.database.update_availability("A40", 0)
        self.database.update_availability("RTX 4090", 6)
        minimum = self.query_tools.comparar_gpus("menos_disponiveis")
        self.assertEqual([gpu["modelo"] for gpu in minimum["gpus"]], ["NVIDIA A100", "NVIDIA A40"])
        minimum_available = self.query_tools.comparar_gpus("menos_disponiveis", somente_disponiveis=True)
        self.assertEqual([gpu["modelo"] for gpu in minimum_available["gpus"]], ["NVIDIA H100"])
        maximum = self.query_tools.comparar_gpus("mais_disponiveis")
        self.assertEqual([gpu["modelo"] for gpu in maximum["gpus"]], ["NVIDIA L40S", "NVIDIA RTX 4090"])

    def test_rejects_invalid_memory_arguments(self):
        for arguments in (
            {"modelos": '["48GB"]'},
            {"memoria_exata_gb": "48"},
            {"memoria_minima_gb": True},
            {"memoria_minima_gb": 0},
            {"memoria_exata_gb": 48, "memoria_minima_gb": 48},
        ):
            with self.subTest(arguments=arguments):
                self.assertIn("erro", self.query_tools.consultar_gpus(**arguments))

    def test_external_update_and_summary(self):
        updated = self.database.update_availability("A100", 2)
        self.assertEqual(updated["quantidade_disponivel"], 2)
        self.assertEqual(self.query_tools.consultar_resumo()["total_disponivel"], 17)
        self.assertEqual(self.database.get_gpu("A100")["memoria_gb"], 80)
        self.assertEqual(self.database.get_gpu("A100")["quantidade_disponivel"], 2)

    def test_reinitialize_preserves_updates_unless_reset(self):
        self.database.update_availability("A100", 0)
        self.database.initialize_db()
        self.assertEqual(self.database.get_gpu("A100")["quantidade_disponivel"], 0)
        self.database.initialize_db(reset_availability=True)
        self.assertEqual(self.database.get_gpu("A100")["quantidade_disponivel"], 4)

    def test_invalid_writes_are_rejected(self):
        with self.assertRaises(ValueError):
            self.database.update_availability("A100", -1)
        with self.assertRaises(ValueError):
            self.database.update_availability("V100", 2)
        self.assertEqual(self.database.get_summary()["total_disponivel"], 19)

    def test_database_connection_is_read_only(self):
        with closing(self.database.connect()) as conn:
            with self.assertRaises(sqlite3.OperationalError):
                conn.execute("UPDATE gpus SET quantidade_disponivel = 0")

    def test_query_tool_instances_isolate_databases_and_schemas(self):
        other_database = GpuDatabase(Path(self.temp.name) / "other.sqlite3")
        other_database.initialize_db()
        other_tools = GpuQueryTools(other_database)

        self.database.update_availability("A100", 2)
        self.assertEqual(self.query_tools.consultar_resumo()["total_disponivel"], 17)
        self.assertEqual(other_tools.consultar_resumo()["total_disponivel"], 19)

        def criteria(query_tools):
            return query_tools.schemas["comparar_gpus"]["parameters"]["properties"]["criterio"]["enum"]

        original_criteria = criteria(other_tools).copy()
        criteria(self.query_tools).append("criterio_local")
        self.assertEqual(criteria(other_tools), original_criteria)


if __name__ == "__main__":
    unittest.main()


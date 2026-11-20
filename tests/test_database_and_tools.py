"""Testes sem dependência de Ollama, rede ou Streamlit."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from gpu_assistant.database import (
    get_gpu,
    get_summary,
    initialize_db,
    list_gpus,
    update_availability,
)
from gpu_assistant.tools import (
    consultar_gpus,
    consultar_resumo,
    verificar_disponibilidade,
)


class DatabaseToolsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        db_path = Path(self.temp.name) / "gpus.sqlite3"
        self.env_patch = patch.dict(os.environ, {"GPU_DB_PATH": str(db_path)})
        self.env_patch.start()
        self.addCleanup(self.env_patch.stop)
        initialize_db()

    def test_initial_data(self):
        self.assertEqual(len(list_gpus()), 5)
        self.assertEqual(get_summary(), {"modelos": 5, "total_disponivel": 19})
        self.assertEqual(get_gpu("A100")["memoria_gb"], 80)
        self.assertEqual(get_gpu("4090")["quantidade_disponivel"], 3)

    def test_names_and_multiple_gpu_queries(self):
        result = consultar_gpus(["a100", "RTX 4090", "nvidia a100", "V100"])
        self.assertEqual(len(result["gpus"]), 2)
        self.assertEqual(result["modelos_nao_encontrados"], ["V100"])
        self.assertEqual(len(consultar_gpus()["gpus"]), 5)

    def test_availability_boolean_and_unknown(self):
        self.assertTrue(verificar_disponibilidade("A100", 2)["atende_solicitacao"])
        self.assertFalse(verificar_disponibilidade("H100", 3)["atende_solicitacao"])
        self.assertIn("erro", verificar_disponibilidade("V100", 1))
        self.assertIn("erro", verificar_disponibilidade("A100", -1))

    def test_external_update_and_summary(self):
        updated = update_availability("A100", 2)
        self.assertEqual(updated["quantidade_disponivel"], 2)
        self.assertEqual(consultar_resumo()["total_disponivel"], 17)
        self.assertEqual(get_gpu("A100")["memoria_gb"], 80)
        self.assertEqual(get_gpu("A100")["quantidade_disponivel"], 2)

    def test_reinitialize_preserves_updates_unless_reset(self):
        update_availability("A100", 0)
        initialize_db()
        self.assertEqual(get_gpu("A100")["quantidade_disponivel"], 0)
        initialize_db(reset_availability=True)
        self.assertEqual(get_gpu("A100")["quantidade_disponivel"], 4)

    def test_invalid_writes_are_rejected(self):
        with self.assertRaises(ValueError):
            update_availability("A100", -1)
        with self.assertRaises(ValueError):
            update_availability("V100", 2)
        self.assertEqual(get_summary()["total_disponivel"], 19)

    def test_database_connection_is_read_only(self):
        from gpu_assistant.database import connect

        with connect() as conn:
            with self.assertRaises(Exception):
                conn.execute("UPDATE gpus SET quantidade_disponivel = 0")


if __name__ == "__main__":
    unittest.main()


"""Simula respostas de um LLM para validar o ciclo de Tool Calling."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from gpu_assistant.agent import GpuAgent
from gpu_assistant.database import initialize_db


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        return {"message": self.responses.pop(0)}


class AgentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        db_path = Path(self.temp.name) / "gpus.sqlite3"
        self.env_patch = patch.dict(os.environ, {"GPU_DB_PATH": str(db_path)})
        self.env_patch.start()
        self.addCleanup(self.env_patch.stop)
        initialize_db()

    def test_tool_call_uses_db_and_variable_model(self):
        fake = FakeClient(
            [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "verificar_disponibilidade",
                                "arguments": {"modelo": "A100", "quantidade": 2},
                            }
                        }
                    ],
                },
                {"role": "assistant", "content": "Sim, há quatro A100 disponíveis."},
            ]
        )
        agent = GpuAgent(client=fake, model_name="modelo-de-teste:1b")
        answer, history = agent.answer("Existem duas A100 disponíveis?")

        self.assertIn("quatro", answer)
        self.assertEqual(len(fake.calls), 2)
        self.assertTrue(all(call["model"] == "modelo-de-teste:1b" for call in fake.calls))
        self.assertEqual({tool.__name__ for tool in fake.calls[0]["tools"]}, {
            "consultar_gpus", "verificar_disponibilidade", "consultar_resumo"
        })
        result = next(message for message in history if message["role"] == "tool")
        self.assertTrue(json.loads(result["content"])["atende_solicitacao"])

    def test_refuses_unverified_fact_if_model_skips_tool(self):
        fake = FakeClient([{"role": "assistant", "content": "Existem 99 A100 disponíveis."}])
        agent = GpuAgent(client=fake, model_name="teste")
        answer, _ = agent.answer("Quantas A100 estão disponíveis?")
        self.assertNotIn("99", answer)
        self.assertIn("Não consegui confirmar", answer)

    def test_greeting_does_not_call_llm(self):
        fake = FakeClient([])
        answer, _ = GpuAgent(client=fake, model_name="teste").answer("Oi!")
        self.assertIn("Olá", answer)
        self.assertEqual(fake.calls, [])


if __name__ == "__main__":
    unittest.main()


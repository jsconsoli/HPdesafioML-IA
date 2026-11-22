"""Simula respostas de um LLM para validar o ciclo de Tool Calling."""

import json
import tempfile
import unittest
from importlib.util import find_spec
from pathlib import Path
from unittest.mock import Mock, patch

from gpu_assistant.agent import GpuAgent
from gpu_assistant.conversation_log import ConversationLog
from gpu_assistant.database import GpuDatabase
from gpu_assistant.tools import GpuQueryTools


def tool_request(name, arguments):
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"function": {"name": name, "arguments": arguments}}],
    }


def client_reply(message):
    return tool_request("conversar_com_cliente", {"mensagem": message})


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
        self.database = GpuDatabase(db_path)
        self.database.initialize_db()
        self.query_tools = GpuQueryTools(self.database)

    def make_agent(self, **kwargs):
        return GpuAgent(query_tools=self.query_tools, **kwargs)

    def test_tool_call_uses_db_and_variable_model(self):
        fake = FakeClient(
            [
                tool_request("verificar_disponibilidade", {"modelo": "A100", "quantidade": 2}),
                client_reply("Sim, há 4 A100 disponíveis."),
            ]
        )
        agent = self.make_agent(client=fake, model_name="modelo-de-teste:1b")
        answer, history = agent.answer("Existem duas A100 disponíveis?")

        self.assertEqual(answer, "Sim, há 4 A100 disponíveis.")
        self.assertEqual(len(fake.calls), 2)
        self.assertTrue(all(call["model"] == "modelo-de-teste:1b" for call in fake.calls))
        self.assertEqual({tool["function"]["name"] for tool in fake.calls[0]["tools"]}, {
            "consultar_gpus", "verificar_disponibilidade", "consultar_resumo", "conversar_com_cliente",
            "comparar_gpus",
        })
        result = next(message for message in history if message["role"] == "tool")
        self.assertTrue(json.loads(result["content"])["atende_solicitacao"])
        self.assertEqual(history[-1], {"role": "assistant", "content": answer})

    def test_retries_invalid_query_and_invalid_client_message(self):
        fake = FakeClient([
            tool_request("consultar_gpus", {"modelos": '["48GB"]'}),
            {"role": "assistant", "content": "Não existem GPUs com 48 GB."},
            tool_request("conversar_com_cliente", {}),
            tool_request("consultar_gpus", {"memoria_exata_gb": 48}),
            client_reply("NVIDIA A40 e NVIDIA L40S somam 10 unidades."),
        ])

        answer, _ = self.make_agent(client=fake, max_tool_rounds=1).answer("Quais GPUs têm 48 GB?")

        self.assertEqual(len(fake.calls), 5)
        retry_messages = fake.calls[1]["messages"]
        self.assertIn("## Correção de ferramenta", retry_messages[0]["content"])
        self.assertEqual([message["role"] for message in retry_messages], ["system", "user", "assistant", "tool"])
        self.assertEqual(retry_messages[1], fake.calls[0]["messages"][1])
        self.assertEqual(retry_messages[-1]["tool_name"], "consultar_gpus")
        first_error = json.loads(retry_messages[-1]["content"])
        self.assertIn("lista de nomes", first_error["erro"])
        self.assertEqual(first_error["argumentos_recebidos"], {"modelos": '["48GB"]'})
        self.assertIn('{"memoria_exata_gb":48}', first_error["orientacao"])
        self.assertEqual(first_error["novas_tentativas_restantes"], 3)
        self.assertIn("campo mensagem", fake.calls[3]["messages"][-1]["content"])
        self.assertIn("NVIDIA A40", answer)
        self.assertIn("NVIDIA L40S", answer)
        self.assertIn("10 unidades", answer)

    def test_retry_decodes_serialized_model_names(self):
        invalid = tool_request("consultar_gpus", {"modelos": '["A100","H100"]'})
        fake = FakeClient([invalid, invalid, client_reply("NVIDIA A100 e NVIDIA H100 t?m 80 GB.")])

        answer, history = self.make_agent(client=fake).answer("Qual a memória da A100 e da H100?")

        results = [json.loads(message["content"]) for message in history if message["role"] == "tool"]
        self.assertIn("lista de nomes", results[0]["erro"])
        self.assertEqual(results[1]["argumentos_normalizados"], {"modelos": ["A100", "H100"]})
        self.assertIn("NVIDIA A100", answer)
        self.assertIn("NVIDIA H100", answer)

    def test_retries_with_corrected_call_printed_as_text(self):
        client = FakeClient([
            tool_request("verificar_disponibilidade", {"modelo": "A100", "quantidade": "2.0"}),
            {"role": "assistant", "content": 'Vou tentar novamente.\n\n'
             '{"name":"verificar_disponibilidade","parameters":{"modelo":"A100","quantidade":2}}'},
            client_reply("Sim, há quatro A100 disponíveis."),
        ])
        answer, _ = self.make_agent(client=client).answer("Existem duas A100 disponíveis?")
        error = json.loads(client.calls[1]["messages"][-1]["content"])
        self.assertEqual(error["tipos_esperados"]["quantidade"], "integer")
        self.assertEqual(error["argumentos_recebidos"]["quantidade"], "2.0")
        self.assertIn("recebeu '2.0' (tipo texto)", error["erro"])
        self.assertIn("inteiro JSON", error["erro"])
        result = json.loads(client.calls[2]["messages"][-1]["content"])
        self.assertTrue(result["atende_solicitacao"])
        self.assertEqual(answer, "Sim, há quatro A100 disponíveis.")

    def test_normalizes_integer_text_before_executing_queries(self):
        client = FakeClient([
            tool_request("verificar_disponibilidade", {"modelo": "A100", "quantidade": "2"}),
            client_reply("Há quantidade suficiente."),
            tool_request("consultar_gpus", {"memoria_exata_gb": " +48 "}),
            client_reply("A40 e L40S têm 48 GB."),
        ])
        agent = self.make_agent(client=client)
        agent.answer("Existem duas A100 disponíveis?")
        agent.answer("Quais GPUs têm exatamente 48 GB?")

        self.assertEqual(len(client.calls), 4)
        availability = json.loads(client.calls[1]["messages"][-1]["content"])
        self.assertEqual(availability["argumentos_normalizados"], {"quantidade": 2})
        self.assertTrue(availability["atende_solicitacao"])
        memory = json.loads(client.calls[3]["messages"][-1]["content"])
        self.assertEqual(memory["argumentos_normalizados"], {"memoria_exata_gb": 48})
        self.assertEqual([gpu["modelo"] for gpu in memory["gpus"]], ["NVIDIA A40", "NVIDIA L40S"])

    def test_retry_reports_boolean_type_and_uses_corrected_filter(self):
        self.database.update_availability("H100", 0)
        client = FakeClient([
            tool_request("consultar_gpus", {"somente_disponiveis": "not true"}),
            tool_request("consultar_gpus", {"somente_disponiveis": True}),
            client_reply("A100, A40, L40S e RTX 4090 estão disponíveis."),
        ])
        self.make_agent(client=client).answer("Liste todas as GPUs disponíveis.")
        error = json.loads(client.calls[1]["messages"][-1]["content"])
        self.assertEqual(error["argumentos_recebidos"]["somente_disponiveis"], "not true")
        self.assertEqual(error["tipos_esperados"]["somente_disponiveis"], "boolean")
        result = json.loads(client.calls[2]["messages"][-1]["content"])
        self.assertNotIn("NVIDIA H100", [gpu["modelo"] for gpu in result["gpus"]])

    def test_normalizes_boolean_text_before_comparing_available_quantities(self):
        self.database.update_availability("H100", 0)
        for text, boolean, model in (
            (" True ", True, "NVIDIA RTX 4090"),
            (" FALSE ", False, "NVIDIA H100"),
        ):
            with self.subTest(value=text):
                client = FakeClient([
                    tool_request("comparar_gpus", {
                        "criterio": "menos_disponiveis",
                        "somente_disponiveis": text,
                    }),
                    client_reply("Comparação concluída."),
                ])
                self.make_agent(client=client).answer("Qual modelo tem menos unidades disponíveis?")

                self.assertEqual(len(client.calls), 2)
                result = json.loads(client.calls[1]["messages"][-1]["content"])
                self.assertIs(result["argumentos_normalizados"]["somente_disponiveis"], boolean)
                self.assertEqual([gpu["modelo"] for gpu in result["gpus"]], [model])

    def test_asks_for_contextual_reply_after_three_retries(self):
        invalid = tool_request("consultar_gpus", {"memoria_exata_gb": "4.8e1"})
        response = "Consultei o total de GPUs, mas não consegui confirmar quais têm 48 GB."
        fake = FakeClient([
            tool_request("consultar_resumo", {}),
            *([invalid] * 4),
            client_reply(response),
        ])

        answer, history = self.make_agent(client=fake).answer("Quais GPUs têm 48 GB?")

        self.assertEqual(len(fake.calls), 6)
        final_call = fake.calls[-1]
        self.assertEqual([tool["function"]["name"] for tool in final_call["tools"]], ["conversar_com_cliente"])
        self.assertIn("## Entrega da resposta", final_call["messages"][0]["content"])
        self.assertIn(
            "As 3 tentativas de correção deste turno se esgotaram.",
            final_call["messages"][0]["content"],
        )
        final_tool_messages = [message for message in final_call["messages"] if message["role"] == "tool"]
        recorded_tool_messages = [message for message in history if message["role"] == "tool"]
        self.assertEqual(final_tool_messages, recorded_tool_messages)
        self.assertEqual(
            [message["tool_name"] for message in final_tool_messages],
            ["consultar_resumo", *(["consultar_gpus"] * 4)],
        )
        total, *failures = [json.loads(message["content"]) for message in final_tool_messages]
        self.assertIn("total_disponivel", total)
        self.assertEqual([failure["novas_tentativas_restantes"] for failure in failures], [3, 2, 1, 0])
        for failure in failures:
            self.assertEqual(failure["erro"], "O parâmetro memoria_exata_gb deve ser um inteiro positivo em GB.")
            self.assertEqual(failure["argumentos_recebidos"], {"memoria_exata_gb": "4.8e1"})
        self.assertEqual(answer, response)
        self.assertEqual(history[-1], {"role": "assistant", "content": response})

    def test_accepts_plain_text_at_tool_round_limit(self):
        response = "Há 19 GPUs disponíveis agora."
        fake = FakeClient([
            tool_request("consultar_resumo", {}),
            {"role": "assistant", "content": response},
        ])

        answer, _ = self.make_agent(client=fake, max_tool_rounds=1).answer("Quantas GPUs estão disponíveis?")

        self.assertEqual(len(fake.calls), 2)
        self.assertEqual([tool["function"]["name"] for tool in fake.calls[-1]["tools"]], ["conversar_com_cliente"])
        self.assertEqual(answer, response)

    def test_blocks_database_calls_at_tool_round_limit(self):
        fake = FakeClient([
            tool_request("consultar_resumo", {}),
            tool_request("consultar_gpus", {"modelo": "A100"}),
        ])
        lookup = Mock()

        with patch.object(self.query_tools, "consultar_gpus", lookup):
            answer, history = self.make_agent(client=fake, max_tool_rounds=1).answer("Quantas A100 estão disponíveis?")

        lookup.assert_not_called()
        self.assertEqual(len(fake.calls), 2)
        self.assertEqual([tool["function"]["name"] for tool in fake.calls[-1]["tools"]], ["conversar_com_cliente"])
        self.assertTrue(answer.strip())
        self.assertEqual(history[-1], {"role": "assistant", "content": answer})

    def test_identity_reply_uses_history_and_skips_database(self):
        client = FakeClient([
            client_reply("Olá! Tudo bem."),
            client_reply("Sou o assistente de consultas à disponibilidade de GPUs."),
            client_reply("Sou o assistente de consultas à disponibilidade de GPUs."),
        ])
        agent = self.make_agent(client=client)
        _, first_history = agent.answer("Oi, tudo bem?")
        answer, history = agent.answer("Com quem estou falando?", first_history)
        self.assertEqual(answer, "Sou o assistente de consultas à disponibilidade de GPUs.")
        self.assertEqual(
            first_history,
            [
                {"role": "user", "content": "Oi, tudo bem?"},
                {"role": "assistant", "content": "Olá! Tudo bem."},
            ],
        )
        self.assertEqual(history[-2], {"role": "user", "content": "Com quem estou falando?"})
        repeated, _ = agent.answer("É o que eu perguntei", history)
        self.assertEqual(repeated, answer)
        turns = (
            ([], "Oi, tudo bem?"),
            (first_history, "Com quem estou falando?"),
            (history, "É o que eu perguntei"),
        )
        for call, (previous_messages, question) in zip(client.calls, turns, strict=True):
            self.assertEqual(call["tools"], client.calls[0]["tools"])
            self.assertEqual(call["messages"][0], {"role": "system", "content": GpuAgent.SYSTEM_PROMPT})
            self.assertEqual(call["messages"][1:-1], previous_messages)
            self.assertEqual(call["messages"][-1], {
                "role": "user",
                "content": f"## Última mensagem do cliente\n\n{question}",
            })

    def test_accepts_text_serialization_of_delivery_call(self):
        client = FakeClient([{
            "role": "assistant",
            "content": '{"mensagem":"Sou o assistente de GPUs."}',
        }])
        answer, _ = self.make_agent(client=client).answer("Quem é você?")
        self.assertEqual(answer, "Sou o assistente de GPUs.")

    def test_listing_and_comparison_results_reach_model(self):
        self.database.update_availability("H100", 0)
        client = FakeClient([
            tool_request("consultar_gpus", {"somente_disponiveis": True}),
            client_reply("Agora est?o dispon?veis A100, A40, L40S e RTX 4090."),
            tool_request("comparar_gpus", {"criterio": "maior_memoria"}),
            client_reply("A100 e H100 empatam com 80 GB."),
        ])
        agent = self.make_agent(client=client)
        _, history = agent.answer("Me liste todos os modelos disponiveis")
        agent.answer("Qual tem a maior mem?ria?", history)
        for index, models in (
            (1, ["NVIDIA A100", "NVIDIA A40", "NVIDIA L40S", "NVIDIA RTX 4090"]),
            (3, ["NVIDIA A100", "NVIDIA H100"]),
        ):
            payload = json.loads(client.calls[index]["messages"][-1]["content"])
            self.assertEqual([gpu["modelo"] for gpu in payload["gpus"]], models)

    def test_can_explain_a_tool_error_through_client_reply(self):
        client = FakeClient([
            tool_request("consultar_gpus", {"memoria_exata_gb": "48 GB"}),
            client_reply("Pode confirmar o valor de mem?ria que voc? quer consultar?"),
        ])
        reply, _ = self.make_agent(client=client).answer("Filtre por mem?ria.")
        self.assertEqual(reply, "Pode confirmar o valor de mem?ria que voc? quer consultar?")
        result = json.loads(client.calls[1]["messages"][-1]["content"])
        self.assertIn("erro", result)

    def test_serialized_empty_list_preserves_listing(self):
        client = FakeClient([
            tool_request("consultar_gpus", {"modelos": "[]"}),
            client_reply("O cadastro tem 19 unidades dispon?veis."),
        ])
        answer, history = self.make_agent(client=client).answer("O que o cadastro tem?")
        self.assertEqual(len(json.loads(history[-2]["content"])["gpus"]), 5)
        self.assertIn("19 unidades", answer)
        self.assertEqual(len(client.calls), 2)

    def test_client_message_ends_turn_before_other_calls(self):
        query = tool_request("consultar_resumo", {})
        clarification = client_reply("Qual modelo você quer consultar?")
        query["tool_calls"].extend(clarification["tool_calls"])
        lookup = Mock()
        with patch.object(self.query_tools, "consultar_resumo", lookup):
            answer, history = self.make_agent(client=FakeClient([query])).answer("aquilo")
        lookup.assert_not_called()
        self.assertEqual(answer, "Qual modelo você quer consultar?")
        self.assertEqual(history[-1], {"role": "assistant", "content": answer})

    def test_model_interprets_quantity_in_a_followup(self):
        fake = FakeClient([
            client_reply("De quantas unidades você precisa?"),
            tool_request("verificar_disponibilidade", {"modelo": "L40S", "quantidade": 12}),
            client_reply("Há 6 unidades disponíveis para uma solicitação de 12."),
        ])
        agent = self.make_agent(client=fake)
        history = [
            {"role": "user", "content": "Quantas L40S estão disponíveis?"},
            {"role": "assistant", "content": "Há 6 unidades disponíveis."},
        ]
        lookup = Mock()
        with patch.object(self.query_tools, "verificar_disponibilidade", lookup):
            answer, history = agent.answer("E para mim?", history)
        lookup.assert_not_called()
        self.assertEqual(answer, "De quantas unidades você precisa?")
        answer, _ = agent.answer("uma dúzia", history)
        self.assertIn("solicitação de 12", answer)
        result = json.loads(fake.calls[2]["messages"][-1]["content"])
        self.assertEqual(result["quantidade_solicitada"], 12)

    @unittest.skipUnless(find_spec("ollama"), "Requer o cliente Python do Ollama.")
    def test_logs_each_ollama_request_and_response(self):
        from httpx import MockTransport, Response
        from ollama import Client

        requests = []
        responses = [
            {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"function": {
                            "name": "consultar_gpus",
                            "arguments": {"modelo": "L40S"},
                        }},
                    ],
                },
                "done": True,
            },
            {
                "message": {"role": "assistant", "content": " Cada L40S tem 48 GB. "},
                "done": True,
            },
        ]

        def respond(request):
            response = responses[len(requests)]
            requests.append(json.loads(request.content))
            return Response(200, json=response)

        def create_client(**kwargs):
            return Client(transport=MockTransport(respond), **kwargs)

        log_path = Path(self.temp.name) / "conversas.log"
        with patch("ollama.Client", side_effect=create_client):
            agent = self.make_agent(
                model_name="modelo-de-teste:1b", conversation_log=ConversationLog(log_path)
            )
            answer, _ = agent.answer("Qual é a memória da L40S?")

        events = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
        self.assertEqual([event["event"] for event in events], ["request", "response"] * 2)
        self.assertEqual([event["payload"] for event in events[::2]], requests)
        self.assertEqual([event["payload"] for event in events[1::2]], responses)
        self.assertEqual(events[0]["request_id"], events[1]["request_id"])
        self.assertEqual(events[2]["request_id"], events[3]["request_id"])
        self.assertNotEqual(events[0]["request_id"], events[2]["request_id"])
        self.assertEqual(requests[0]["messages"][1], {
            "role": "user",
            "content": "## Última mensagem do cliente\n\nQual é a memória da L40S?",
        })
        self.assertEqual(requests[1]["messages"][1], requests[0]["messages"][1])
        self.assertEqual(
            [message["role"] for message in requests[1]["messages"]],
            ["system", "user", "assistant", "tool"],
        )
        tool_result = json.loads(events[2]["payload"]["messages"][-1]["content"])
        self.assertEqual(tool_result["gpus"][0]["memoria_gb"], 48)
        schemas = {tool["function"]["name"]: tool["function"] for tool in requests[0]["tools"]}
        for name in ("consultar_gpus", "comparar_gpus", "verificar_disponibilidade", "consultar_resumo"):
            self.assertTrue(schemas[name]["description"].startswith(self.query_tools.description))
        self.assertIn("Ferramenta geral de conversa", schemas["conversar_com_cliente"]["description"])
        self.assertEqual(requests[0]["messages"][0]["content"], GpuAgent.SYSTEM_PROMPT)
        for heading in ("## Papel da LLM", "## Histórico da conversa", "## Turno atual"):
            self.assertIn(heading, requests[0]["messages"][0]["content"])
        properties = schemas["consultar_gpus"]["parameters"]["properties"]
        self.assertEqual(properties["modelos"]["items"], {"type": "string"})
        self.assertEqual(properties["memoria_exata_gb"]["type"], "integer")
        self.assertEqual(answer, "Cada L40S tem 48 GB.")


if __name__ == "__main__":
    unittest.main()


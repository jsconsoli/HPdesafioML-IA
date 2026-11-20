"""Orquestra a conversa e as chamadas de ferramentas de consulta via Ollama."""

import json
import re
from typing import Any

from .config import get_model_name, get_ollama_host
from .tools import READ_ONLY_TOOLS, TOOLS_BY_NAME


SYSTEM_PROMPT = """Você é um assistente de consultas à disponibilidade de GPUs.
Responda em português de forma objetiva e natural, salvo se o usuário pedir outro idioma.

REGRAS IMPORTANTES:
- Para responder qualquer pergunta factual sobre GPUs, quantidades, memória ou
  disponibilidade, use uma ou mais das ferramentas fornecidas nesta conversa.
- Não invente dados. As informações do banco são a única fonte de verdade.
- 'quantidade_disponivel' é a quantidade LIVRE agora, não o estoque total.
  Não há dados sobre unidades em uso, estoque físico total ou reservas futuras.
- 'memoria_gb' é a memória POR GPU, não a memória agregada da infraestrutura.
- Se um modelo não for encontrado, informe isso claramente.
- Quando uma pergunta não puder ser respondida com os dados disponíveis, diga
  que a informação não consta no sistema.
- Você não pode alterar, adicionar, reservar nem remover GPUs.
- Para perguntas sobre dois ou mais modelos, consulte todos os modelos citados.
- Nunca atribua ao banco informações que a ferramenta não tenha retornado.
"""


def _obj_to_dict(value: Any) -> dict:
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump(exclude_none=True)
    raise TypeError("A resposta do modelo não possui formato reconhecido.")


def _simple_greeting(question: str) -> bool:
    return bool(
        re.fullmatch(
            r"\s*(oi+|ol[aá]|opa|bom dia|boa tarde|boa noite|obrigad[oa]|valeu)[!.?\s]*",
            question.casefold(),
        )
    )


class GpuAgent:
    """Agente sem ferramentas de escrita; ``client`` é injetável para testes."""

    def __init__(
        self,
        *,
        model_name: str | None = None,
        host: str | None = None,
        client: Any = None,
        max_tool_rounds: int = 4,
    ) -> None:
        self.model_name = model_name or get_model_name()
        self.host = host or get_ollama_host()
        self.max_tool_rounds = max_tool_rounds
        if client is None:
            # Importação tardia: os testes da lógica não exigem Ollama instalado.
            from ollama import Client

            client = Client(host=self.host)
        self.client = client

    def answer(self, question: str, history: list[dict] | None = None) -> tuple[str, list[dict]]:
        """Retorna (resposta, histórico) para manter contexto entre perguntas."""
        if not question.strip():
            return "Digite uma pergunta sobre as GPUs.", list(history or [])

        conversation = list(history or [])
        conversation.append({"role": "user", "content": question})

        # Saudações simples dispensam chamada ao modelo ou ao banco.
        if _simple_greeting(question):
            response = "Olá! Posso consultar os modelos, memórias e quantidades disponíveis de GPUs."
            conversation.append({"role": "assistant", "content": response})
            return response, conversation

        used_tool = False
        for _ in range(self.max_tool_rounds + 1):
            result = self.client.chat(
                model=self.model_name,
                messages=[{"role": "system", "content": SYSTEM_PROMPT}, *conversation],
                tools=list(READ_ONLY_TOOLS),
                options={"temperature": 0},
                stream=False,
            )
            raw_message = result["message"] if isinstance(result, dict) else result.message
            message = _obj_to_dict(raw_message)
            tool_calls = message.get("tool_calls") or []

            if not tool_calls:
                text = (message.get("content") or "").strip()
                if not used_tool:
                    # Uma resposta factual sem consulta não deve chegar ao usuário.
                    text = (
                        "Não consegui confirmar essa informação na base de GPUs. "
                        "Tente reformular sua consulta."
                    )
                elif not text:
                    text = "Não consegui formular uma resposta a partir da consulta realizada."
                conversation.append({"role": "assistant", "content": text})
                return text, conversation

            if _ >= self.max_tool_rounds:
                break

            conversation.append(message)
            for tool_call in tool_calls:
                call = _obj_to_dict(tool_call)
                function = _obj_to_dict(call.get("function", {}))
                name = function.get("name", "")
                arguments = function.get("arguments", {})

                try:
                    if isinstance(arguments, str):
                        arguments = json.loads(arguments)
                    if not isinstance(arguments, dict):
                        raise ValueError("Argumentos inválidos.")
                    if name not in TOOLS_BY_NAME:
                        raise ValueError("Ferramenta não permitida.")
                    tool_result = TOOLS_BY_NAME[name](**arguments)
                    used_tool = True
                except (ValueError, TypeError, KeyError, FileNotFoundError) as exc:
                    tool_result = {"erro": str(exc)}
                except Exception:
                    # Não expor detalhes internos em mensagens do agente.
                    tool_result = {"erro": "Falha ao consultar a base de GPUs."}

                conversation.append(
                    {
                        "role": "tool",
                        "tool_name": name,
                        "content": json.dumps(tool_result, ensure_ascii=False),
                    }
                )

        response = "Não consegui concluir a consulta. Tente fazer uma pergunta mais específica."
        conversation.append({"role": "assistant", "content": response})
        return response, conversation


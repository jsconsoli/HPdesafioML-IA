"""Orquestra a conversa e as chamadas de ferramentas de consulta via Ollama."""

import json
import re
from typing import Any

from .config import get_model_name, get_ollama_host
from .conversation_log import ConversationLog
from .tools import ClientReplyTool, GpuQueryTools


def _obj_to_dict(value: Any) -> dict:
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump(exclude_none=True)
    raise TypeError("A resposta do modelo não possui formato reconhecido.")


def _retry_call_from_text(content: str, last_message: dict) -> list[dict]:
    """Recupera a chamada quando o Ollama devolve o JSON do retry como texto."""
    if last_message["role"] != "tool" or "erro" not in json.loads(last_message["content"]):
        return []
    match = re.search(r"(?m)^[ \t]*(\{[^\r\n]*\})\s*\Z", content)
    if not match:
        return []
    try:
        call = json.loads(match[1])
    except json.JSONDecodeError:
        return []
    if not isinstance(call, dict) or call.get("name") != last_message.get("tool_name"):
        return []
    arguments = call.get("parameters", call.get("arguments"))
    if not isinstance(arguments, dict) or set(call) not in ({"name", "parameters"}, {"name", "arguments"}):
        return []
    return [{"function": {"name": call["name"], "arguments": arguments}}]


class GpuAgent:
    """Agente sem ferramentas de escrita; ``client`` é injetável para testes."""

    SYSTEM_PROMPT = (
        "## Papel da LLM\n"
        "Você é um assistente de consultas à disponibilidade de GPUs. "
        "Seu acesso ao estoque é somente leitura. Você pode verificar disponibilidade, "
        "mas não pode fazer, confirmar ou prometer reservas. "
        "Se perguntarem sobre reservas, explique esse limite sem indicar canais ou procedimentos de reserva.\n\n"
        "## Histórico da conversa\n"
        "As mensagens anteriores ao bloco «Última mensagem do cliente» formam o histórico.\n"
        "- `user` identifica mensagens do cliente.\n"
        "- `assistant` identifica mensagens e chamadas de ferramentas da LLM.\n"
        "- `tool` identifica resultados das ferramentas.\n"
        "Use o histórico para interpretar referências no pedido atual.\n\n"
        "## Turno atual\n"
        "Responda ao bloco «Última mensagem do cliente». As chamadas e resultados que vêm depois "
        "desse bloco pertencem ao turno atual. Consulte dados de GPUs apenas quando forem "
        "necessários para responder ao pedido atual."
    )

    CURRENT_USER_HEADER = "## Última mensagem do cliente\n\n"
    MAX_TOOL_RETRIES = 3

    RETRY_PROMPT = (
        "\n\n## Correção de ferramenta\n"
        "Leia o erro e os argumentos recebidos na última resposta da ferramenta. "
        "Se o erro for de tipo, chame a mesma ferramenta e ajuste o tipo do argumento. "
        "Emita uma chamada estruturada em vez de escrever o JSON como texto. "
        "Preserve o escopo, os nomes e os critérios pedidos pelo usuário."
    )

    DELIVERY_PROMPT = (
        "\n\n## Entrega da resposta\n"
        "Entregue a resposta ao usuário com o contexto disponível. "
        "Use os diagnósticos das ferramentas para entender quais consultas falharam. "
        "Use os resultados bem-sucedidos deste turno para os dados de GPUs. "
        "Se a consulta ficou incompleta, explique em linguagem natural o que ficou sem confirmação. "
        "Erros de argumentos se referem aos valores que o assistente enviou à ferramenta. "
        "Peça esclarecimento apenas se faltar informação na mensagem do usuário ou no histórico."
    )

    def __init__(
        self,
        *,
        model_name: str | None = None,
        host: str | None = None,
        client: Any = None,
        max_tool_rounds: int = 4,
        query_tools: GpuQueryTools | None = None,
        conversation_log: ConversationLog | None = None,
    ) -> None:
        self.model_name = model_name or get_model_name()
        self.host = host or get_ollama_host()
        self.max_tool_rounds = max_tool_rounds
        self.query_tools = query_tools if query_tools is not None else GpuQueryTools()
        self._client_reply_tool = ClientReplyTool()
        self._tool_schemas = {
            **self.query_tools.schemas,
            self._client_reply_tool.name: self._client_reply_tool.schema(),
        }
        self._ollama_tools = [
            {"type": "function", "function": {"name": name, **schema}}
            for name, schema in self._tool_schemas.items()
        ]
        if client is None:
            # Importação tardia: os testes da lógica não exigem Ollama instalado.
            from ollama import Client

            log = conversation_log if conversation_log is not None else ConversationLog()
            client = Client(
                host=self.host,
                event_hooks={"request": [log.log_request], "response": [log.log_response]},
            )
        self.client = client

    def _execute_tool(self, name: str, arguments: Any, *, retrying: bool) -> dict:
        try:
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            if not isinstance(arguments, dict):
                raise ValueError("Passe os argumentos como um objeto JSON.")
            if name == self._client_reply_tool.name:
                return self._client_reply_tool.execute(arguments)
            schema = self.query_tools.schemas.get(name)
            if schema is None:
                raise ValueError("Ferramenta não permitida.")

            normalized = {}
            for argument, field in schema["parameters"]["properties"].items():
                value = arguments.get(argument)
                if not isinstance(value, str):
                    continue
                literal = value.strip()
                if field["type"] == "boolean" and re.fullmatch(r"true|false", literal, flags=re.IGNORECASE | re.ASCII):
                    normalized[argument] = literal.lower() == "true"
                elif field["type"] == "integer" and re.fullmatch(r"[+-]?[0-9]+", literal):
                    normalized[argument] = int(literal)

            if name == "consultar_gpus" and isinstance(arguments.get("modelos"), str):
                # Uma lista vazia mantém o significado de listagem completa.
                # Listas com nomes passam pela orientação de correção primeiro.
                try:
                    names = json.loads(arguments["modelos"])
                except json.JSONDecodeError:
                    names = None
                if isinstance(names, list) and (not names or retrying):
                    normalized["modelos"] = names

            result = getattr(self.query_tools, name)(**{**arguments, **normalized})
            if normalized and "erro" not in result:
                result["argumentos_normalizados"] = normalized
            return result
        except json.JSONDecodeError:
            return {"erro": "Os argumentos precisam ser um objeto JSON válido."}
        except (ValueError, TypeError, KeyError, FileNotFoundError) as exc:
            return {"erro": str(exc)}
        except Exception:
            return {"erro": "Falha ao consultar a base de GPUs."}

    def answer(self, question: str, history: list[dict] | None = None) -> tuple[str, list[dict]]:
        """Retorna (resposta, histórico) para manter contexto entre perguntas."""
        if not question.strip():
            return "Digite uma pergunta sobre as GPUs.", list(history or [])

        conversation = list(history or [])
        current_user_index = len(conversation)
        conversation.append({"role": "user", "content": question})
        current_user_message = {"role": "user", "content": self.CURRENT_USER_HEADER + question}

        retries_left = self.MAX_TOOL_RETRIES
        query_rounds_left = self.max_tool_rounds
        needs_retry = False
        delivery_only = False
        response = ""

        while True:
            delivery_only = delivery_only or (
                retries_left == 0 if needs_retry else query_rounds_left <= 0
            )
            prompt = self.SYSTEM_PROMPT
            tools = self._ollama_tools
            if delivery_only:
                prompt += self.DELIVERY_PROMPT
                if needs_retry and retries_left == 0:
                    prompt += f"\nAs {self.MAX_TOOL_RETRIES} tentativas de correção deste turno se esgotaram."
                tools = [tool for tool in self._ollama_tools if tool["function"]["name"] == self._client_reply_tool.name]
            elif needs_retry:
                retries_left -= 1
                prompt += f"{self.RETRY_PROMPT}\nTentativa {self.MAX_TOOL_RETRIES - retries_left} de {self.MAX_TOOL_RETRIES}."

            result = self.client.chat(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": prompt},
                    *conversation[:current_user_index],
                    current_user_message,
                    *conversation[current_user_index + 1:],
                ],
                tools=tools,
                options={"temperature": 0},
                stream=False,
            )
            raw_message = result["message"] if isinstance(result, dict) else result.message
            message = _obj_to_dict(raw_message)
            tool_calls = message.get("tool_calls") or []
            if not tool_calls:
                # Alguns modelos imprimem a chamada de entrega como texto.
                delivery_text = re.fullmatch(
                    rf'\s*(?:{self._client_reply_tool.name}\s*,\s*)?(\{{\s*"mensagem"\s*:.*\}})\s*',
                    message.get("content") or "",
                    flags=re.IGNORECASE | re.DOTALL,
                )
                if delivery_text:
                    try:
                        arguments = json.loads(delivery_text[1])
                    except json.JSONDecodeError:
                        arguments = {}
                    tool_calls = [{"function": {"name": self._client_reply_tool.name, "arguments": arguments}}]
                    message = {"role": "assistant", "tool_calls": tool_calls}

            if not tool_calls and needs_retry and not delivery_only:
                tool_calls = _retry_call_from_text(message.get("content") or "", conversation[-1])
                if tool_calls:
                    message = {"role": "assistant", "tool_calls": tool_calls}

            if not tool_calls:
                if needs_retry and not delivery_only:
                    continue
                response = (message.get("content") or "").strip()
                if response or delivery_only:
                    break
                delivery_only = True
                continue

            functions = [
                _obj_to_dict(_obj_to_dict(call).get("function", {})) for call in tool_calls
            ]
            delivery = next(
                (i for i, function in enumerate(functions) if function.get("name") == self._client_reply_tool.name), None
            )
            if delivery is not None:
                functions = [functions[delivery]]
                message = {**message, "tool_calls": [tool_calls[delivery]]}
            elif delivery_only:
                break
            elif not needs_retry:
                query_rounds_left -= 1
            retrying = needs_retry
            needs_retry = False
            conversation.append(message)
            for function in functions:
                name = function.get("name", "")
                if not isinstance(name, str):
                    name = ""
                tool_result = self._execute_tool(name, function.get("arguments", {}), retrying=retrying)
                if "erro" in tool_result:
                    needs_retry = True
                    schema = self._tool_schemas.get(name)
                    usage = schema["description"] if schema else (
                        f"Use uma das ferramentas disponíveis ({', '.join(self._tool_schemas)})."
                    )
                    tool_result = {
                        **tool_result,
                        "argumentos_recebidos": function.get("arguments", {}),
                        "tipos_esperados": {
                            argument: field["type"]
                            for argument, field in (schema or {}).get("parameters", {}).get("properties", {}).items()
                        },
                        "orientacao": usage,
                        "novas_tentativas_restantes": retries_left,
                    }
                elif name == self._client_reply_tool.name:
                    response = tool_result["mensagem"]
                    conversation[-1] = {"role": "assistant", "content": response}
                    return response, conversation

                conversation.append(
                    {
                        "role": "tool",
                        "tool_name": name,
                        "content": json.dumps(tool_result, ensure_ascii=False),
                    }
                )

            if delivery_only:
                break

        response = response or "Não consegui gerar uma resposta agora."
        conversation.append({"role": "assistant", "content": response})
        return response, conversation


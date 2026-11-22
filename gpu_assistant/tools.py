"""Consultas de GPUs e contratos das ferramentas expostas ao Ollama."""

from .database import GpuDatabase


def _tool_schema(description: str, properties: dict, required: tuple[str, ...] = ()) -> dict:
    return {
        "description": description,
        "parameters": {"type": "object", "properties": properties, "required": list(required)},
    }


class GpuQueryTools:
    """Consultas de GPUs no SQLite em modo somente leitura.

    Cada registro contém um modelo, a memória por unidade e as unidades livres.
    O total disponível é a soma dessas unidades.
    """

    description = (
        "Ferramentas de consulta de GPUs. Use para consultar informações sobre modelos, memória e disponibilidade."
    )

    def __init__(self, database: GpuDatabase | None = None) -> None:
        self.database = database if database is not None else GpuDatabase()
        self._comparisons = {
            "maior_memoria": ("memoria_gb", max),
            "menor_memoria": ("memoria_gb", min),
            "mais_disponiveis": ("quantidade_disponivel", max),
            "menos_disponiveis": ("quantidade_disponivel", min),
        }
        self.schemas = self._build_schemas()

    def consultar_gpus(
        self,
        modelos: list[str] | None = None,
        *,
        modelo: str | None = None,
        memoria_exata_gb: int | None = None,
        memoria_minima_gb: int | None = None,
        somente_disponiveis: bool = False,
    ) -> dict:
        """Consulta modelos, filtrando por nome ou memória por unidade."""
        if type(somente_disponiveis) is not bool:
            return {"erro": "Use um booleano em somente_disponiveis."}
        if modelo is not None:
            if not isinstance(modelo, str) or not modelo.strip():
                return {"erro": "O parâmetro modelo deve ser um nome de GPU."}
            if modelos is not None:
                return {"erro": "Use modelo para um único nome ou modelos para uma lista."}
            modelos = [modelo]
        if modelos is not None and (
            not isinstance(modelos, list)
            or not all(isinstance(nome, str) and nome.strip() for nome in modelos)
        ):
            return {
                "erro": "O parâmetro modelos deve ser uma lista de nomes. "
                "Para um único nome, passe esse nome como texto em modelo."
            }
        for filtro, valor in (
            ("memoria_exata_gb", memoria_exata_gb),
            ("memoria_minima_gb", memoria_minima_gb),
        ):
            if valor is not None and (type(valor) is not int or valor < 1):
                return {"erro": f"O parâmetro {filtro} deve ser um inteiro positivo em GB."}
        if memoria_exata_gb is not None and memoria_minima_gb is not None:
            return {"erro": "Use apenas um filtro de memória na mesma consulta."}

        found = []
        unknown = []
        seen = set()
        for name in modelos or []:
            gpu = self.database.get_gpu(name)
            if gpu is None:
                unknown.append(name)
            elif gpu["modelo"] not in seen:
                found.append(gpu)
                seen.add(gpu["modelo"])

        gpus = [
            gpu for gpu in (found if modelos else self.database.list_gpus())
            if (memoria_exata_gb is None or gpu["memoria_gb"] == memoria_exata_gb)
            and (memoria_minima_gb is None or gpu["memoria_gb"] >= memoria_minima_gb)
            and (not somente_disponiveis or gpu["quantidade_disponivel"] > 0)
        ]
        result = {"gpus": gpus}
        if modelos:
            result["modelos_nao_encontrados"] = unknown
        return result

    def verificar_disponibilidade(self, modelo: str, quantidade: int) -> dict:
        """Verifica se o estoque disponível de um modelo atende a uma solicitação."""
        if not isinstance(modelo, str) or not modelo.strip():
            return {"erro": "O parâmetro modelo deve ser um nome de GPU."}
        if type(quantidade) is not int:
            tipo = "texto" if isinstance(quantidade, str) else type(quantidade).__name__
            return {
                "erro": f"O argumento quantidade enviado à ferramenta recebeu {quantidade!r} (tipo {tipo}). "
                "Envie um inteiro JSON, como 2, sem aspas."
            }
        if quantidade < 1:
            return {"erro": "O argumento quantidade deve ser maior ou igual a 1."}
        gpu = self.database.get_gpu(modelo)
        if gpu is None:
            return {"modelo_nao_encontrado": modelo}
        return {
            "modelo": gpu["modelo"],
            "quantidade_solicitada": quantidade,
            "quantidade_disponivel": gpu["quantidade_disponivel"],
            "atende_solicitacao": gpu["quantidade_disponivel"] >= quantidade,
        }

    def consultar_resumo(self) -> dict:
        """Retorna o número de modelos cadastrados e a soma das unidades livres agora."""
        return self.database.get_summary()

    def comparar_gpus(self, criterio: str, somente_disponiveis: bool = False) -> dict:
        """Retorna todos os modelos empatados no extremo solicitado."""
        comparison = self._comparisons.get(criterio) if isinstance(criterio, str) else None
        if comparison is None:
            return {"erro": f"Use um dos critérios disponíveis ({', '.join(self._comparisons)})."}
        result = self.consultar_gpus(somente_disponiveis=somente_disponiveis)
        if "erro" in result or not result["gpus"]:
            return result
        field, aggregate = comparison
        best = aggregate(gpu[field] for gpu in result["gpus"])
        return {"criterio": criterio, "gpus": [gpu for gpu in result["gpus"] if gpu[field] == best]}

    def _build_schemas(self) -> dict:
        schemas = {
            "consultar_gpus": _tool_schema(
                'Use para listar modelos ou obter a memória por unidade e a quantidade livre de cada modelo. '
                'Use {} para listar o cadastro completo. Para listar todos os modelos com unidades livres, '
                'use {"somente_disponiveis":true}. Omita os filtros por nome e memória nessas listagens. '
                'Preencha modelo ou modelos apenas com os nomes pedidos pelo usuário. '
                'Ao buscar GPUs por memória em GB, use um dos filtros de memória. '
                'Para exatamente 48 GB, use {"memoria_exata_gb":48}. '
                'Para pelo menos 48 GB, use {"memoria_minima_gb":48}. '
                'Use um único filtro em GB e passe o valor como número inteiro.',
                {
                    "modelo": {
                        "type": "string",
                        "description": "Nome de um único modelo pedido pelo usuário. Para memória em GB, use o filtro de memória.",
                    },
                    "modelos": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Lista JSON de nomes pedidos pelo usuário. Passe um array de strings.",
                    },
                    "memoria_exata_gb": {
                        "type": "integer",
                        "description": "Memória exata por GPU em GB, como 48. Omita quando o filtro for mínimo.",
                    },
                    "memoria_minima_gb": {
                        "type": "integer",
                        "description": "Memória mínima por GPU em GB, como 48. Omita quando o filtro for exato.",
                    },
                    "somente_disponiveis": {
                        "type": "boolean",
                        "description": "Envie o booleano JSON true, sem aspas, para excluir modelos com zero unidades livres.",
                    },
                },
            ),
            "comparar_gpus": _tool_schema(
                'Para saber quais modelos têm menos unidades disponíveis, use {"criterio":"menos_disponiveis"}. '
                'Para a maior quantidade disponível, use {"criterio":"mais_disponiveis"}. '
                "Para comparar memória por unidade, use maior_memoria ou menor_memoria. "
                "Retorna todos os modelos empatados.",
                {
                    "criterio": {"type": "string", "enum": list(self._comparisons)},
                    "somente_disponiveis": {
                        "type": "boolean",
                        "description": "Opcional. Use o booleano JSON true, sem aspas, se o usuário limitar a comparação "
                        "a modelos com unidades livres. Omita para comparar todo o cadastro.",
                    },
                },
                ("criterio",),
            ),
            "verificar_disponibilidade": _tool_schema(
                'Use somente para verificar se há unidades suficientes quando o usuário informa '
                'um modelo específico e quantas unidades precisa. Para consultar quantas estão livres '
                'ou buscar GPUs com determinada memória em GB, use consultar_gpus. '
                'Para pedir duas A100, use {"modelo":"A100","quantidade":2}.',
                {
                    "modelo": {"type": "string", "description": 'Nome do modelo solicitado, como "A100".'},
                    "quantidade": {
                        "type": "integer",
                        "description": "Número de unidades pedido pelo usuário. Envie um inteiro positivo em JSON, sem aspas.",
                    },
                },
                ("modelo", "quantidade"),
            ),
            "consultar_resumo": _tool_schema(
                "Retorna dois totais do cadastro de GPUs: o número de modelos cadastrados "
                "e a soma das unidades livres agora. Use {}.",
                {},
            ),
        }
        return {
            name: {**schema, "description": f"{self.description} {schema['description']}"}
            for name, schema in schemas.items()
        }


class ClientReplyTool:
    """Valida a mensagem escrita pelo modelo para entrega ao cliente."""

    name = "conversar_com_cliente"

    def schema(self) -> dict:
        return _tool_schema(
            "Ferramenta geral de conversa. Encerra o uso de ferramentas e entrega a mensagem ao cliente. "
            "Use também para conversar ou pedir a informação que falta. "
            "Ao responder sobre disponibilidade atual, use os resultados bem-sucedidos do turno.",
            {"mensagem": {"type": "string", "description": "Mensagem completa a enviar ao cliente."}},
            ("mensagem",),
        )

    def execute(self, arguments: dict) -> dict:
        message = arguments.get("mensagem")
        if arguments.keys() != {"mensagem"} or not isinstance(message, str) or not message.strip():
            raise ValueError("Passe a mensagem completa no campo mensagem.")
        return {"mensagem": message}


# Mantém a API de consulta do módulo.
def consultar_gpus(
    modelos: list[str] | None = None,
    *,
    modelo: str | None = None,
    memoria_exata_gb: int | None = None,
    memoria_minima_gb: int | None = None,
    somente_disponiveis: bool = False,
) -> dict:
    return GpuQueryTools().consultar_gpus(
        modelos,
        modelo=modelo,
        memoria_exata_gb=memoria_exata_gb,
        memoria_minima_gb=memoria_minima_gb,
        somente_disponiveis=somente_disponiveis,
    )


def comparar_gpus(criterio: str, somente_disponiveis: bool = False) -> dict:
    return GpuQueryTools().comparar_gpus(criterio, somente_disponiveis)


def verificar_disponibilidade(modelo: str, quantidade: int) -> dict:
    return GpuQueryTools().verificar_disponibilidade(modelo, quantidade)


def consultar_resumo() -> dict:
    return GpuQueryTools().consultar_resumo()

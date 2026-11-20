"""As únicas ferramentas expostas ao modelo: todas são consultas."""

from .database import get_gpu, get_summary, list_gpus


def consultar_gpus(modelos: list[str] | None = None) -> dict:
    """Consulta modelos, memória (GB por GPU) e quantidades atualmente disponíveis.

    Args:
        modelos: Nomes das GPUs desejadas. Se omitido, retorna todos os modelos.
    """
    if modelos is None:
        return {"gpus": list_gpus()}
    if not isinstance(modelos, list) or not all(isinstance(m, str) for m in modelos):
        return {"erro": "O parâmetro modelos deve ser uma lista de nomes."}
    if not modelos:
        return {"gpus": list_gpus()}

    found = []
    unknown = []
    seen = set()
    for name in modelos:
        gpu = get_gpu(name)
        if gpu is None:
            unknown.append(name)
        elif gpu["modelo"] not in seen:
            found.append(gpu)
            seen.add(gpu["modelo"])
    return {"gpus": found, "modelos_nao_encontrados": unknown}


def verificar_disponibilidade(modelo: str, quantidade: int) -> dict:
    """Verifica se há pelo menos a quantidade pedida de um modelo específico.

    Args:
        modelo: Nome da GPU, por exemplo A100 ou NVIDIA H100.
        quantidade: Número de unidades que a pessoa deseja utilizar (>= 1).
    """
    if type(quantidade) is not int or quantidade < 1:
        return {"erro": "A quantidade solicitada deve ser um inteiro positivo."}
    gpu = get_gpu(modelo)
    if gpu is None:
        return {"erro": f"Modelo não encontrado na infraestrutura simulada: {modelo}"}
    return {
        "modelo": gpu["modelo"],
        "quantidade_solicitada": quantidade,
        "quantidade_disponivel": gpu["quantidade_disponivel"],
        "atende_solicitacao": gpu["quantidade_disponivel"] >= quantidade,
    }


def consultar_resumo() -> dict:
    """Retorna número de modelos cadastrados e soma das GPUs disponíveis agora.

    Não representa o inventário total de hardware nem unidades em uso.
    """
    return get_summary()


READ_ONLY_TOOLS = (consultar_gpus, verificar_disponibilidade, consultar_resumo)
TOOLS_BY_NAME = {tool.__name__: tool for tool in READ_ONLY_TOOLS}


# GPU Assistant — Desafio HP 2026

Projeto desenvolvido para um desafio técnico de estágio em Machine Learning e IA na HP. O agente responde, em linguagem natural, a perguntas sobre a disponibilidade de GPUs em uma infraestrutura simulada. É possível consultar modelos, memória por unidade, estoque por modelo, totais e suficiência para uma quantidade solicitada.

## Decisões do projeto

- **Python e SQLite.** Python concentra a integração com o modelo, a validação dos argumentos e a execução das consultas. Com cinco modelos de GPU, SQLite oferece persistência em um arquivo local e dispensa a configuração de um servidor PostgreSQL.
- **Ollama e execução local.** Consideramos a API da OpenAI e escolhemos rodar o modelo no computador disponível para o desafio. Começamos com o Llama 3.2 3B e passamos para o **Qwen3 4B (`qwen3:4b`)** após observar, nos testes locais, respostas mais coerentes a saudações e perguntas sobre o papel do assistente. O modelo continua configurável para comparar a seleção de ferramentas e o tempo de resposta.
- **Estoque somente leitura para o agente.** O Python abre o SQLite em modo de leitura nas consultas. Um script separado altera a quantidade disponível para demonstrar como o agente responde a mudanças no estoque.
- **Duas interfaces para o mesmo agente.** O Streamlit oferece chat no navegador e mostra o modelo em uso e os totais na barra lateral. A CLI permite testar o mesmo fluxo pelo terminal, inclusive com outro modelo já baixado.

## Dados e escopo

O banco armazena o nome do modelo, a memória por GPU e a quantidade de unidades livres naquele momento. A quantidade física total, as GPUs em uso, as reservas e a disponibilidade futura ficam fora do escopo. Confirmar que há unidades suficientes significa verificar o estoque atual.

| GPU | Memória por unidade | Unidades livres inicialmente |
| --- | ---: | ---: |
| NVIDIA A100 | 80 GB | 4 |
| NVIDIA L40S | 48 GB | 6 |
| NVIDIA RTX 4090 | 24 GB | 3 |
| NVIDIA H100 | 80 GB | 2 |
| NVIDIA A40 | 48 GB | 4 |

A base inicial tem **cinco modelos e 19 unidades livres**.

## Como o agente funciona

O [agente Python](gpu_assistant/agent.py) envia ao Ollama o contexto da conversa e os esquemas das ferramentas autorizadas. Quando o modelo pede uma consulta, o agente valida o nome e os argumentos, executa a função e devolve o resultado ao modelo. Se a consulta falhar, o agente envia um diagnóstico e permite até três tentativas de correção no turno. A resposta pode vir como texto ou pela ferramenta `conversar_com_cliente`.

O prompt define o papel do assistente, o uso do histórico e o limite sobre reservas. As [descrições das ferramentas](gpu_assistant/tools.py) especificam o que cada função recebe e devolve. Essa separação permite ajustar a seleção de ferramentas sem alterar as consultas ao banco.

| Ferramenta | Finalidade |
| --- | --- |
| `consultar_gpus` | Lista modelos ou retorna memória por unidade e quantidade livre por modelo. Aceita filtros por nome, memória ou presença de unidades livres. |
| `verificar_disponibilidade` | Recebe um modelo e uma quantidade. Retorna o número de unidades livres e `atende_solicitacao`, que indica se há pelo menos a quantidade pedida. |
| `consultar_resumo` | Conta os modelos cadastrados e soma as unidades livres. |
| `comparar_gpus` | Encontra a maior ou a menor memória por unidade, ou a maior ou a menor disponibilidade, incluindo empates. |
| `conversar_com_cliente` | Entrega uma mensagem, como uma saudação, uma resposta final ou um pedido de esclarecimento. |

## Como executar

É preciso ter **Python 3.10 ou superior** e [Ollama](https://ollama.com/download) instalado. O [Qwen3 4B](https://ollama.com/library/qwen3:4b) usa a memória e o processamento do computador local. A velocidade varia conforme o hardware.

### 1. Prepare o ambiente

Na raiz do repositório clonado, use os comandos do seu sistema.

**Windows (PowerShell)**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

**Linux ou macOS**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

### 2. Baixe o modelo e inicialize o banco

Com o Ollama em execução, rode:

```bash
ollama pull qwen3:4b
python init_db.py
```

Se o serviço do Ollama estiver parado, inicie `ollama serve` em outro terminal. `python init_db.py` preserva as quantidades já atualizadas. `python init_db.py --reset` restaura as quantidades iniciais dos cinco modelos.

O `.env.example` já configura `OLLAMA_MODEL=qwen3:4b`, `OLLAMA_HOST=http://localhost:11434` e `GPU_DB_PATH=data/gpus.sqlite3`. Para trocar de modelo, baixe-o pelo Ollama e edite `OLLAMA_MODEL` no `.env`. O valor no arquivo tem prioridade para a seleção do modelo, e a CLI aceita a opção explícita `--model <nome-do-modelo>`.

### 3. Abra o chat

**No navegador**, execute o Streamlit com o Python do ambiente virtual e abra a URL exibida no terminal.

```bash
python -m streamlit run app.py
```

**No terminal**, digite `sair` para encerrar o chat.

```bash
python chat_cli.py
```

## Como validar o comportamento

Comece com “Quais GPUs estão disponíveis?”. Em seguida, compare “Quantas A100 estão livres?” com “Tem duas A100 livres?”. A primeira pergunta pede o total do modelo. A segunda verifica se o estoque atende a uma quantidade específica. Pergunte “E se forem cinco?” no mesmo chat para testar o uso do histórico. Para conferir o escopo, pergunte se o assistente pode reservar uma GPU.

Para simular uma mudança, rode o comando abaixo em outro terminal com o ambiente virtual ativo. Ele define a disponibilidade da A100 em duas unidades. Depois, pergunte novamente pelo total de A100 livres.

```bash
python update_availability.py A100 2
```

As duas interfaces registram requisições e respostas do Ollama em `conversas.log`, na raiz do projeto. Os registros em JSON incluem o modelo, o histórico enviado, as chamadas de ferramentas, os argumentos e as métricas retornadas pelo Ollama. Cada par de requisição e resposta compartilha o mesmo `request_id`. Nos testes manuais, esse log permite conferir se uma resposta correta veio da ferramenta esperada.

Os testes automatizados cobrem o banco, as ferramentas e a orquestração com respostas simuladas do Ollama. Eles rodam sem o serviço e sem baixar o modelo.

```bash
python -m unittest discover -s tests -v
```

# GPU Assistant

 O agente responde, em linguagem natural, a perguntas sobre a disponibilidade de GPUs em uma infraestrutura simulada. É possível consultar modelos, memória por unidade, estoque por modelo, totais e suficiência para uma quantidade solicitada.

## Decisões do projeto

- **Python e SQLite.** Python foi utilizado para integrar o modelo, validar os argumentos e executar as consultas. O SQLite foi adotado para armazenar o estoque dos cinco modelos de GPU em um arquivo local.
- **Ollama e execução local.** Após considerar a API da OpenAI, optou-se pela execução local com Ollama. Inicialmente, foi utilizado o Llama 3.2 3B. Nos testes locais, o **Qwen3 4B (`qwen3:4b`)** apresentou respostas mais coerentes a saudações e perguntas sobre o papel do assistente e passou a ser o modelo padrão. A configuração permite testar outros modelos e comparar a seleção de ferramentas e o tempo de resposta.
- **Estoque somente leitura para o agente.** O acesso do agente ao SQLite foi limitado a consultas em modo somente leitura. Um script separado permite alterar a quantidade disponível e demonstrar como o agente responde a mudanças no estoque.
- **Duas interfaces para o mesmo agente.** Foi implementado um chat em Streamlit, com o modelo em uso e os totais na barra lateral. A interface de linha de comando (CLI) permite testar o mesmo fluxo pelo terminal, inclusive com outro modelo já baixado.

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

O agente Python envia ao Ollama o contexto da conversa e os esquemas das ferramentas autorizadas. O modelo decide quando consultar o estoque e qual ferramenta usar. Quando o modelo pede uma consulta, o agente valida a chamada, consulta o SQLite e devolve o resultado ao modelo. O modelo redige a resposta e pode entregá-la em texto ou pela ferramenta `conversar_com_cliente`. [Código do agente](gpu_assistant/agent.py)

Foram usadas expressões regulares (regex) para reconhecer valores enviados como texto, como `"2"` e `"true"`, e convertê-los para inteiro ou booleano conforme o esquema da ferramenta. Se uma chamada falhar, o agente devolve ao modelo um diagnóstico com os argumentos recebidos e os tipos esperados. O modelo tem até três tentativas de correção por turno. Ao esgotar esse limite, o agente pede ao modelo que explique o que ficou sem confirmação.

O prompt define o papel do assistente, o uso do histórico e o limite sobre reservas. Em perguntas sobre disponibilidade, o modelo é orientado a consultar o estoque no turno atual. O esquema de cada ferramenta descreve o que ela recebe e devolve. As descrições ficam separadas da lógica das consultas e podem ser ajustadas para orientar a seleção de ferramentas. [Esquemas das ferramentas](gpu_assistant/tools.py)

| Ferramenta | Finalidade |
| --- | --- |
| `consultar_gpus` | Lista modelos ou retorna memória por unidade e quantidade livre por modelo. Aceita filtros por nome, memória ou presença de unidades livres. |
| `verificar_disponibilidade` | Recebe um modelo e uma quantidade. Para um modelo cadastrado, retorna o número de unidades livres e `atende_solicitacao`, que indica se há pelo menos a quantidade pedida. |
| `consultar_resumo` | Conta os modelos cadastrados e soma as unidades livres. |
| `comparar_gpus` | Encontra a maior ou a menor memória por unidade, ou a maior ou a menor disponibilidade, incluindo empates. |
| `conversar_com_cliente` | Entrega a mensagem escrita pelo modelo, como uma saudação, uma resposta final ou um pedido de esclarecimento. |

## Como executar

É preciso ter **Python 3.10 ou superior** e [Ollama](https://ollama.com/download) instalado. O [Qwen3 4B](https://ollama.com/library/qwen3:4b) usa a memória e o processamento do computador local. A velocidade varia conforme o hardware.

### Ambiente dos testes locais

Os testes locais com o Qwen3 4B foram realizados no **Windows 11 25H2** com **Intel Core i7-11800H**, **16 GB de RAM**, **NVIDIA GeForce GTX 1650 com 4 GB de VRAM** e **Ollama 0.40.16**. Nessa máquina, o Ollama distribuiu o modelo entre a CPU e a GPU. Durante os testes manuais, algumas respostas levaram dezenas de segundos.

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

Para testar uma mudança de estoque, parta da base inicial, com quatro A100 livres. Pergunte “Tem duas A100 livres?” e confira se a resposta indica estoque suficiente. Em outro terminal, com o ambiente virtual ativo, reduza o estoque para uma unidade.

```bash
python update_availability.py A100 1
```

Repita a pergunta no mesmo chat. A resposta esperada indica uma A100 livre, quantidade insuficiente para o pedido. No teste manual com o Qwen3 4B, o log registrou uma chamada a `verificar_disponibilidade` em cada turno, e as respostas refletiram a mudança de estoque.

As duas interfaces registram requisições e respostas do Ollama em `conversas.log`, na raiz do projeto. Os registros em JSON incluem o modelo, o histórico enviado, as chamadas de ferramentas, os argumentos e as métricas retornadas pelo Ollama. Cada par de requisição e resposta compartilha o mesmo `request_id`.

Os testes automatizados cobrem o banco, as ferramentas e a orquestração com respostas simuladas do Ollama. Eles rodam sem o serviço e sem baixar o modelo.

```bash
python -m unittest discover -s tests -v
```

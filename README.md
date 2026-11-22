# GPU Assistant — Desafio HP 2026

Assistente conversacional em Python para **consultar a disponibilidade simulada de GPUs**. Utiliza SQLite como fonte de dados, funções de consulta expostas ao LLM por **Tool Calling**, Ollama para executar modelos locais e uma interface opcional em Streamlit.

> Projeto-base em desenvolvimento. O modelo de linguagem não foi definido como vencedor: seu nome é configurável para permitir experimentos comparativos. O funcionamento com um LLM real deve ser validado depois da escolha dos candidatos.

## Funcionalidades

- Listar modelos de GPUs cadastrados.
- Consultar memória **por GPU** e quantidade **disponível agora**, para um ou vários modelos.
- Verificar se existe a quantidade solicitada de um modelo.
- Consultar o número de modelos cadastrados e o total **disponível** de unidades.
- Demonstrar a atualização da disponibilidade com um script externo ao agente.
- Trocar o modelo de linguagem sem modificar o código-fonte.

**Escopo:** os dados não incluem quantidade física total, GPUs em uso, servidores, reservas ou disponibilidade futura. O agente nunca recebe ferramentas para modificar o banco de dados.

## Dados simulados

| GPU | Disponíveis inicialmente | Memória por GPU |
| --- | ---: | ---: |
| NVIDIA A100 | 4 | 80 GB |
| NVIDIA L40S | 6 | 48 GB |
| NVIDIA RTX 4090 | 3 | 24 GB |
| NVIDIA H100 | 2 | 80 GB |
| NVIDIA A40 | 4 | 48 GB |

São inicialmente **19 GPUs disponíveis** distribuídas em cinco modelos. A inicialização usa esses valores para reproduzir a tabela do enunciado.

## Como funciona

1. O usuário faz uma pergunta no Streamlit ou no terminal.
2. O modelo recebe o histórico, a última mensagem do usuário e as descrições das ferramentas autorizadas. Ele decide se precisa consultar dados.
3. Quando o modelo solicita uma função, o Python valida seu nome e argumentos.
4. A função consulta o SQLite em modo **somente leitura**.
5. O resultado volta ao modelo. Se houver erro na chamada, o agente envia o erro e exemplos da ferramenta para orientar até três novas tentativas por pergunta.
6. O modelo escreve a mensagem e usa `conversar_com_cliente` para entregá-la. A mensagem pode responder à consulta, apresentar o assistente ou pedir esclarecimento.

Durante uma nova tentativa, o agente pode decodificar uma lista JSON de nomes que o Ollama enviou dentro de uma string. A ferramenta valida os nomes e filtros após a conversão, e os argumentos normalizados aparecem no resultado registrado no log.

Quando falta contexto, o agente pode pedir esclarecimento antes de consultar o SQLite. Para um pedido como “histórico”, ele pergunta se o usuário quer rever as consultas da sessão ou saber como a disponibilidade mudou. A base contém a disponibilidade atual. Uma verificação de suficiência exige uma quantidade pedida pelo usuário.

Uma lista de GPUs disponíveis inclui modelos com pelo menos uma unidade livre. As comparações de maior ou menor memória e maior disponibilidade incluem todos os empates.

As consultas ficam agrupadas em `GpuQueryTools`. A descrição do grupo orienta o modelo a usar a consulta pertinente quando o usuário pedir dados sobre GPUs. `conversar_com_cliente` é uma ferramenta geral de conversa.

Ferramentas disponíveis

| Ferramenta | Descrição |
| --- | --- |
| `consultar_gpus` | Retorna modelo, quantidade disponível e memória por GPU. Use `modelo` para um nome e `modelos` para uma lista. Por memória, use `memoria_exata_gb` ou `memoria_minima_gb`. |
| `verificar_disponibilidade(modelo, quantidade)` | Compara a quantidade solicitada com a quantidade disponível no banco. |
| `consultar_resumo()` | Retorna número de modelos e soma das unidades disponíveis. |
| `comparar_gpus(criterio, somente_disponiveis=False)` | Retorna os modelos com maior ou menor memória por unidade ou maior ou menor quantidade disponível, incluindo empates. |
| `conversar_com_cliente(mensagem)` | Encerra o uso de ferramentas no turno e entrega a mensagem escrita pelo modelo. |

Somente a aplicação executa as funções autorizadas sobre o SQLite. O prompt base define o papel do assistente e como usar o histórico. Os esquemas enviados no campo `tools` descrevem quando consultar e como usar os resultados.

## Pré-requisitos

- Python **3.10 ou superior** (recomendado 3.11 ou 3.12).
- [Ollama](https://ollama.com/download) instalado e em execução, para o chat com LLM local.
- Um modelo **compatível com Tool Calling** baixado no Ollama.
- Memória RAM/VRAM suficiente para o modelo escolhido. Uma GPU dedicada não é obrigatória para todos os modelos, mas usar apenas CPU pode ser mais lento.

**Atenção:** o modelo é executado no computador da pessoa que testa a aplicação; o repositório Git não contém os pesos do modelo. Os testes automatizados não precisam do Ollama.

## Instalação

Abra um terminal na pasta do projeto e crie um ambiente virtual.

No Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

No Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Inicialize os cinco modelos do enunciado:

```bash
python init_db.py
```

Esse comando é idempotente: não sobrescreve uma disponibilidade já alterada. Para restaurar as disponibilidades iniciais dos cinco modelos:

```bash
python init_db.py --reset
```

### Configurar um LLM

No arquivo `.env`, configure:

```dotenv
OLLAMA_MODEL=llama3.2:3b
OLLAMA_HOST=http://localhost:11434
GPU_DB_PATH=data/gpus.sqlite3
```

`llama3.2:3b` é **apenas uma opção inicial para teste**, não o modelo final escolhido para a entrega. É possível usar outro modelo compatível com ferramentas, alterando `OLLAMA_MODEL` ou o campo de modelo na interface.

Com Ollama instalado, baixe o modelo que configurou, por exemplo:

```bash
ollama pull llama3.2:3b
```

No Windows, o Ollama costuma permanecer aberto em segundo plano após a instalação. No Linux, se o serviço não estiver rodando, execute `ollama serve` em outro terminal.

### Executar a interface web

```bash
streamlit run app.py
```

### Ou conversar pelo terminal

```bash
python chat_cli.py
```

Para comparar modelos sem editar `.env`:

```bash
python chat_cli.py --model llama3.2:3b
```

### Consultar o log das conversas

O terminal e o Streamlit gravam as chamadas ao Ollama em `conversas.log`, na raiz do projeto. O arquivo é criado na primeira chamada e acumula registros em UTF-8, com um objeto JSON por linha. O Git ignora esse arquivo.

- `request` contém o modelo, o prompt de sistema, o histórico, as definições das ferramentas, os resultados já consultados e as opções enviadas ao Ollama.
- `response` contém a resposta original do Ollama, inclusive chamadas de ferramentas com seus argumentos, antes de o agente decidir o que mostrar no chat. O mesmo `request_id` liga a resposta à requisição.

Para acompanhar as últimas chamadas pelo PowerShell, depois de fazer uma pergunta:

```powershell
Get-Content -Encoding UTF8 .\conversas.log -Tail 6 -Wait
```

## Simular alteração de disponibilidade

Somente o script externo modifica a coluna `quantidade_disponivel`. Por exemplo:

```bash
python update_availability.py "NVIDIA A100" 2
```

Depois, pergunte ao agente quantas A100 estão disponíveis: a consulta deverá usar o novo valor. A memória de cada GPU e os nomes dos modelos permanecem inalterados.

## Testes automatizados (sem LLM)

```bash
python -m unittest discover -s tests -v
```

Esses testes verificam a inicialização, os registros, os aliases dos modelos, a disponibilidade, atualizações externas, o modo somente leitura e um ciclo de Tool Calling **simulado**. Não demonstram, por si só, que um modelo real já foi avaliado. Essa etapa dependerá da escolha dos modelos candidatos.

## Exemplos de perguntas

- Quais GPUs estão disponíveis?
- Qual a memória da A100 e da H100?
- Quantas RTX 4090 estão disponíveis?
- Existem duas A100 disponíveis?
- Quantas GPUs estão disponíveis no total?
- Quais GPUs têm pelo menos 48 GB?
- Quantas GPUs estão em uso? *(a resposta correta deve indicar que esse dado não está disponível)*

## Organização

```text
hp-gpu-assistant/
├── app.py                       # Chat via navegador
├── chat_cli.py                  # Chat via terminal
├── init_db.py                   # Cria e popula o SQLite
├── update_availability.py       # Atualização administrativa separada
├── gpu_assistant/
│   ├── agent.py                 # Orquestração e Tool Calling
│   ├── config.py                # Variáveis de configuração
│   ├── database.py              # Acesso ao SQLite
│   └── tools.py                 # Ferramentas de consulta
├── tests/                       # Testes offline
├── data/                        # SQLite local (não versionado)
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

## Decisões e limitações

- **SQLite:** são apenas cinco modelos e consultas locais; não precisamos manter um servidor PostgreSQL.
- **Sem treinamento/fine-tuning:** o desafio trata da integração do LLM com dados externos, não de memorizar dados mutáveis.
- **Modelo configurável:** permite avaliar precisão de Tool Calling, tempo de resposta e consumo de recursos antes de escolher um modelo final.
- **Somente leitura no agente:** operações de escrita ficam em um script separado.
- **Sem reservas:** não existe no enunciado uma regra de reserva, agenda ou utilização; não fabricamos esses dados.
- **Interface simples:** o frontend é complementar à integração com o modelo.
- **Respostas factuais:** o LLM interpreta a pergunta, seleciona as consultas e escreve a mensagem usando os dados retornados.

## Referências técnicas

- [Ollama — instalação e quickstart](https://docs.ollama.com/quickstart)
- [Ollama — Tool Calling](https://docs.ollama.com/capabilities/tool-calling)
- [SQLite — quando utilizar](https://www.sqlite.org/whentouse.html)


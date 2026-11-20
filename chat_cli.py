"""Interface de terminal opcional para testar o agente sem Streamlit."""

import argparse

from gpu_assistant.agent import GpuAgent
from gpu_assistant.config import get_model_name


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat local de consultas a GPUs.")
    parser.add_argument("--model", default=get_model_name(), help="Nome do modelo no Ollama")
    args = parser.parse_args()
    agent = GpuAgent(model_name=args.model)
    history = []
    print(f"GPU Assistant | modelo: {args.model} | digite 'sair' para encerrar")
    while True:
        try:
            question = input("Você: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if question.casefold() in ("sair", "exit", "quit"):
            break
        try:
            answer, history = agent.answer(question, history)
        except Exception as exc:
            print(f"Falha na conexão com o modelo: {exc}")
            continue
        print(f"Assistente: {answer}\n")


if __name__ == "__main__":
    main()


"""Altera SOMENTE o número de unidades disponíveis, fora do agente."""

import argparse

from gpu_assistant.database import update_availability


def main() -> None:
    parser = argparse.ArgumentParser(description="Atualiza a disponibilidade simulada.")
    parser.add_argument("modelo", help='Modelo, como "NVIDIA A100" ou "A100"')
    parser.add_argument("quantidade", type=int, help="Nova quantidade disponível (>= 0)")
    args = parser.parse_args()
    try:
        result = update_availability(args.modelo, args.quantidade)
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))
    print(f"{result['modelo']}: {result['quantidade_disponivel']} disponíveis agora.")


if __name__ == "__main__":
    main()


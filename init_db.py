"""Inicializa os cinco registros do enunciado, sem sobrescrever atualizações."""

import argparse

from gpu_assistant.database import GpuDatabase


def main() -> None:
    parser = argparse.ArgumentParser(description="Inicializa a base simulada de GPUs.")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Restaura a disponibilidade inicial dos cinco modelos do enunciado.",
    )
    args = parser.parse_args()
    database = GpuDatabase()
    database.initialize_db(reset_availability=args.reset)
    summary = database.get_summary()
    print(f"Banco pronto: {database.db_path}")
    print(f"{summary['modelos']} modelos; {summary['total_disponivel']} unidades disponíveis.")


if __name__ == "__main__":
    main()


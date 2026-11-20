"""Configurações portáveis, sem caminhos específicos de um computador."""

import os
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env", override=False)


def get_db_path() -> Path:
    """Retorna o arquivo SQLite configurado (relativo à raiz, por padrão)."""
    configured = Path(os.getenv("GPU_DB_PATH", "data/gpus.sqlite3"))
    return configured if configured.is_absolute() else PROJECT_ROOT / configured


def get_model_name() -> str:
    return os.getenv("OLLAMA_MODEL", "llama3.2:3b").strip()


def get_ollama_host() -> str:
    return os.getenv("OLLAMA_HOST", "http://localhost:11434").strip()


"""Configurações portáveis, sem caminhos específicos de um computador."""

import os
from pathlib import Path

from dotenv import dotenv_values, load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ".env"
load_dotenv(ENV_PATH, override=False)


def get_db_path() -> Path:
    """Retorna o arquivo SQLite configurado (relativo à raiz, por padrão)."""
    configured = Path(os.getenv("GPU_DB_PATH", "data/gpus.sqlite3"))
    return configured if configured.is_absolute() else PROJECT_ROOT / configured


def get_model_name() -> str:
    # Leia a configuração do projeto a cada rerun do Streamlit.
    configured = (dotenv_values(ENV_PATH).get("OLLAMA_MODEL") or "").strip()
    return configured or os.getenv("OLLAMA_MODEL", "").strip() or "qwen3:4b"


def get_ollama_host() -> str:
    return os.getenv("OLLAMA_HOST", "http://localhost:11434").strip()


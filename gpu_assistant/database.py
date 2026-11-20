"""Acesso ao SQLite; somente os scripts administrativos realizam escritas."""

import sqlite3
from pathlib import Path

from .config import get_db_path


INITIAL_GPUS = (
    ("NVIDIA A100", 4, 80),
    ("NVIDIA L40S", 6, 48),
    ("NVIDIA RTX 4090", 3, 24),
    ("NVIDIA H100", 2, 80),
    ("NVIDIA A40", 4, 48),
)


def _path(db_path: Path | None = None) -> Path:
    return db_path if db_path is not None else get_db_path()


def connect(*, read_only: bool = True, db_path: Path | None = None) -> sqlite3.Connection:
    """Abre a base em modo somente leitura, exceto nos scripts de manutenção."""
    path = _path(db_path)
    if read_only:
        if not path.is_file():
            raise FileNotFoundError("Banco não inicializado. Execute: python init_db.py")
        # O parâmetro mode=ro impede escritas mesmo se alguém errar uma consulta.
        conn = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def initialize_db(*, reset_availability: bool = False, db_path: Path | None = None) -> None:
    """Cria e popula a base sem substituir dados existentes por padrão.

    --reset redefine apenas a quantidade disponível dos cinco modelos originais.
    """
    with connect(read_only=False, db_path=db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS gpus (
                modelo TEXT PRIMARY KEY,
                quantidade_disponivel INTEGER NOT NULL
                    CHECK (quantidade_disponivel >= 0),
                memoria_gb INTEGER NOT NULL CHECK (memoria_gb > 0)
            )
            """
        )
        for modelo, quantidade, memoria in INITIAL_GPUS:
            if reset_availability:
                conn.execute(
                    """
                    INSERT INTO gpus (modelo, quantidade_disponivel, memoria_gb)
                    VALUES (?, ?, ?)
                    ON CONFLICT(modelo) DO UPDATE SET
                        quantidade_disponivel = excluded.quantidade_disponivel
                    """,
                    (modelo, quantidade, memoria),
                )
            else:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO gpus
                        (modelo, quantidade_disponivel, memoria_gb)
                    VALUES (?, ?, ?)
                    """,
                    (modelo, quantidade, memoria),
                )


def list_gpus(*, db_path: Path | None = None) -> list[dict]:
    with connect(db_path=db_path) as conn:
        rows = conn.execute(
            """
            SELECT modelo, quantidade_disponivel, memoria_gb
            FROM gpus ORDER BY modelo
            """
        ).fetchall()
    return [dict(row) for row in rows]


def get_gpu(modelo: str, *, db_path: Path | None = None) -> dict | None:
    """Aceita nomes completos ou abreviados, como A100, L40S e 4090."""
    if not isinstance(modelo, str) or not modelo.strip():
        return None
    normalized = "".join(ch for ch in modelo.casefold() if ch.isalnum())
    for gpu in list_gpus(db_path=db_path):
        full = "".join(ch for ch in gpu["modelo"].casefold() if ch.isalnum())
        short = full.removeprefix("nvidia")
        aliases = {full, short}
        if short.startswith("rtx"):
            aliases.add(short.removeprefix("rtx"))
        if normalized in aliases:
            return gpu
    return None


def get_summary(*, db_path: Path | None = None) -> dict:
    with connect(db_path=db_path) as conn:
        row = conn.execute(
            """
            SELECT COUNT(*) AS modelos,
                   COALESCE(SUM(quantidade_disponivel), 0) AS total_disponivel
            FROM gpus
            """
        ).fetchone()
    return dict(row)


def update_availability(modelo: str, quantidade: int, *, db_path: Path | None = None) -> dict:
    """Uso exclusivo do script administrativo; nunca expor como ferramenta ao LLM."""
    if type(quantidade) is not int or quantidade < 0:
        raise ValueError("A quantidade disponível deve ser um inteiro não negativo.")
    existing = get_gpu(modelo, db_path=db_path)
    if existing is None:
        raise ValueError(f"Modelo de GPU não cadastrado: {modelo}")
    with connect(read_only=False, db_path=db_path) as conn:
        conn.execute(
            "UPDATE gpus SET quantidade_disponivel = ? WHERE modelo = ?",
            (quantidade, existing["modelo"]),
        )
    return {"modelo": existing["modelo"], "quantidade_disponivel": quantidade}


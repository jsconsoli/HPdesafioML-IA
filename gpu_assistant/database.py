"""Acesso ao SQLite para consultas e manutenção administrativa."""

import sqlite3
from contextlib import closing
from pathlib import Path

from .config import get_db_path


class GpuDatabase:
    """Mantém o caminho do banco e abre conexões por operação."""

    _initial_gpus = (
        ("NVIDIA A100", 4, 80),
        ("NVIDIA L40S", 6, 48),
        ("NVIDIA RTX 4090", 3, 24),
        ("NVIDIA H100", 2, 80),
        ("NVIDIA A40", 4, 48),
    )

    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = db_path if db_path is not None else get_db_path()

    def connect(self, *, read_only: bool = True) -> sqlite3.Connection:
        """Abre uma conexão somente leitura por padrão."""
        path = self.db_path
        if read_only:
            if not path.is_file():
                raise FileNotFoundError("Banco não inicializado. Execute: python init_db.py")
            conn = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(path)
        conn.row_factory = sqlite3.Row
        return conn

    def initialize_db(self, *, reset_availability: bool = False) -> None:
        """Cria e popula a base, com redefinição opcional das quantidades iniciais."""
        with closing(self.connect(read_only=False)) as conn, conn:
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
            for modelo, quantidade, memoria in self._initial_gpus:
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

    def list_gpus(self) -> list[dict]:
        with closing(self.connect()) as conn:
            rows = conn.execute(
                """
                SELECT modelo, quantidade_disponivel, memoria_gb
                FROM gpus ORDER BY modelo
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def get_gpu(self, modelo: str) -> dict | None:
        """Aceita nomes completos ou abreviados, como A100, L40S e 4090."""
        if not isinstance(modelo, str) or not modelo.strip():
            return None
        normalized = "".join(ch for ch in modelo.casefold() if ch.isalnum())
        for gpu in self.list_gpus():
            full = "".join(ch for ch in gpu["modelo"].casefold() if ch.isalnum())
            short = full.removeprefix("nvidia")
            aliases = {full, short}
            if short.startswith("rtx"):
                aliases.add(short.removeprefix("rtx"))
            if normalized in aliases:
                return gpu
        return None

    def get_summary(self) -> dict:
        with closing(self.connect()) as conn:
            row = conn.execute(
                """
                SELECT COUNT(*) AS modelos,
                       COALESCE(SUM(quantidade_disponivel), 0) AS total_disponivel
                FROM gpus
                """
            ).fetchone()
        return dict(row)

    def update_availability(self, modelo: str, quantidade: int) -> dict:
        """Atualiza a quantidade disponível para uso administrativo."""
        if type(quantidade) is not int or quantidade < 0:
            raise ValueError("A quantidade disponível deve ser um inteiro não negativo.")
        existing = self.get_gpu(modelo)
        if existing is None:
            raise ValueError(f"Modelo de GPU não cadastrado: {modelo}")
        with closing(self.connect(read_only=False)) as conn, conn:
            conn.execute(
                "UPDATE gpus SET quantidade_disponivel = ? WHERE modelo = ?",
                (quantidade, existing["modelo"]),
            )
        return {"modelo": existing["modelo"], "quantidade_disponivel": quantidade}


def connect(*, read_only: bool = True, db_path: Path | None = None) -> sqlite3.Connection:
    return GpuDatabase(db_path).connect(read_only=read_only)


def initialize_db(*, reset_availability: bool = False, db_path: Path | None = None) -> None:
    GpuDatabase(db_path).initialize_db(reset_availability=reset_availability)


def list_gpus(*, db_path: Path | None = None) -> list[dict]:
    return GpuDatabase(db_path).list_gpus()


def get_gpu(modelo: str, *, db_path: Path | None = None) -> dict | None:
    return GpuDatabase(db_path).get_gpu(modelo)


def get_summary(*, db_path: Path | None = None) -> dict:
    return GpuDatabase(db_path).get_summary()


def update_availability(modelo: str, quantidade: int, *, db_path: Path | None = None) -> dict:
    return GpuDatabase(db_path).update_availability(modelo, quantidade)


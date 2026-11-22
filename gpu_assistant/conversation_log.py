"""Registra os corpos das chamadas ao Ollama para diagnóstico local."""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from os import SEEK_END
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING
from uuid import uuid4

from .config import PROJECT_ROOT

if TYPE_CHECKING:
    from httpx import Request, Response

if sys.platform == "win32":
    import msvcrt


class ConversationLog:
    """Registra cada chamada ao Ollama como um par de eventos JSON."""

    _REQUEST_ID_KEY = "gpu_assistant_request_id"

    def __init__(self, path: Path | None = None) -> None:
        self.path = path if path is not None else PROJECT_ROOT / "conversas.log"
        self._write_lock = Lock()
        self._logger = logging.getLogger(__name__)

    def _append_event(self, event: str, request_id: str, body: bytes, **metadata: int) -> None:
        try:
            payload = json.loads(body)
        except ValueError:
            payload = body.decode("utf-8", errors="replace")

        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "request_id": request_id,
            "event": event,
            "payload": payload,
            **metadata,
        }
        line = (json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8", errors="backslashreplace")
        try:
            with self._write_lock, self.path.open("ab", buffering=0) as log:
                if sys.platform == "win32":
                    # O bloqueio também coordena escritores de outras instâncias e processos.
                    # O byte fica além do conteúdo, permitindo que o log seja lido.
                    log.seek(1 << 62)
                    msvcrt.locking(log.fileno(), msvcrt.LK_LOCK, 1)
                    log.seek(0, SEEK_END)
                log.write(line)
        except OSError as exc:
            self._logger.warning("Falha ao gravar o log em %s (%s).", self.path, exc)

    def log_request(self, request: Request) -> None:
        request_id = uuid4().hex
        request.extensions[self._REQUEST_ID_KEY] = request_id
        self._append_event("request", request_id, request.content)

    def log_response(self, response: Response) -> None:
        # O chat usa stream=False. O HTTPX mantém o corpo lido para o cliente Ollama.
        body = response.read()
        request_id = response.request.extensions[self._REQUEST_ID_KEY]
        self._append_event("response", request_id, body, status_code=response.status_code)

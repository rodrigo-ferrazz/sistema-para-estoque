"""O relógio da aplicação: a hora de agora, em UTC, para todo módulo que grava ou compara
datas. Um só, para que os testes controlem a hora de todos com um override."""
from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime

Relogio = Callable[[], datetime]


def agora_utc() -> datetime:
    return datetime.now(UTC)


def get_relogio() -> Relogio:
    """Dependência FastAPI. Em testes, sobrescreva para controlar a hora de todos os módulos."""
    return agora_utc

from __future__ import annotations

from typing import Any
from uuid import UUID

from app.config import Settings


async def load_conversation_messages(
    *,
    settings: Settings,
    connection_factory: Any,
    conversation_id: UUID | None,
    limit: int | None = None,
) -> list[dict[str, str]]:
    if conversation_id is None:
        return []
    message_limit = limit or settings.agent_conversation_message_limit
    async with connection_factory() as connection:
        rows = await connection.fetch(
            """
            SELECT role, content
            FROM ai.messages
            WHERE conversation_id = $1
            ORDER BY created_at DESC
            LIMIT $2
            """,
            conversation_id,
            message_limit,
        )
    messages = [
        {"role": str(row["role"]), "content": str(row["content"])}
        for row in reversed(rows)
        if row.get("content")
    ]
    return messages

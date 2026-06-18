import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import asyncpg

from app.config import get_settings

logger = logging.getLogger(__name__)


class DatabasePool:
    _pool: asyncpg.Pool | None = None

    @classmethod
    async def create_pool(cls) -> asyncpg.Pool:
        if cls._pool is None:
            settings = get_settings()
            cls._pool = await asyncpg.create_pool(
                dsn=str(settings.database_url),
                min_size=1,
                max_size=5,
                max_queries=50000,
                max_inactive_connection_lifetime=300,
                command_timeout=30,
                timeout=5,
            )
            logger.info("Database pool created")
        return cls._pool

    @classmethod
    async def close_pool(cls) -> None:
        if cls._pool is not None:
            await cls._pool.close()
            cls._pool = None
            logger.info("Database pool closed")


@asynccontextmanager
async def get_db_connection(use_transaction: bool = True) -> AsyncIterator[asyncpg.Connection]:
    pool = await DatabasePool.create_pool()
    async with pool.acquire() as connection:
        if use_transaction:
            async with connection.transaction():
                yield connection
        else:
            yield connection

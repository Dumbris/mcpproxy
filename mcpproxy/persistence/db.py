"""SQLite database operations for tool metadata."""

import sqlite3
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Iterable

from ..models.schemas import ToolMetadata


class DatabaseManager:
    """SQLite database manager for tool metadata."""

    def __init__(self, db_path: str = "proxy.db"):
        self.db_path = Path(db_path)
        self.is_memory_db = str(db_path) == ":memory:"
        self._shared_conn = None

        if self.is_memory_db:
            # For in-memory databases, use a shared connection
            self._shared_conn = sqlite3.connect(":memory:")
            self._shared_conn.row_factory = sqlite3.Row
            self._create_tables(self._shared_conn)
        else:
            # Ensure parent directory exists for file-based databases
            if self.db_path.parent != Path('.'):
                self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._init_database()

    def _init_database(self) -> None:
        """Initialize database schema."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                self._create_tables(conn)
        except sqlite3.OperationalError as e:
            if "unable to open database file" in str(e):
                raise sqlite3.OperationalError(
                    f"Unable to create database at {self.db_path}. "
                    f"Please check that the directory exists and is writable. "
                    f"You can set MCPPROXY_DATA_DIR environment variable to specify a different location."
                ) from e
            raise

    def _create_tables(self, conn) -> None:
        """Create database tables."""
        conn.execute("""
            CREATE TABLE IF NOT EXISTS tools (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                hash TEXT NOT NULL,
                server_name TEXT NOT NULL,
                faiss_vector_id INTEGER UNIQUE,
                params_json TEXT
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_tools_hash ON tools(hash)")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_tools_server ON tools(server_name)"
        )
        conn.commit()

    @asynccontextmanager
    async def get_connection(self):
        """Get async database connection."""
        if self.is_memory_db:
            # Use shared connection for in-memory databases
            yield self._shared_conn
        else:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            try:
                yield conn
            finally:
                conn.close()

    async def insert_tool(self, tool: ToolMetadata) -> int:
        """Insert new tool metadata."""
        async with self.get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO tools (name, description, hash, server_name, faiss_vector_id, params_json)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                (
                    tool.name,
                    tool.description,
                    tool.hash,
                    tool.server_name,
                    tool.faiss_vector_id,
                    tool.params_json,
                ),
            )
            conn.commit()
            return cursor.lastrowid

    async def update_tool(self, tool: ToolMetadata) -> None:
        """Update existing tool metadata."""
        async with self.get_connection() as conn:
            conn.execute(
                """
                UPDATE tools 
                SET description=?, hash=?, faiss_vector_id=?, params_json=?
                WHERE id=?
            """,
                (
                    tool.description,
                    tool.hash,
                    tool.faiss_vector_id,
                    tool.params_json,
                    tool.id,
                ),
            )
            conn.commit()

    async def get_tool_by_hash(self, hash: str) -> ToolMetadata | None:
        """Get tool by hash."""
        async with self.get_connection() as conn:
            row = conn.execute("SELECT * FROM tools WHERE hash=?", (hash,)).fetchone()
            if row:
                return ToolMetadata(**dict(row))
            return None

    async def get_all_tools(self) -> list[ToolMetadata]:
        """Get all tools."""
        async with self.get_connection() as conn:
            rows = conn.execute("SELECT * FROM tools").fetchall()
            return [ToolMetadata(**dict(row)) for row in rows]

    async def get_tools_by_server(self, server_name: str) -> list[ToolMetadata]:
        """Get tools by server name."""
        async with self.get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM tools WHERE server_name=?", (server_name,)
            ).fetchall()
            return [ToolMetadata(**dict(row)) for row in rows]

    async def delete_tools_by_server(self, server_name: str) -> None:
        """Delete all tools for a server."""
        async with self.get_connection() as conn:
            conn.execute("DELETE FROM tools WHERE server_name=?", (server_name,))
            conn.commit()

    async def delete_tool_by_id(self, tool_id: int) -> None:
        """Delete a specific tool by ID."""
        async with self.get_connection() as conn:
            conn.execute("DELETE FROM tools WHERE id=?", (tool_id,))
            conn.commit()

    async def delete_tool_by_name_and_server(self, name: str, server_name: str) -> None:
        """Delete a specific tool by name and server."""
        async with self.get_connection() as conn:
            conn.execute(
                "DELETE FROM tools WHERE name=? AND server_name=?", (name, server_name)
            )
            conn.commit()

    async def get_tools_by_ids(self, ids: Iterable[int]) -> list[ToolMetadata]:
        """Get tools by IDs."""
        if not ids:
            return []

        placeholders = ",".join("?" * len(list(ids)))
        async with self.get_connection() as conn:
            rows = conn.execute(
                f"SELECT * FROM tools WHERE id IN ({placeholders})", list(ids)
            ).fetchall()
            return [ToolMetadata(**dict(row)) for row in rows]

    async def get_max_faiss_vector_id(self) -> int | None:
        """Get the maximum faiss_vector_id from the database."""
        async with self.get_connection() as conn:
            row = conn.execute("SELECT MAX(faiss_vector_id) FROM tools").fetchone()
            return row[0] if row and row[0] is not None else None

    async def reset_database(self) -> None:
        """Reset database by dropping and recreating tables."""
        async with self.get_connection() as conn:
            conn.execute("DROP TABLE IF EXISTS tools")
            self._create_tables(conn)

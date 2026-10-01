from __future__ import annotations

import io
import re
from typing import Any, Protocol, Sequence


class BulkImportSink(Protocol):
    """Transactional destination for complete, bounded CSV chunks."""

    def begin(self) -> None: ...

    def write_rows(
        self, table: str, columns: Sequence[str], rows: Sequence[Sequence[Any]]
    ) -> int: ...

    def commit(self) -> None: ...

    def rollback(self) -> None: ...


class PostgreSQLCopySink:
    """DB-API adapter using psycopg 3 COPY or psycopg2 copy_expert."""

    _identifier = re.compile(r"^[a-z_][a-z0-9_]*$")

    def __init__(self, connection: Any) -> None:
        self.connection = connection

    def begin(self) -> None:
        # DB-API connections start transactions on first command.
        return None

    def write_rows(
        self, table: str, columns: Sequence[str], rows: Sequence[Sequence[Any]]
    ) -> int:
        if not self._identifier.fullmatch(table) or any(
            not self._identifier.fullmatch(column) for column in columns
        ):
            raise ValueError("unsafe PostgreSQL identifier")
        statement = f"COPY {table} ({', '.join(columns)}) FROM STDIN"
        buffer = io.StringIO()
        for row in rows:
            fields = []
            for value in row:
                if value is None:
                    fields.append(r"\N")
                    continue
                field = str(value)
                fields.append(
                    field.replace("\\", r"\\")
                    .replace("\t", r"\t")
                    .replace("\n", r"\n")
                    .replace("\r", r"\r")
                )
            buffer.write("\t".join(fields) + "\n")
        payload = buffer.getvalue()
        cursor = self.connection.cursor()
        try:
            if hasattr(cursor, "copy"):
                with cursor.copy(statement) as copy:
                    copy.write(payload)
            elif hasattr(cursor, "copy_expert"):
                buffer.seek(0)
                cursor.copy_expert(statement, buffer)
            else:
                raise TypeError("connection cursor does not support PostgreSQL COPY")
        finally:
            cursor.close()
        return len(rows)

    def commit(self) -> None:
        self.connection.commit()

    def rollback(self) -> None:
        self.connection.rollback()

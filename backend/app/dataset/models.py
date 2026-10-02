from __future__ import annotations

from pathlib import Path


class DatasetError(ValueError):
    """Base error for incomplete or invalid dataset operations."""


class DatasetValidationError(DatasetError):
    def __init__(
        self,
        message: str,
        *,
        file: Path | None = None,
        row: int | None = None,
        column: str | None = None,
    ) -> None:
        location = ":".join(
            value
            for value in (
                str(file) if file else "",
                str(row) if row is not None else "",
                column or "",
            )
            if value
        )
        super().__init__(f"{location}: {message}" if location else message)
        self.file = file
        self.row = row
        self.column = column

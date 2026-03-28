"""
Коннекторы к источникам данных / Data source connectors.

Модуль предоставляет унифицированный интерфейс для чтения данных
из CSV-файлов и DuckDB. Все коннекторы возвращают Polars DataFrame.

Each connector returns a Polars DataFrame so that downstream
profiling and quality checks work with a single data format.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb
import polars as pl


class CSVConnector:
    """
    Коннектор для CSV-файлов / CSV file connector.

    Обёртка вокруг Polars `read_csv` с удобными настройками по умолчанию.
    Thin wrapper around Polars `read_csv` with sensible defaults.
    """

    def __init__(
        self,
        path: str | Path,
        separator: str = ",",
        encoding: str = "utf-8",
        has_header: bool = True,
        infer_schema_length: int = 10_000,
    ) -> None:
        self.path = Path(path)
        self.separator = separator
        self.encoding = encoding
        self.has_header = has_header
        self.infer_schema_length = infer_schema_length

    def read(self, **kwargs: Any) -> pl.DataFrame:
        """
        Прочитать CSV и вернуть Polars DataFrame.
        Read CSV file and return a Polars DataFrame.
        """
        if not self.path.exists():
            raise FileNotFoundError(
                f"Файл не найден / File not found: {self.path}"
            )

        return pl.read_csv(
            self.path,
            separator=self.separator,
            encoding=self.encoding,
            has_header=self.has_header,
            infer_schema_length=self.infer_schema_length,
            **kwargs,
        )

    def read_lazy(self, **kwargs: Any) -> pl.LazyFrame:
        """
        Ленивое чтение — полезно для больших файлов.
        Lazy scan — useful for large files where you want to push down filters.
        """
        if not self.path.exists():
            raise FileNotFoundError(
                f"Файл не найден / File not found: {self.path}"
            )

        return pl.scan_csv(
            self.path,
            separator=self.separator,
            has_header=self.has_header,
            infer_schema_length=self.infer_schema_length,
            **kwargs,
        )


class DuckDBConnector:
    """
    Коннектор к DuckDB / DuckDB connector.

    Позволяет выполнять SQL-запросы и получать результат как Polars DataFrame.
    Runs SQL queries against DuckDB and returns Polars DataFrames.
    DuckDB хранит данные in-process — никаких серверов не нужно.
    """

    def __init__(self, database: str = ":memory:") -> None:
        """
        Подключение к базе DuckDB (по умолчанию in-memory).
        Connect to a DuckDB database (in-memory by default).
        """
        self.database = database
        self._conn: duckdb.DuckDBPyConnection | None = None

    @property
    def conn(self) -> duckdb.DuckDBPyConnection:
        """Ленивая инициализация соединения / Lazy connection init."""
        if self._conn is None:
            self._conn = duckdb.connect(self.database)
        return self._conn

    def query(self, sql: str) -> pl.DataFrame:
        """
        Выполнить SQL-запрос и вернуть результат как Polars DataFrame.
        Execute a SQL query and return the result as a Polars DataFrame.
        """
        result = self.conn.execute(sql)
        # DuckDB -> Arrow -> Polars — самый быстрый путь без копирования
        arrow_table = result.fetch_arrow_table()
        return pl.from_arrow(arrow_table)

    def register_dataframe(self, name: str, df: pl.DataFrame) -> None:
        """
        Зарегистрировать Polars DataFrame как таблицу в DuckDB.
        Register a Polars DataFrame as a virtual table in DuckDB.

        Это позволяет делать SQL-запросы по DataFrame.
        After registration you can query the DataFrame with SQL.
        """
        # DuckDB умеет работать с Arrow напрямую
        arrow_table = df.to_arrow()
        self.conn.register(name, arrow_table)

    def load_csv(self, table_name: str, csv_path: str | Path) -> None:
        """
        Загрузить CSV прямо в DuckDB-таблицу.
        Load a CSV directly into a DuckDB table.
        """
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Файл не найден / File not found: {csv_path}"
            )
        self.conn.execute(
            f"CREATE OR REPLACE TABLE {table_name} "
            f"AS SELECT * FROM read_csv_auto('{csv_path}')"
        )

    def close(self) -> None:
        """Закрыть соединение / Close the connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> DuckDBConnector:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

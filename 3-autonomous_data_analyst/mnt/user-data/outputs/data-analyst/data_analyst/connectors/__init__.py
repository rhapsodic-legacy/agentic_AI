"""
Autonomous Data Analyst - Database Connectors

Unified interface for connecting to various data sources:
- SQLite, PostgreSQL, MySQL
- CSV, Excel, Parquet files
- BigQuery, Snowflake
- REST APIs
"""

from abc import ABC, abstractmethod
from typing import Optional, Any
from dataclasses import dataclass, field
from pathlib import Path
import json


@dataclass
class ConnectionConfig:
    """Database connection configuration."""
    type: str  # sqlite, postgresql, mysql, csv, excel, bigquery, etc.
    
    # For databases
    host: Optional[str] = None
    port: Optional[int] = None
    database: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None
    
    # For files
    filepath: Optional[str] = None
    
    # For cloud services
    project_id: Optional[str] = None
    credentials_path: Optional[str] = None
    
    # Additional options
    options: dict = field(default_factory=dict)


@dataclass
class QueryResult:
    """Result from a database query."""
    success: bool
    data: Optional[list[dict]] = None
    columns: Optional[list[str]] = None
    row_count: int = 0
    error: Optional[str] = None
    query: Optional[str] = None
    execution_time_ms: float = 0
    
    def to_dataframe(self):
        """Convert to pandas DataFrame."""
        import pandas as pd
        if self.data:
            return pd.DataFrame(self.data)
        return pd.DataFrame()


@dataclass
class TableSchema:
    """Schema information for a table."""
    name: str
    columns: list[dict]  # [{name, type, nullable, primary_key}, ...]
    row_count: Optional[int] = None
    sample_data: Optional[list[dict]] = None


class BaseConnector(ABC):
    """Abstract base class for database connectors."""
    
    @abstractmethod
    def connect(self) -> bool:
        """Establish connection."""
        pass
    
    @abstractmethod
    def disconnect(self) -> None:
        """Close connection."""
        pass
    
    @abstractmethod
    def execute(self, query: str) -> QueryResult:
        """Execute a query."""
        pass
    
    @abstractmethod
    def get_tables(self) -> list[str]:
        """Get list of tables."""
        pass
    
    @abstractmethod
    def get_schema(self, table: str) -> TableSchema:
        """Get schema for a table."""
        pass
    
    def get_sample_data(self, table: str, limit: int = 5) -> QueryResult:
        """Get sample data from a table."""
        return self.execute(f"SELECT * FROM {table} LIMIT {limit}")


class SQLiteConnector(BaseConnector):
    """SQLite database connector."""
    
    def __init__(self, config: ConnectionConfig):
        self.config = config
        self.conn = None
    
    def connect(self) -> bool:
        import sqlite3
        try:
            db_path = self.config.database or self.config.filepath or ":memory:"
            self.conn = sqlite3.connect(db_path)
            self.conn.row_factory = sqlite3.Row
            return True
        except Exception as e:
            print(f"SQLite connection error: {e}")
            return False
    
    def disconnect(self) -> None:
        if self.conn:
            self.conn.close()
            self.conn = None
    
    def execute(self, query: str) -> QueryResult:
        import time
        
        if not self.conn:
            return QueryResult(success=False, error="Not connected", query=query)
        
        start = time.time()
        try:
            cursor = self.conn.cursor()
            cursor.execute(query)
            
            # Check if it's a SELECT query
            if query.strip().upper().startswith("SELECT") or query.strip().upper().startswith("PRAGMA"):
                rows = cursor.fetchall()
                columns = [desc[0] for desc in cursor.description] if cursor.description else []
                data = [dict(zip(columns, row)) for row in rows]
                
                return QueryResult(
                    success=True,
                    data=data,
                    columns=columns,
                    row_count=len(data),
                    query=query,
                    execution_time_ms=(time.time() - start) * 1000,
                )
            else:
                self.conn.commit()
                return QueryResult(
                    success=True,
                    row_count=cursor.rowcount,
                    query=query,
                    execution_time_ms=(time.time() - start) * 1000,
                )
                
        except Exception as e:
            return QueryResult(success=False, error=str(e), query=query)
    
    def get_tables(self) -> list[str]:
        result = self.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
        if result.success and result.data:
            return [row['name'] for row in result.data]
        return []
    
    def get_schema(self, table: str) -> TableSchema:
        result = self.execute(f"PRAGMA table_info({table})")
        
        columns = []
        if result.success and result.data:
            for row in result.data:
                columns.append({
                    "name": row['name'],
                    "type": row['type'],
                    "nullable": not row['notnull'],
                    "primary_key": bool(row['pk']),
                    "default": row['dflt_value'],
                })
        
        # Get row count
        count_result = self.execute(f"SELECT COUNT(*) as cnt FROM {table}")
        row_count = count_result.data[0]['cnt'] if count_result.success else None
        
        # Get sample
        sample = self.get_sample_data(table, 3)
        
        return TableSchema(
            name=table,
            columns=columns,
            row_count=row_count,
            sample_data=sample.data if sample.success else None,
        )


class PostgreSQLConnector(BaseConnector):
    """PostgreSQL database connector."""
    
    def __init__(self, config: ConnectionConfig):
        self.config = config
        self.conn = None
    
    def connect(self) -> bool:
        try:
            import psycopg2
            import psycopg2.extras
            
            self.conn = psycopg2.connect(
                host=self.config.host or "localhost",
                port=self.config.port or 5432,
                database=self.config.database,
                user=self.config.username,
                password=self.config.password,
                **self.config.options,
            )
            return True
        except ImportError:
            print("psycopg2 not installed. Run: pip install psycopg2-binary")
            return False
        except Exception as e:
            print(f"PostgreSQL connection error: {e}")
            return False
    
    def disconnect(self) -> None:
        if self.conn:
            self.conn.close()
            self.conn = None
    
    def execute(self, query: str) -> QueryResult:
        import time
        import psycopg2.extras
        
        if not self.conn:
            return QueryResult(success=False, error="Not connected", query=query)
        
        start = time.time()
        try:
            cursor = self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cursor.execute(query)
            
            if query.strip().upper().startswith("SELECT"):
                rows = cursor.fetchall()
                columns = [desc[0] for desc in cursor.description] if cursor.description else []
                data = [dict(row) for row in rows]
                
                return QueryResult(
                    success=True,
                    data=data,
                    columns=columns,
                    row_count=len(data),
                    query=query,
                    execution_time_ms=(time.time() - start) * 1000,
                )
            else:
                self.conn.commit()
                return QueryResult(
                    success=True,
                    row_count=cursor.rowcount,
                    query=query,
                    execution_time_ms=(time.time() - start) * 1000,
                )
                
        except Exception as e:
            self.conn.rollback()
            return QueryResult(success=False, error=str(e), query=query)
    
    def get_tables(self) -> list[str]:
        result = self.execute("""
            SELECT table_name FROM information_schema.tables 
            WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
        """)
        if result.success and result.data:
            return [row['table_name'] for row in result.data]
        return []
    
    def get_schema(self, table: str) -> TableSchema:
        result = self.execute(f"""
            SELECT column_name, data_type, is_nullable, column_default
            FROM information_schema.columns
            WHERE table_name = '{table}'
            ORDER BY ordinal_position
        """)
        
        columns = []
        if result.success and result.data:
            for row in result.data:
                columns.append({
                    "name": row['column_name'],
                    "type": row['data_type'],
                    "nullable": row['is_nullable'] == 'YES',
                    "default": row['column_default'],
                })
        
        count_result = self.execute(f"SELECT COUNT(*) as cnt FROM {table}")
        row_count = count_result.data[0]['cnt'] if count_result.success else None
        
        sample = self.get_sample_data(table, 3)
        
        return TableSchema(
            name=table,
            columns=columns,
            row_count=row_count,
            sample_data=sample.data if sample.success else None,
        )


class CSVConnector(BaseConnector):
    """CSV file connector using pandas and DuckDB for SQL queries."""
    
    def __init__(self, config: ConnectionConfig):
        self.config = config
        self.dataframes = {}  # table_name -> DataFrame
        self.duckdb_conn = None
    
    def connect(self) -> bool:
        import pandas as pd
        
        try:
            filepath = self.config.filepath
            if not filepath:
                return False
            
            path = Path(filepath)
            
            # Load single file or directory of files
            if path.is_file():
                table_name = path.stem
                self.dataframes[table_name] = pd.read_csv(filepath)
            elif path.is_dir():
                for csv_file in path.glob("*.csv"):
                    table_name = csv_file.stem
                    self.dataframes[table_name] = pd.read_csv(csv_file)
            
            # Set up DuckDB for SQL queries
            try:
                import duckdb
                self.duckdb_conn = duckdb.connect(":memory:")
                for name, df in self.dataframes.items():
                    self.duckdb_conn.register(name, df)
            except ImportError:
                pass  # Will use pandas-based queries
            
            return len(self.dataframes) > 0
            
        except Exception as e:
            print(f"CSV connection error: {e}")
            return False
    
    def disconnect(self) -> None:
        self.dataframes = {}
        if self.duckdb_conn:
            self.duckdb_conn.close()
            self.duckdb_conn = None
    
    def execute(self, query: str) -> QueryResult:
        import time
        import pandas as pd
        
        start = time.time()
        
        # Try DuckDB first
        if self.duckdb_conn:
            try:
                result = self.duckdb_conn.execute(query).fetchdf()
                return QueryResult(
                    success=True,
                    data=result.to_dict('records'),
                    columns=list(result.columns),
                    row_count=len(result),
                    query=query,
                    execution_time_ms=(time.time() - start) * 1000,
                )
            except Exception as e:
                return QueryResult(success=False, error=str(e), query=query)
        
        # Fallback: simple query parsing for basic SELECT
        return QueryResult(
            success=False, 
            error="DuckDB not available. Install with: pip install duckdb",
            query=query
        )
    
    def get_tables(self) -> list[str]:
        return list(self.dataframes.keys())
    
    def get_schema(self, table: str) -> TableSchema:
        import pandas as pd
        
        if table not in self.dataframes:
            return TableSchema(name=table, columns=[])
        
        df = self.dataframes[table]
        
        columns = []
        for col in df.columns:
            dtype = str(df[col].dtype)
            columns.append({
                "name": col,
                "type": dtype,
                "nullable": df[col].isnull().any(),
            })
        
        sample_data = df.head(3).to_dict('records')
        
        return TableSchema(
            name=table,
            columns=columns,
            row_count=len(df),
            sample_data=sample_data,
        )
    
    def add_dataframe(self, name: str, df) -> None:
        """Add a DataFrame as a table."""
        self.dataframes[name] = df
        if self.duckdb_conn:
            self.duckdb_conn.register(name, df)


class ExcelConnector(CSVConnector):
    """Excel file connector."""
    
    def connect(self) -> bool:
        import pandas as pd
        
        try:
            filepath = self.config.filepath
            if not filepath:
                return False
            
            # Load all sheets
            xlsx = pd.ExcelFile(filepath)
            for sheet_name in xlsx.sheet_names:
                df = pd.read_excel(xlsx, sheet_name=sheet_name)
                # Clean sheet name for table name
                table_name = sheet_name.replace(" ", "_").lower()
                self.dataframes[table_name] = df
            
            # Set up DuckDB
            try:
                import duckdb
                self.duckdb_conn = duckdb.connect(":memory:")
                for name, df in self.dataframes.items():
                    self.duckdb_conn.register(name, df)
            except ImportError:
                pass
            
            return len(self.dataframes) > 0
            
        except Exception as e:
            print(f"Excel connection error: {e}")
            return False


class ParquetConnector(CSVConnector):
    """Parquet file connector."""
    
    def connect(self) -> bool:
        import pandas as pd
        
        try:
            filepath = self.config.filepath
            if not filepath:
                return False
            
            path = Path(filepath)
            
            if path.is_file():
                table_name = path.stem
                self.dataframes[table_name] = pd.read_parquet(filepath)
            elif path.is_dir():
                for parquet_file in path.glob("*.parquet"):
                    table_name = parquet_file.stem
                    self.dataframes[table_name] = pd.read_parquet(parquet_file)
            
            # Set up DuckDB
            try:
                import duckdb
                self.duckdb_conn = duckdb.connect(":memory:")
                for name, df in self.dataframes.items():
                    self.duckdb_conn.register(name, df)
            except ImportError:
                pass
            
            return len(self.dataframes) > 0
            
        except Exception as e:
            print(f"Parquet connection error: {e}")
            return False


# =============================================================================
# Connector Factory
# =============================================================================

def create_connector(config: ConnectionConfig) -> BaseConnector:
    """Create a connector based on configuration type."""
    connectors = {
        "sqlite": SQLiteConnector,
        "postgresql": PostgreSQLConnector,
        "postgres": PostgreSQLConnector,
        "csv": CSVConnector,
        "excel": ExcelConnector,
        "xlsx": ExcelConnector,
        "parquet": ParquetConnector,
    }
    
    connector_class = connectors.get(config.type.lower())
    if not connector_class:
        raise ValueError(f"Unsupported connector type: {config.type}")
    
    return connector_class(config)


class DataSource:
    """
    Unified data source interface.
    
    Usage:
        source = DataSource.from_sqlite("path/to/db.sqlite")
        source = DataSource.from_csv("path/to/data.csv")
        source = DataSource.from_postgresql(host="localhost", database="mydb")
        
        source.connect()
        tables = source.get_tables()
        result = source.query("SELECT * FROM users LIMIT 10")
    """
    
    def __init__(self, config: ConnectionConfig):
        self.config = config
        self.connector = create_connector(config)
        self._connected = False
    
    @classmethod
    def from_sqlite(cls, database: str) -> "DataSource":
        return cls(ConnectionConfig(type="sqlite", database=database))
    
    @classmethod
    def from_csv(cls, filepath: str) -> "DataSource":
        return cls(ConnectionConfig(type="csv", filepath=filepath))
    
    @classmethod
    def from_excel(cls, filepath: str) -> "DataSource":
        return cls(ConnectionConfig(type="excel", filepath=filepath))
    
    @classmethod
    def from_parquet(cls, filepath: str) -> "DataSource":
        return cls(ConnectionConfig(type="parquet", filepath=filepath))
    
    @classmethod
    def from_postgresql(
        cls,
        host: str = "localhost",
        port: int = 5432,
        database: str = "",
        username: str = "",
        password: str = "",
    ) -> "DataSource":
        return cls(ConnectionConfig(
            type="postgresql",
            host=host,
            port=port,
            database=database,
            username=username,
            password=password,
        ))
    
    def connect(self) -> bool:
        self._connected = self.connector.connect()
        return self._connected
    
    def disconnect(self) -> None:
        self.connector.disconnect()
        self._connected = False
    
    @property
    def is_connected(self) -> bool:
        return self._connected
    
    def query(self, sql: str) -> QueryResult:
        return self.connector.execute(sql)
    
    def get_tables(self) -> list[str]:
        return self.connector.get_tables()
    
    def get_schema(self, table: str) -> TableSchema:
        return self.connector.get_schema(table)
    
    def get_all_schemas(self) -> list[TableSchema]:
        return [self.get_schema(t) for t in self.get_tables()]
    
    def describe(self) -> str:
        """Get a description of the data source."""
        tables = self.get_tables()
        desc = f"Data Source: {self.config.type}\n"
        desc += f"Tables: {len(tables)}\n\n"
        
        for table in tables:
            schema = self.get_schema(table)
            desc += f"📊 {table} ({schema.row_count} rows)\n"
            for col in schema.columns:
                desc += f"   • {col['name']}: {col['type']}\n"
            desc += "\n"
        
        return desc

from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DatabaseConfig:
    database_name:str
    table_name:str


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir:Path
    raw_data_path:Path
    database_info: DatabaseConfig
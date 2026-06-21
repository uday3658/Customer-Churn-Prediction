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


@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir:Path
    input_data_path:Path
    transformed_data_path:Path


@dataclass(frozen=True)
class ModelTrainingConfig:
    root_dir:Path
    training_data_path:Path
    base_model_path:Path
    tuned_model_path:Path     


@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    input_data_path: Path
    base_model_path: Path
    tuned_model_path: Path
    scores_file_path: Path
    base_cm_path: Path
    tuned_cm_path: Path
    roc_curve_path:Path       
from src.CustomerChurnPrediction.constants import *
from src.CustomerChurnPrediction.utils.common import read_yaml, create_directories
from src.CustomerChurnPrediction.entity.config_entity import DatabaseConfig,DataIngestionConfig,DataTransformationConfig

class ConfigurationManager:
    def __init__(self,config_filepath = CONFIG_FILE_PATH):
        self.config = read_yaml(config_filepath)

        create_directories([self.config.artifacts_root])
    
    def get_data_ingetion_config(self):
        config = self.config.data_ingestion
        db = config.database

        create_directories([config.root_dir])

        database_config = DatabaseConfig(
            database_name=db.database_name,
            table_name=db.table_name
        )

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            raw_data_path=config.raw_data_path,
            database_info=database_config
        )

        return data_ingestion_config
    def get_data_transformation_config(self):

        config = self.config.data_transformation

        create_directories([config.root_dir])

        data_transformation_config = DataTransformationConfig(
            root_dir=config.root_dir,
            input_data_path=config.input_data_path,
            transformed_data_path=config.transformed_data_path
        )

        return data_transformation_config
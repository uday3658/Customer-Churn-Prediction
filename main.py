import sys
from CustomerChurnPrediction.config.configuration import ConfigurationManager
from CustomerChurnPrediction.utils.logger import logger
from CustomerChurnPrediction.utils.exception import CustomException
from CustomerChurnPrediction.components.data_ingestion import DataIngestion
from CustomerChurnPrediction.pipelines.stage_01_data_ingestion import DataIngestionTrainingPipeline


STAGE_NAME = "Data Ingestion"

try:
    logger.info(f">>>>>>>>> stage {STAGE_NAME} started <<<<<<<<<<")
    obj = DataIngestionTrainingPipeline()
    obj.main()
    logger.info(f">>>>>>>>> stage {STAGE_NAME} Completed <<<<<<<<<<\n\nX============X")

except Exception as e:
    raise CustomException(e, sys)
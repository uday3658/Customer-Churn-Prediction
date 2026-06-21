import os
import sys
from CustomerChurnPrediction.config.configuration import ConfigurationManager
from CustomerChurnPrediction.utils.logger import logger
from CustomerChurnPrediction.utils.exception import CustomException
from CustomerChurnPrediction.components.data_ingestion import DataIngestion


STAGE_NAME = "Data Inegstion"

class DataIngestionTrainingPipeline:
    def __init__(self):
        pass
    
    def main(self):
        config = ConfigurationManager()
        data_ingetion_config = config.get_data_ingetion_config()
        data_ingetion = DataIngestion(data_ingetion_config)
        data_ingetion.save_raw_data()

if __name__ == '__main__':
    try:
        logger.info(f">>>>>>>>>> stage {STAGE_NAME} started <<<<<<<<<<")
        obj = DataIngestionTrainingPipeline()
        obj.main()
        logger.info(f">>>>>>>>>> stage {STAGE_NAME} Completed <<<<<<<<<<\n\nX===========X")
    except Exception as e:
        raise CustomException(e,sys)
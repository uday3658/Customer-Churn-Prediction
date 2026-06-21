import os
import sys

from CustomerChurnPrediction.config.configuration import ConfigurationManager
from CustomerChurnPrediction.utils.logger import logger
from CustomerChurnPrediction.utils.exception import CustomException
from CustomerChurnPrediction.components.data_transformation import DataTransformation


STAGE_NAME = "Data Transformation"


class DataTransformationPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_transformtion_config = config.get_data_transformation_config()
        data_transformation = DataTransformation(data_transformtion_config)
        data_transformation.get_transform_data()


if __name__ == '__main__':
    try:
        logger.info(f">>>>>>>>>> stage {STAGE_NAME} started <<<<<<<<<<<<")
        obj = DataTransformationPipeline()
        obj.main()
        logger.info(f">>>>>>>>>> stage {STAGE_NAME} Completed <<<<<<<<<<<<\n\nX==========X")
    except Exception as e:
        raise CustomException(e, sys)
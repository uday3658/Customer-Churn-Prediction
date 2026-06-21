import os
import sys

from CustomerChurnPrediction.config.configuration import ConfigurationManager
from CustomerChurnPrediction.utils.logger import logger
from CustomerChurnPrediction.utils.exception import CustomException
from CustomerChurnPrediction.components.model_training import ModelTraining


STAGE_NAME = "Model Training"


class ModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        model_training_config = config.get_model_training()
        model_training = ModelTraining(model_training_config)
        model_training.run_training()


if __name__ == '__main__':
    try:
        logger.info(f">>>>>>>>>> stage {STAGE_NAME} started <<<<<<<<<<<<")
        obj = ModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>>>>>> stage {STAGE_NAME} Completed <<<<<<<<<<<<\n\nX==========X")
    except Exception as e:
        raise CustomException(e, sys)
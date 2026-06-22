
import os
import sys
import yaml
import pandas as pd
from box import ConfigBox

import json

from CustomerChurnPrediction.utils.logger import logger
from CustomerChurnPrediction.utils.exception import CustomException
from ensure import ensure_annotations
from pathlib import Path


@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """reads yaml file and returns

    Args:
        path_to_yaml (str): path like input

    Raises:
        ValueError: if yaml file is empty
        e: empty file

    Returns:
        ConfigBox: ConfigBox type
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except Exception as e:
        raise CustomException(e,sys)

    
@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    """create list of directories

    Args:
        path_to_directories (list): list of path of directories
        ignore_log (bool, optional): ignore if multiple dirs is to be created. Defaults to False.
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"created directory at: {path}")


def save_data(df: pd.DataFrame, file_path: str) -> None:
    """
    Save a dataframe to CSV, creating the destination directory if needed.
 
    Args:
        df (pd.DataFrame): Dataframe to save.
        file_path (str): Full path (including filename) to save the CSV to.
    """
    create_directories([os.path.dirname(file_path)])
 
    df.to_csv(file_path, index=False)
 
    logger.info(f"Data saved to {file_path}: {df.shape[0]} rows, {df.shape[1]} columns")
    
@ensure_annotations
def save_json(path: Path, data: dict):
    """save json data

    Args:
        path (Path): path to json file
        data (dict): data to be saved in json file
    """
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

    logger.info(f"json file saved at: {path}")


@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    """load json files data

    Args:
        path (Path): path to json file

    Returns:
        ConfigBox: data as class attributes instead of dict
    """
    with open(path) as f:
        content = json.load(f)

    logger.info(f"json file loaded succesfully from: {path}")
    return ConfigBox(content)
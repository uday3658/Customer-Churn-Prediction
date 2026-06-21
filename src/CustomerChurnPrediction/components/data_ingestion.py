import os
import sys
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine
from src.CustomerChurnPrediction.utils.logger import logger
from src.CustomerChurnPrediction.utils.exception import CustomException
from src.CustomerChurnPrediction.entity.config_entity import DataIngestionConfig
from src.CustomerChurnPrediction.utils.common import create_directories,save_data

load_dotenv()

class DataIngestion:
    def __init__(self,config:DataIngestionConfig):
        self.config = config
    
        
    
    def read_sql_data(self,database_name: str, table_name: str) -> pd.DataFrame:
        """
        Connect to a MySQL database and fetch all rows from the given table.

        Database credentials (user, password, host, port) are read from
        environment variables (DB_USER, DB_PASSWORD, DB_HOST, DB_PORT).
        The password is URL-encoded to safely handle special characters
        when building the SQLAlchemy connection string.

        Args:
            database_name (str): Name of the database to connect to.
            table_name (str): Name of the table to query (all rows/columns
                are fetched via SELECT *).

        Returns:
            pd.DataFrame: The fetched table data as a DataFrame.

        Raises:
            CustomException: If the connection or query fails for any reason.
        """
        
        try:
            logger.info(f"Connecting to database: {database_name}")

            # read credentials from .env
            user     = os.getenv("DB_USER")
            password = os.getenv("DB_PASSWORD")
            host     = os.getenv("DB_HOST")
            port     = os.getenv("DB_PORT")

            # create connection
            connection_url = f"mysql+pymysql://{user}:{password}@{host}:{port}/{database_name}"
            engine         = create_engine(connection_url)

            # fetch data
            df = pd.read_sql(f"SELECT * FROM {table_name}", engine)

            logger.info(f"Data fetched successfully: {df.shape[0]} rows, {df.shape[1]} columns")

            return df
        except Exception as e:
            raise CustomException(e,sys)

    def save_raw_data(self):

        """
        Read data from the configured database table and save it
        as a raw CSV file.

        Steps:
        1. Read database and table information from config.
        2. Fetch data from MySQL.
        3. Create the target directory if it does not exist.
        4. Save the data as a CSV file.

        Raises:
            CustomException:
                Raised when data retrieval or file saving fails.
        """

        database_name = self.config.database_info.database_name
        table_name = self.config.database_info.table_name
        df = self.read_sql_data(database_name,table_name)
        save_data(df,self.config.raw_data_path)
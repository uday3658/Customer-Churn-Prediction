import pandas as pd
import pandas as pd
from CustomerChurnPrediction.utils.logger import logger
from CustomerChurnPrediction.entity.config_entity import DataTransformationConfig
from CustomerChurnPrediction.utils.common import create_directories,save_data

class DataTransformation:
    """Handles cleaning, encoding, and feature transformation for the Telco Churn dataset."""

    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def get_input_data(self) -> pd.DataFrame:
        """Load raw CSV data from the configured path."""
        
        df = pd.read_csv(self.config.input_data_path)
        logger.info(f"Loaded raw data with shape {df.shape}")
        
        return df

    def drop_unnecessary_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop columns that have no predictive value (e.g., customerID)."""
       
        df = df.drop(columns=['customerID'])
        logger.info("Dropped column: customerID")
       
        return df


    def binary_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode binary categorical columns (Yes/No, Male/Female) as 1/0."""

        binary_cols = [
            'gender',
            'Partner',
            'Dependents',
            'PhoneService',
            'PaperlessBilling',
            'Churn'
        ]

        df[binary_cols] = df[binary_cols].replace({
            'Yes': 1,
            'No': 0,
            'Male': 1,
            'Female': 0
        })

        logger.info(f"Binary encoded columns: {binary_cols}")
        return df

    def one_hot_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        """One-hot encode multi-category columns, dropping the first category to avoid redundancy."""

        multi_cat_cols = [
            'MultipleLines',
            'InternetService',
            'OnlineSecurity',
            'OnlineBackup',
            'DeviceProtection',
            'TechSupport',
            'StreamingTV',
            'StreamingMovies',
            'Contract',
            'PaymentMethod'
        ]

        df = pd.get_dummies(df, columns=multi_cat_cols, drop_first=True)
        logger.info(f"One-hot encoded columns: {multi_cat_cols}; new shape {df.shape}")
        
        return df

    def collapse_redundant_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Collapse 'No internet/phone service' dummy columns into single flags to reduce multicollinearity (decided via VIF analysis in notebook)."""

        # Collapse all "No internet service" dummies into one column
        no_internet_cols = [col for col in df.columns if 'No internet service' in col]

        if no_internet_cols:
            df['No_internet_service'] = df[no_internet_cols].any(axis=1).astype(int)
            df = df.drop(columns=no_internet_cols)
            logger.info(f"Collapsed columns into No_internet_service: {no_internet_cols}")

        # Handle PhoneService redundancy similarly
        if 'MultipleLines_No phone service' in df.columns:
            df['No_phone_service'] = df['MultipleLines_No phone service'].astype(int)
            df = df.drop(columns=['MultipleLines_No phone service'])
            logger.info("Collapsed MultipleLines_No phone service into No_phone_service")

        return df

    def bool_to_int(self, df: pd.DataFrame) -> pd.DataFrame:
        
        """Convert any boolean columns (created during one-hot encoding) to int (0/1)."""
       
        bool_cols = df.select_dtypes(include='bool').columns

        if len(bool_cols) > 0:
            df[bool_cols] = df[bool_cols].astype(int)
            logger.info(f"Converted bool columns to int: {list(bool_cols)}")

        return df

    def get_transform_data(self) -> pd.DataFrame:
       
        """Run the full transformation pipeline and return the model-ready DataFrame."""

        df = self.get_input_data()

        df = self.drop_unnecessary_columns(df)

        df = self.binary_encoding(df)

        df = self.one_hot_encoding(df)

        df = self.collapse_redundant_columns(df)

        df = self.bool_to_int(df)

        logger.info(f"Transformation pipeline complete. Final shape: {df.shape}")
        save_data(df,self.config.transformed_data_path)
import joblib
import pandas as pd
from CustomerChurnPrediction.utils.logger import logger
from CustomerChurnPrediction.constants import *
from CustomerChurnPrediction.utils.common import read_yaml

class PredictionPipeline:
    """
    Applies the same transformation logic used during training
    and returns churn prediction using the tuned XGBoost model.
    """

    THRESHOLD = 0.25  # must match ModelTraining.THRESHOLD
    MODEL_PATH = "model/tuned_model.pkl"

    def __init__(self):
        config = read_yaml(CONFIG_FILE_PATH)
        model_path = config.model_training.tuned_model_path
        logger.info("Loading tuned model...")
        self.model = joblib.load(model_path)
        logger.info("Model loaded successfully.")


    def binary_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService',
                       'PaperlessBilling']          # ← no 'Churn' at inference
        df[binary_cols] = df[binary_cols].replace(
            {'Yes': 1, 'No': 0, 'Male': 1, 'Female': 0}
        )
        df[binary_cols] = df[binary_cols].astype(int)
        return df

    def one_hot_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        multi_cat_cols = [
            'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
            'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
            'Contract', 'PaymentMethod'
        ]
        return pd.get_dummies(df, columns=multi_cat_cols, drop_first=True)

    def bool_to_int(self, df: pd.DataFrame) -> pd.DataFrame:
        bool_cols = df.select_dtypes(include='bool').columns
        df[bool_cols] = df[bool_cols].astype(int)
        return df

    def collapse_redundant_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        no_internet_cols = [
            'OnlineSecurity_No internet service',
            'OnlineBackup_No internet service',
            'DeviceProtection_No internet service',
            'TechSupport_No internet service',
            'StreamingTV_No internet service',
            'StreamingMovies_No internet service',
        ]
        existing = [c for c in no_internet_cols if c in df.columns]
        if existing:
            df['No_internet_service'] = df[existing].any(axis=1).astype(int)
            df = df.drop(columns=existing)

        if 'MultipleLines_No phone service' in df.columns:
            df['No_phone_service'] = df['MultipleLines_No phone service'].astype(int)
            df = df.drop(columns=['MultipleLines_No phone service'])

        return df

    def align_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure inference data has exactly the same columns the model
        was trained on — fills any missing dummy columns with 0,
        drops any extra columns.
        """
        trained_cols = self.model.get_booster().feature_names
        for col in trained_cols:
            if col not in df.columns:
                df[col] = 0                        
        df = df[trained_cols]                     
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Full transformation pipeline — mirrors DataTransformation.get_transform_data()."""
        if 'customerID' in df.columns:
            df = df.drop(columns='customerID')

        df = self.binary_encoding(df)
        df = self.one_hot_encoding(df)
        df = self.collapse_redundant_columns(df)
        df = self.bool_to_int(df)
        df = self.align_columns(df)               

        return df

    # ------------------------------------------------------------------ #
    #  Prediction                                                          #
    # ------------------------------------------------------------------ #

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Args:
            df: Raw input dataframe (same format as raw_data.csv, minus Churn column)

        Returns:
            Original dataframe with two new columns added:
                - churn_probability  (float)
                - churn_prediction   (0 or 1, using threshold=0.25)
        """
        logger.info(f"Running prediction on {len(df)} records...")

        transformed = self.transform(df.copy())

        proba = self.model.predict_proba(transformed)[:, 1]
        prediction = (proba >= self.THRESHOLD).astype(int)

        df['churn_probability'] = proba.round(4)
        df['churn_prediction'] = prediction

        logger.info(f"Prediction complete. Churners found: {prediction.sum()} / {len(df)}")
        return df
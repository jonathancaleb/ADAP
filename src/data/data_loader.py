import pandas as pd
import numpy as np
from typing import Optional, Tuple

class DataLoader:
    def __init__(self):
        """Initialize the DataLoader class"""
        self.raw_data = None
        self.processed_data = None

    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load data from a CSV file
        
        Args:
            file_path (str): Path to the CSV file
            
        Returns:
            pd.DataFrame: Loaded data
        """
        try:
            self.raw_data = pd.read_csv(file_path)
            return self.raw_data
        except Exception as e:
            raise Exception(f"Error loading data: {str(e)}")

    def preprocess_data(self, data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Preprocess the data by handling missing values and converting data types
        
        Args:
            data (pd.DataFrame, optional): Data to preprocess. If None, uses self.raw_data
            
        Returns:
            pd.DataFrame: Preprocessed data
        """
        if data is None and self.raw_data is None:
            raise ValueError("No data available for preprocessing")
            
        df = data if data is not None else self.raw_data.copy()
        
        # Basic preprocessing steps
        # Handle missing values
        df = df.fillna(df.mean(numeric_only=True))
        
        # Convert date columns if they exist
        date_columns = df.select_dtypes(include=['object']).columns
        for col in date_columns:
            try:
                df[col] = pd.to_datetime(df[col])
            except:
                continue
                
        self.processed_data = df
        return df

    def split_features_target(self, 
                            target_column: str,
                            features: Optional[list] = None) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Split the data into features and target
        
        Args:
            target_column (str): Name of the target column
            features (list, optional): List of feature columns to use
            
        Returns:
            Tuple[pd.DataFrame, pd.Series]: Features and target data
        """
        if self.processed_data is None:
            raise ValueError("No processed data available")
            
        if features is None:
            features = [col for col in self.processed_data.columns 
                       if col != target_column]
            
        X = self.processed_data[features]
        y = self.processed_data[target_column]
        
        return X, y
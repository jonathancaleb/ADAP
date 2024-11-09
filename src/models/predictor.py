import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from typing import Tuple, Dict, Any

class CoffeeGrowthPredictor:
    def __init__(self):
        """Initialize the CoffeeGrowthPredictor class"""
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        
    def prepare_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for training by scaling features
        
        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Target variable
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Scaled features and target
        """
        self.feature_names = X.columns
        X_scaled = self.scaler.fit_transform(X)
        return X_scaled, y.values
        
    def train(self, X: pd.DataFrame, y: pd.Series, model_type: str = 'rf') -> Dict[str, Any]:
        """
        Train the model
        
        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Target variable
            model_type (str): Type of model to use ('rf' for Random Forest, 'lr' for Linear Regression)
            
        Returns:
            Dict[str, Any]: Dictionary containing model performance metrics
        """
        # Prepare data
        X_scaled, y = self.prepare_data(X, y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        # Select and train model
        if model_type == 'rf':
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        else:
            self.model = LinearRegression()
            
        self.model.fit(X_train, y_train)
        
        # Calculate metrics
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        
        if model_type == 'rf':
            feature_importance = dict(zip(self.feature_names, 
                                       self.model.feature_importances_))
        else:
            feature_importance = dict(zip(self.feature_names, 
                                       self.model.coef_))
            
        return {
            'train_score': train_score,
            'test_score': test_score,
            'feature_importance': feature_importance
        }
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the trained model
        
        Args:
            X (pd.DataFrame): Features to predict on
            
        Returns:
            np.ndarray: Predictions
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
            
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
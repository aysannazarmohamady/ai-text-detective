import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from typing import Tuple, Optional
from .config import Config


class DataProcessor:
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self._data = None
        self._is_processed = False
    
    def load_data(self, filepath: str) -> pd.DataFrame:
        self._data = pd.read_csv(filepath)
        return self._data
    
    def get_data(self) -> pd.DataFrame:
        if self._data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        return self._data
    
    def validate_data(self) -> bool:
        required_columns = ['text', 'generated']
        return all(col in self._data.columns for col in required_columns)
    
    def clean_data(self) -> pd.DataFrame:
        if not self.validate_data():
            raise ValueError("Invalid data format")
        
        self._data = self._data.dropna()
        self._data = self._data[self._data['text'].str.len() > 10]
        return self._data
    
    def split_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if self._data is None:
            raise ValueError("No data to split")
        
        train_data, test_data = train_test_split(
            self._data,
            test_size=self.config.MODEL.test_size,
            random_state=self.config.MODEL.random_state,
            stratify=self._data['generated']
        )
        return train_data, test_data
    
    def apply_smote(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        smote = SMOTE(
            k_neighbors=self.config.TRAINING.smote_k_neighbors,
            random_state=self.config.MODEL.random_state
        )
        return smote.fit_resample(X, y)
    
    def get_class_distribution(self) -> dict:
        if self._data is None:
            return {}
        
        counts = self._data['generated'].value_counts()
        total = len(self._data)
        return {
            'human_count': counts.get(0, 0),
            'ai_count': counts.get(1, 0),
            'human_percentage': (counts.get(0, 0) / total) * 100,
            'ai_percentage': (counts.get(1, 0) / total) * 100
        }
    
    def get_text_statistics(self) -> dict:
        if self._data is None:
            return {}
        
        human_texts = self._data[self._data['generated'] == 0]['text']
        ai_texts = self._data[self._data['generated'] == 1]['text']
        
        return {
            'human_avg_length': human_texts.str.len().mean(),
            'ai_avg_length': ai_texts.str.len().mean(),
            'human_median_length': human_texts.str.len().median(),
            'ai_median_length': ai_texts.str.len().median(),
        }

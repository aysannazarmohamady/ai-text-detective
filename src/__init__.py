"""
AI Text Detective: Advanced machine learning system for detecting AI-generated text.

This package provides tools for:
- Data processing and feature extraction
- Training multiple ML models
- Making predictions on new text
- Explaining model decisions with SHAP
- Interactive web interface
"""

__version__ = "1.0.0"
__author__ = "Your Name"

from .data_processor import DataProcessor
from .feature_extractor import FeatureExtractor
from .model_trainer import ModelTrainer
from .predictor import TextPredictor, ModelComparator
from .config import Config

__all__ = [
    'DataProcessor',
    'FeatureExtractor', 
    'ModelTrainer',
    'TextPredictor',
    'ModelComparator',
    'Config'
]

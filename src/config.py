from dataclasses import dataclass
from typing import Dict, List


@dataclass
class ModelConfig:
    random_state: int = 42
    test_size: float = 0.2
    cv_folds: int = 5
    
    
@dataclass 
class FeatureConfig:
    max_features_tfidf: int = 5000
    ngram_range: tuple = (1, 2)
    min_df: int = 2
    max_df: float = 0.95
    bert_model_name: str = "bert-base-uncased"
    max_sequence_length: int = 512


@dataclass
class TrainingConfig:
    models_to_train: List[str] = None
    use_smote: bool = True
    smote_k_neighbors: int = 3
    
    def __post_init__(self):
        if self.models_to_train is None:
            self.models_to_train = ["logistic", "xgboost", "ensemble"]


@dataclass
class XGBoostParams:
    n_estimators: int = 100
    max_depth: int = 6
    learning_rate: float = 0.1
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.1
    reg_lambda: float = 1.0


@dataclass
class LogisticParams:
    C: float = 1.0
    max_iter: int = 1000
    solver: str = "liblinear"
    class_weight: str = "balanced"


class Config:
    MODEL = ModelConfig()
    FEATURES = FeatureConfig()
    TRAINING = TrainingConfig()
    XGBOOST = XGBoostParams()
    LOGISTIC = LogisticParams()
    
    DATA_PATH = "data/"
    MODELS_PATH = "models/"
    RESULTS_PATH = "results/"

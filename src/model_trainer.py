import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import xgboost as xgb
import joblib
import os
from typing import Dict, Tuple, List, Optional
from .config import Config
from .data_processor import DataProcessor
from .feature_extractor import FeatureExtractor


class ModelTrainer:
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.models = {}
        self.feature_extractor = None
        self.data_processor = None
        self.feature_names = []
        
    def setup_components(self):
        self.data_processor = DataProcessor(self.config)
        self.feature_extractor = FeatureExtractor(self.config)
        os.makedirs(self.config.MODELS_PATH, exist_ok=True)
    
    def create_logistic_model(self) -> LogisticRegression:
        return LogisticRegression(
            C=self.config.LOGISTIC.C,
            max_iter=self.config.LOGISTIC.max_iter,
            solver=self.config.LOGISTIC.solver,
            class_weight=self.config.LOGISTIC.class_weight,
            random_state=self.config.MODEL.random_state
        )
    
    def create_xgboost_model(self) -> xgb.XGBClassifier:
        return xgb.XGBClassifier(
            n_estimators=self.config.XGBOOST.n_estimators,
            max_depth=self.config.XGBOOST.max_depth,
            learning_rate=self.config.XGBOOST.learning_rate,
            subsample=self.config.XGBOOST.subsample,
            colsample_bytree=self.config.XGBOOST.colsample_bytree,
            reg_alpha=self.config.XGBOOST.reg_alpha,
            reg_lambda=self.config.XGBOOST.reg_lambda,
            random_state=self.config.MODEL.random_state,
            eval_metric='logloss'
        )
    
    def create_ensemble_model(self) -> VotingClassifier:
        logistic_model = self.create_logistic_model()
        xgboost_model = self.create_xgboost_model()
        
        return VotingClassifier(
            estimators=[
                ('logistic', logistic_model),
                ('xgboost', xgboost_model)
            ],
            voting='soft'
        )
    
    def prepare_data(self, data_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        self.data_processor.load_data(data_path)
        self.data_processor.clean_data()
        
        train_data, test_data = self.data_processor.split_data()
        
        X_train, self.feature_names = self.feature_extractor.extract_features(
            train_data['text'].tolist()
        )
        y_train = train_data['generated'].values
        
        X_test, _ = self.feature_extractor.extract_features(
            test_data['text'].tolist()
        )
        y_test = test_data['generated'].values
        
        if self.config.TRAINING.use_smote:
            X_train, y_train = self.data_processor.apply_smote(X_train, y_train)
        
        return X_train, X_test, y_train, y_test
    
    def train_single_model(self, model_name: str, X_train: np.ndarray, y_train: np.ndarray) -> object:
        if model_name == "logistic":
            model = self.create_logistic_model()
        elif model_name == "xgboost":
            model = self.create_xgboost_model()
        elif model_name == "ensemble":
            model = self.create_ensemble_model()
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        model.fit(X_train, y_train)
        self.models[model_name] = model
        return model
    
    def evaluate_model(self, model: object, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        return {
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'roc_auc_score': roc_auc_score(y_test, y_pred_proba),
            'accuracy': (y_pred == y_test).mean()
        }
    
    def cross_validate_model(self, model: object, X: np.ndarray, y: np.ndarray) -> Dict:
        cv = StratifiedKFold(n_splits=self.config.MODEL.cv_folds, shuffle=True, 
                            random_state=self.config.MODEL.random_state)
        
        scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        
        return {
            'mean_accuracy': scores.mean(),
            'std_accuracy': scores.std(),
            'scores': scores.tolist()
        }
    
    def save_model(self, model: object, model_name: str):
        model_path = os.path.join(self.config.MODELS_PATH, f"{model_name}_model.pkl")
        joblib.dump(model, model_path)
        
        extractor_path = os.path.join(self.config.MODELS_PATH, "feature_extractor.pkl")
        joblib.dump(self.feature_extractor, extractor_path)
    
    def train_all_models(self, data_path: str) -> Dict:
        self.setup_components()
        X_train, X_test, y_train, y_test = self.prepare_data(data_path)
        
        results = {}
        
        for model_name in self.config.TRAINING.models_to_train:
            print(f"Training {model_name} model...")
            
            model = self.train_single_model(model_name, X_train, y_train)
            
            test_results = self.evaluate_model(model, X_test, y_test)
            cv_results = self.cross_validate_model(model, X_train, y_train)
            
            results[model_name] = {
                'test_results': test_results,
                'cv_results': cv_results
            }
            
            self.save_model(model, model_name)
            print(f"{model_name} model trained and saved successfully!")
        
        return results


if __name__ == "__main__":
    trainer = ModelTrainer()
    results = trainer.train_all_models("data/ai_generated_text.csv")
    
    for model_name, metrics in results.items():
        print(f"\n{model_name.upper()} RESULTS:")
        print(f"Test Accuracy: {metrics['test_results']['accuracy']:.4f}")
        print(f"ROC-AUC Score: {metrics['test_results']['roc_auc_score']:.4f}")
        print(f"CV Mean Accuracy: {metrics['cv_results']['mean_accuracy']:.4f} ± {metrics['cv_results']['std_accuracy']:.4f}")

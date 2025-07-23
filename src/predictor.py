import numpy as np
import joblib
import os
from typing import Dict, Optional, List
from .config import Config

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


class TextPredictor:
    def __init__(self, model_name: str = "ensemble", config: Optional[Config] = None):
        self.config = config or Config()
        self.model_name = model_name
        self.model = None
        self.feature_extractor = None
        self.explainer = None
        self._is_loaded = False
    
    def load_model(self):
        model_path = os.path.join(self.config.MODELS_PATH, f"{self.model_name}_model.pkl")
        extractor_path = os.path.join(self.config.MODELS_PATH, "feature_extractor.pkl")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        if not os.path.exists(extractor_path):
            raise FileNotFoundError(f"Feature extractor not found: {extractor_path}")
        
        self.model = joblib.load(model_path)
        self.feature_extractor = joblib.load(extractor_path)
        self._is_loaded = True
    
    def setup_explainer(self, background_texts: List[str]):
        if not SHAP_AVAILABLE:
            print("Warning: SHAP not available. Explanations will be limited.")
            return
            
        if not self._is_loaded:
            self.load_model()
        
        background_features, _ = self.feature_extractor.extract_features(background_texts)
        self.explainer = shap.Explainer(self.model, background_features)
    
    def predict(self, text: str) -> Dict:
        if not self._is_loaded:
            self.load_model()
        
        features, feature_names = self.feature_extractor.extract_features([text])
        
        prediction = self.model.predict(features)[0]
        probability = self.model.predict_proba(features)[0]
        
        return {
            'text': text,
            'prediction': int(prediction),
            'prediction_label': 'AI Generated' if prediction == 1 else 'Human Written',
            'ai_probability': float(probability[1]),
            'human_probability': float(probability[0]),
            'confidence': float(max(probability))
        }
    
    def predict_batch(self, texts: List[str]) -> List[Dict]:
        if not self._is_loaded:
            self.load_model()
        
        features, _ = self.feature_extractor.extract_features(texts)
        
        predictions = self.model.predict(features)
        probabilities = self.model.predict_proba(features)
        
        results = []
        for i, text in enumerate(texts):
            results.append({
                'text': text,
                'prediction': int(predictions[i]),
                'prediction_label': 'AI Generated' if predictions[i] == 1 else 'Human Written',
                'ai_probability': float(probabilities[i][1]),
                'human_probability': float(probabilities[i][0]),
                'confidence': float(max(probabilities[i]))
            })
        
        return results
    
    def explain_prediction(self, text: str, background_texts: Optional[List[str]] = None) -> Dict:
        if not SHAP_AVAILABLE:
            prediction_result = self.predict(text)
            return {
                'prediction': prediction_result,
                'feature_importance': [],
                'explanation_summary': 'SHAP explanations not available. Install shap package for detailed analysis.'
            }
            
        if self.explainer is None and background_texts is not None:
            self.setup_explainer(background_texts)
        
        if self.explainer is None:
            return {'error': 'Explainer not set up. Provide background_texts parameter.'}
        
        features, feature_names = self.feature_extractor.extract_features([text])
        shap_values = self.explainer(features)
        
        prediction_result = self.predict(text)
        
        feature_importance = []
        for i, (name, value, shap_val) in enumerate(zip(feature_names, features[0], shap_values.values[0])):
            if abs(shap_val) > 0.001:
                feature_importance.append({
                    'feature_name': name,
                    'feature_value': float(value),
                    'shap_value': float(shap_val),
                    'importance': abs(float(shap_val))
                })
        
        feature_importance.sort(key=lambda x: x['importance'], reverse=True)
        
        return {
            'prediction': prediction_result,
            'feature_importance': feature_importance[:10],
            'explanation_summary': self._generate_explanation_summary(feature_importance[:5])
        }
    
    def _generate_explanation_summary(self, top_features: List[Dict]) -> str:
        if not top_features:
            return "Unable to generate explanation."
        
        prediction_type = "AI-generated" if top_features[0]['shap_value'] > 0 else "human-written"
        
        key_factors = []
        for feature in top_features[:3]:
            feature_name = feature['feature_name']
            if 'readability' in feature_name.lower():
                key_factors.append("text readability patterns")
            elif 'sentiment' in feature_name.lower():
                key_factors.append("sentiment characteristics")
            elif 'length' in feature_name.lower():
                key_factors.append("text length patterns")
            elif 'tfidf' in feature_name.lower():
                key_factors.append("vocabulary usage")
            else:
                key_factors.append("linguistic patterns")
        
        factors_text = ", ".join(key_factors[:2])
        return f"Text classified as {prediction_type} based on {factors_text}."


class ModelComparator:
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.models = {}
        self.predictors = {}
    
    def load_all_models(self):
        available_models = ['logistic', 'xgboost', 'ensemble']
        
        for model_name in available_models:
            try:
                predictor = TextPredictor(model_name, self.config)
                predictor.load_model()
                self.predictors[model_name] = predictor
            except FileNotFoundError:
                print(f"Model {model_name} not found, skipping...")
    
    def compare_predictions(self, text: str) -> Dict:
        if not self.predictors:
            self.load_all_models()
        
        results = {}
        for model_name, predictor in self.predictors.items():
            results[model_name] = predictor.predict(text)
        
        return results
    
    def get_consensus_prediction(self, text: str) -> Dict:
        predictions = self.compare_predictions(text)
        
        ai_probabilities = [pred['ai_probability'] for pred in predictions.values()]
        avg_ai_prob = np.mean(ai_probabilities)
        
        consensus_pred = 1 if avg_ai_prob > 0.5 else 0
        consensus_label = 'AI Generated' if consensus_pred == 1 else 'Human Written'
        
        return {
            'text': text,
            'consensus_prediction': consensus_pred,
            'consensus_label': consensus_label,
            'average_ai_probability': float(avg_ai_prob),
            'individual_predictions': predictions,
            'agreement_level': self._calculate_agreement(predictions)
        }
    
    def _calculate_agreement(self, predictions: Dict) -> str:
        pred_values = [pred['prediction'] for pred in predictions.values()]
        agreement_ratio = pred_values.count(pred_values[0]) / len(pred_values)
        
        if agreement_ratio == 1.0:
            return "Full Agreement"
        elif agreement_ratio >= 0.67:
            return "Majority Agreement"
        else:
            return "Low Agreement"

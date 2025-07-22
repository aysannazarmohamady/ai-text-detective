import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer, AutoModel
import torch
import textstat
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from typing import List, Tuple, Optional
from .config import Config

nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)


class LinguisticFeatureExtractor:
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()
    
    def extract_readability_features(self, text: str) -> dict:
        return {
            'flesch_kincaid_grade': textstat.flesch_kincaid_grade(text),
            'flesch_reading_ease': textstat.flesch_reading_ease(text),
            'automated_readability_index': textstat.automated_readability_index(text),
            'coleman_liau_index': textstat.coleman_liau_index(text)
        }
    
    def extract_structural_features(self, text: str) -> dict:
        sentences = nltk.sent_tokenize(text)
        words = nltk.word_tokenize(text)
        
        return {
            'char_count': len(text),
            'word_count': len(words),
            'sentence_count': len(sentences),
            'avg_word_length': np.mean([len(word) for word in words]),
            'avg_sentence_length': len(words) / len(sentences) if sentences else 0,
            'punctuation_ratio': sum(c in '.,!?;:' for c in text) / len(text)
        }
    
    def extract_sentiment_features(self, text: str) -> dict:
        scores = self.sia.polarity_scores(text)
        return {
            'sentiment_compound': scores['compound'],
            'sentiment_positive': scores['pos'],
            'sentiment_negative': scores['neg'],
            'sentiment_neutral': scores['neu']
        }
    
    def extract_all_features(self, text: str) -> dict:
        features = {}
        features.update(self.extract_readability_features(text))
        features.update(self.extract_structural_features(text))
        features.update(self.extract_sentiment_features(text))
        return features


class TfidfFeatureExtractor:
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.vectorizer = None
        self._is_fitted = False
    
    def fit(self, texts: List[str]) -> 'TfidfFeatureExtractor':
        self.vectorizer = TfidfVectorizer(
            max_features=self.config.FEATURES.max_features_tfidf,
            ngram_range=self.config.FEATURES.ngram_range,
            min_df=self.config.FEATURES.min_df,
            max_df=self.config.FEATURES.max_df,
            stop_words='english'
        )
        self.vectorizer.fit(texts)
        self._is_fitted = True
        return self
    
    def transform(self, texts: List[str]) -> np.ndarray:
        if not self._is_fitted:
            raise ValueError("TfidfFeatureExtractor must be fitted first")
        return self.vectorizer.transform(texts).toarray()
    
    def fit_transform(self, texts: List[str]) -> np.ndarray:
        return self.fit(texts).transform(texts)


class BertFeatureExtractor:
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.tokenizer = None
        self.model = None
        self._is_loaded = False
    
    def load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.FEATURES.bert_model_name)
        self.model = AutoModel.from_pretrained(self.config.FEATURES.bert_model_name)
        self.model.eval()
        self._is_loaded = True
    
    def extract_embeddings(self, texts: List[str], batch_size: int = 8) -> np.ndarray:
        if not self._is_loaded:
            self.load_model()
        
        embeddings = []
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                encoded = self.tokenizer(
                    batch_texts,
                    truncation=True,
                    padding=True,
                    max_length=self.config.FEATURES.max_sequence_length,
                    return_tensors='pt'
                )
                
                outputs = self.model(**encoded)
                batch_embeddings = outputs.last_hidden_state.mean(dim=1)
                embeddings.append(batch_embeddings.numpy())
        
        return np.vstack(embeddings)


class FeatureExtractor:
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.linguistic_extractor = LinguisticFeatureExtractor()
        self.tfidf_extractor = TfidfFeatureExtractor(config)
        self.bert_extractor = BertFeatureExtractor(config)
        
    def extract_features(self, texts: List[str], include_bert: bool = False) -> Tuple[np.ndarray, List[str]]:
        linguistic_features = []
        feature_names = []
        
        for text in texts:
            features = self.linguistic_extractor.extract_all_features(text)
            linguistic_features.append(list(features.values()))
            if not feature_names:
                feature_names.extend(features.keys())
        
        linguistic_array = np.array(linguistic_features)
        
        tfidf_features = self.tfidf_extractor.fit_transform(texts)
        tfidf_names = [f"tfidf_{i}" for i in range(tfidf_features.shape[1])]
        feature_names.extend(tfidf_names)
        
        combined_features = np.hstack([linguistic_array, tfidf_features])
        
        if include_bert:
            bert_features = self.bert_extractor.extract_embeddings(texts)
            bert_names = [f"bert_{i}" for i in range(bert_features.shape[1])]
            feature_names.extend(bert_names)
            combined_features = np.hstack([combined_features, bert_features])
        
        return combined_features, feature_names

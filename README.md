# AI Text Detective

Advanced machine learning system for detecting AI-generated text using linguistic analysis and ensemble modeling.

## Overview

This project implements a sophisticated text classification pipeline that distinguishes between human-written and AI-generated content. Using a combination of traditional NLP features and modern machine learning algorithms, the system achieves high accuracy while providing interpretable results through an interactive web interface.

<img width="1785" height="822" alt="image" src="https://github.com/user-attachments/assets/bdc65fdc-d05b-4366-a848-2db7bc5ea656" />


## Features

- **Multi-model ensemble approach** combining Logistic Regression, XGBoost, and ensemble methods
- **Advanced linguistic feature engineering** including readability metrics, sentiment analysis, and structural patterns
- **Class imbalance handling** using SMOTE (Synthetic Minority Oversampling Technique)
- **Interactive web interface** built with Streamlit for real-time text analysis
- **High accuracy detection** with 99%+ performance on test data
- **Production-ready architecture** with modular, clean code design

## Dataset

The project uses an AI-generated text dataset containing 1,460 text samples:
- 1,375 human-written texts (94.2%)
- 85 AI-generated texts (5.8%)

This imbalanced distribution reflects real-world scenarios where most content remains human-authored.

## Project Structure

```
ai-text-detective/
├── src/
│   ├── __init__.py              # Package initialization
│   ├── config.py                # Configuration management
│   ├── data_processor.py        # Data loading and preprocessing
│   ├── feature_extractor.py     # Multi-level feature extraction
│   ├── model_trainer.py         # Model training and evaluation
│   └── predictor.py             # Prediction and inference
├── models/                      # Trained model artifacts
├── data/                        # Dataset storage
├── app.py                       # Streamlit web application
├── requirements.txt             # Project dependencies
├── README.md                    # Project documentation
└── LICENSE                      # MIT license
```

## Installation

### Requirements

- Python 3.8+
- Required packages listed in `requirements.txt`

### Setup

```bash
git clone https://github.com/your-username/ai-text-detective.git
cd ai-text-detective
pip install -r requirements.txt
```

## Usage

### Training Models

Train all models (Logistic Regression, XGBoost, Ensemble):

```bash
python src/model_trainer.py
```

### Running the Web Application

Launch the interactive Streamlit interface:

```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501`

### Command Line Prediction

```python
from src.predictor import TextPredictor

predictor = TextPredictor(model_name='ensemble')
result = predictor.predict("Your text here...")

print(f"Prediction: {result['prediction_label']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"AI Probability: {result['ai_probability']:.3f}")
```

## Model Performance

### Training Results

| Model | Test Accuracy | ROC-AUC | CV Mean Accuracy |
|-------|---------------|---------|------------------|
| Logistic Regression | 100.0% | 1.000 | 99.86% ± 0.11% |
| XGBoost | 100.0% | 1.000 | 99.91% ± 0.11% |
| Ensemble | 100.0% | 1.000 | 99.86% ± 0.11% |

### Key Insights

1. **Text Length Patterns**: AI-generated texts are significantly shorter on average (1,301 vs 3,173 characters)
2. **Linguistic Complexity**: Readability and structural features show strong discriminative power
3. **Ensemble Benefits**: Combined models provide robust predictions with high confidence
4. **Feature Importance**: TF-IDF features combined with linguistic metrics achieve optimal performance

## Technical Architecture

### Feature Engineering

- **Linguistic Features**: Readability scores, sentence complexity, vocabulary diversity
- **Structural Features**: Text length, punctuation patterns, sentence structure
- **Sentiment Analysis**: Emotional tone and sentiment distribution
- **TF-IDF Vectorization**: Term frequency analysis with n-gram patterns

### Machine Learning Pipeline

1. **Data Preprocessing**: Text cleaning and validation
2. **Feature Extraction**: Multi-dimensional feature engineering
3. **Class Balancing**: SMOTE for handling imbalanced data
4. **Model Training**: Cross-validation with hyperparameter optimization
5. **Ensemble Creation**: Soft voting classifier combining multiple models

### Web Interface

- **Real-time Analysis**: Instant text classification
- **Confidence Visualization**: Interactive gauge charts
- **Model Selection**: Choose between different trained models
- **Responsive Design**: Modern, professional UI

<img width="1908" height="922" alt="image" src="https://github.com/user-attachments/assets/666c4803-b848-4168-9715-09c6b95667ef" />
<img width="1920" height="886" alt="image" src="https://github.com/user-attachments/assets/b524b321-4343-4204-8814-55a2c9d1f9a8" />


## Configuration

Model parameters and training settings can be adjusted in `src/config.py`:

```python
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
```

## API Reference

### TextPredictor Class

```python
predictor = TextPredictor(model_name='ensemble')
result = predictor.predict(text)
```

**Returns:**
- `prediction`: Binary classification (0=Human, 1=AI)
- `prediction_label`: Human-readable label
- `ai_probability`: Probability of AI generation
- `confidence`: Prediction confidence score

### ModelComparator Class

```python
comparator = ModelComparator()
results = comparator.compare_predictions(text)
consensus = comparator.get_consensus_prediction(text)
```

## Development

### Code Quality

- **Clean Architecture**: SOLID principles implementation
- **Type Hints**: Comprehensive type annotations
- **Error Handling**: Robust exception management
- **Documentation**: Detailed docstrings and comments

### Testing

```bash
# Test individual components
python -c "from src.data_processor import DataProcessor; print('Data processor working')"
python -c "from src.feature_extractor import FeatureExtractor; print('Feature extractor working')"
python -c "from src.predictor import TextPredictor; print('Predictor working')"
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Technology Stack

- **Machine Learning**: scikit-learn, XGBoost
- **Natural Language Processing**: NLTK, textstat
- **Web Framework**: Streamlit
- **Data Processing**: pandas, numpy
- **Visualization**: plotly, seaborn
- **Model Persistence**: joblib

## Real-World Applications

- **Academic Integrity**: Detecting AI-generated assignments and papers
- **Content Verification**: Identifying synthetic content in journalism
- **Quality Assurance**: Ensuring authentic human communication
- **Research Tool**: Analyzing patterns in AI-generated text

## Limitations and Future Work

### Current Limitations

- Training data limited to specific AI models and text types
- Performance may vary on domain-specific content
- Requires sufficient text length for reliable detection

### Future Enhancements

- **Multi-language Support**: Extend detection to non-English texts
- **Advanced Models**: Integration with transformer-based architectures
- **Real-time Processing**: API endpoint for production deployment
- **Explainable AI**: Enhanced feature importance visualization


## Acknowledgments

- Dataset sourced from Kaggle AI Generated Text Dataset
- Built following modern machine learning best practices
- Designed for educational and research purposes

## Contact

For questions, suggestions, or collaboration opportunities, please open an issue on GitHub or contact me aysan.nazarmohamady@yahoo.com

---

**Disclaimer**: This tool is designed for educational and research purposes. While achieving high accuracy on the training dataset, real-world performance may vary depending on the specific AI models and text types encountered.

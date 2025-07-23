import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from src.predictor import TextPredictor, ModelComparator
from src.config import Config
import os


class StreamlitApp:
    def __init__(self):
        self.config = Config()
        self.predictor = None
        self.comparator = ModelComparator(self.config)
        
    def setup_page(self):
        st.set_page_config(
            page_title="AI Text Detective",
            page_icon="🤖",
            layout="wide"
        )
        
        st.title("🤖 AI Text Detective")
        st.markdown("**Advanced machine learning system for detecting AI-generated text**")
        st.markdown("---")
    
    def create_sidebar(self):
        with st.sidebar:
            st.header("Settings")
            
            model_options = ['ensemble', 'logistic', 'xgboost']
            selected_model = st.selectbox(
                "Select Model",
                model_options,
                index=0
            )
            
            show_explanation = st.checkbox("Show Detailed Explanation", value=True)
            compare_models = st.checkbox("Compare All Models", value=False)
            
            st.markdown("---")
            st.header("About")
            st.markdown("""
            This system analyzes text using:
            - Linguistic features
            - TF-IDF vectorization  
            - Advanced ML algorithms
            - SHAP explanations
            """)
            
            return selected_model, show_explanation, compare_models
    
    def load_predictor(self, model_name: str):
        try:
            if self.predictor is None or self.predictor.model_name != model_name:
                self.predictor = TextPredictor(model_name, self.config)
                self.predictor.load_model()
            return True
        except FileNotFoundError as e:
            st.error(f"Model not found: {e}")
            return False
    
    def create_prediction_gauge(self, probability: float):
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = probability * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "AI Probability (%)"},
            delta = {'reference': 50},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkred" if probability > 0.5 else "darkgreen"},
                'steps': [
                    {'range': [0, 25], 'color': "lightgreen"},
                    {'range': [25, 50], 'color': "yellow"},
                    {'range': [50, 75], 'color': "orange"},
                    {'range': [75, 100], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 50
                }
            }
        ))
        
        fig.update_layout(height=300)
        return fig
    
    def display_prediction_results(self, result: dict):
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.metric(
                "Prediction",
                result['prediction_label'],
                f"{result['confidence']:.2%} confidence"
            )
            
            if result['prediction'] == 1:
                st.error("🤖 AI Generated Text Detected")
            else:
                st.success("👤 Human Written Text Detected")
        
        with col2:
            gauge_fig = self.create_prediction_gauge(result['ai_probability'])
            st.plotly_chart(gauge_fig, use_container_width=True)
    
    def display_feature_importance(self, explanation: dict):
        if 'feature_importance' not in explanation:
            return
        
        st.subheader("🔍 Feature Analysis")
        
        features_df = pd.DataFrame(explanation['feature_importance'])
        
        if not features_df.empty:
            fig = px.bar(
                features_df.head(8),
                x='importance',
                y='feature_name',
                orientation='h',
                title="Most Important Features",
                color='shap_value',
                color_continuous_scale='RdBu'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            st.info(f"**Explanation:** {explanation['explanation_summary']}")
    
    def display_model_comparison(self, comparison: dict):
        st.subheader("📊 Model Comparison")
        
        models_data = []
        for model_name, result in comparison['individual_predictions'].items():
            models_data.append({
                'Model': model_name.title(),
                'Prediction': result['prediction_label'],
                'AI Probability': result['ai_probability'],
                'Confidence': result['confidence']
            })
        
        models_df = pd.DataFrame(models_data)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.dataframe(models_df, use_container_width=True)
        
        with col2:
            fig = px.bar(
                models_df,
                x='Model',
                y='AI Probability',
                title="AI Probability by Model",
                color='AI Probability',
                color_continuous_scale='RdYlBu_r'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.metric("Agreement Level", comparison['agreement_level'])
    
    def main_interface(self):
        selected_model, show_explanation, compare_models = self.create_sidebar()
        
        st.subheader("Enter Text for Analysis")
        
        text_input = st.text_area(
            "Paste your text here:",
            height=200,
            placeholder="Enter the text you want to analyze..."
        )
        
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            analyze_button = st.button("🔍 Analyze Text", type="primary")
        
        with col2:
            clear_button = st.button("🗑️ Clear Text")
        
        if clear_button:
            st.rerun()
        
        if analyze_button and text_input.strip():
            if len(text_input.strip()) < 50:
                st.warning("Please enter at least 50 characters for accurate analysis.")
                return
            
            with st.spinner("Analyzing text..."):
                if compare_models:
                    comparison = self.comparator.get_consensus_prediction(text_input)
                    
                    st.subheader("🎯 Consensus Prediction")
                    self.display_prediction_results({
                        'prediction': comparison['consensus_prediction'],
                        'prediction_label': comparison['consensus_label'],
                        'ai_probability': comparison['average_ai_probability'],
                        'confidence': max(comparison['average_ai_probability'], 
                                        1 - comparison['average_ai_probability'])
                    })
                    
                    self.display_model_comparison(comparison)
                    
                else:
                    if not self.load_predictor(selected_model):
                        return
                    
                    result = self.predictor.predict(text_input)
                    self.display_prediction_results(result)
                    
                    if show_explanation:
                        with st.spinner("Generating explanation..."):
                            explanation = self.predictor.explain_prediction(
                                text_input,
                                background_texts=[
                                    "This is a sample human text for background.",
                                    "Machine learning is a fascinating field of study."
                                ]
                            )
                            self.display_feature_importance(explanation)
        
        elif analyze_button:
            st.warning("Please enter some text to analyze.")
    
    def display_demo_examples(self):
        st.subheader("📝 Try These Examples")
        
        examples = {
            "Human-like Example": """
                I've always been fascinated by the way technology evolves. Like, remember when 
                we thought flip phones were the coolest thing ever? Now I'm typing this on 
                a device that's basically a computer in my pocket. It's wild how fast things change.
            """,
            "AI-like Example": """
                Artificial intelligence represents a transformative technology that enables 
                computers to perform tasks typically requiring human intelligence. Through 
                machine learning algorithms, these systems can analyze data, identify patterns, 
                and make predictions with remarkable accuracy.
            """
        }
        
        col1, col2 = st.columns(2)
        
        for i, (title, example) in enumerate(examples.items()):
            with col1 if i % 2 == 0 else col2:
                st.text_area(title, example.strip(), height=120, key=f"example_{i}")
    
    def run(self):
        self.setup_page()
        
        tab1, tab2 = st.tabs(["🔍 Text Analysis", "📝 Examples"])
        
        with tab1:
            self.main_interface()
        
        with tab2:
            self.display_demo_examples()
        
        st.markdown("---")
        st.markdown(
            "**Built with:** Scikit-learn, XGBoost, Transformers, SHAP, Streamlit | "
            "**Source:** [GitHub Repository](https://github.com/your-username/ai-text-detective)"
        )


if __name__ == "__main__":
    app = StreamlitApp()
    app.run()

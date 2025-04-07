"""
Medical Waste Classifier Web App
Integrated with previous classification logic
"""
# Add these imports at the VERY TOP of your file
import matplotlib
matplotlib.use('Agg')  # Set backend before other imports
import matplotlib.pyplot as plt

# --------------------------
# IMPORTS & CONFIGURATION
# --------------------------
from flask import Flask, render_template, request
import pandas as pd
import re
import numpy as np
import pickle
import ssl
import nltk
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from pathlib import Path
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# --------------------------
# CLASSIFIER CLASS (FROM PREVIOUS CODE)
# --------------------------
class WasteClassifier:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.model = RandomForestClassifier()
        self.keyword_patterns = {}
        self.hazard_mapping = {}

    def preprocess_text(self, text: str) -> str:
        """Clean and normalize input text (from previous code)"""
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        tokens = [
            self.lemmatizer.lemmatize(token)
            for token in text.split()
            if token not in self.stop_words
        ]
        return ' '.join(tokens)

    def create_features(self, text: str) -> np.ndarray:
        """Generate features (from previous code)"""
        processed_text = self.preprocess_text(text)
        tfidf_features = self.vectorizer.transform([processed_text])
        domain_features = np.array([
            int(bool(re.search(pattern, text.lower())))
            for pattern in self.keyword_patterns.values()
        ]).reshape(1, -1)
        return np.hstack((tfidf_features.toarray(), domain_features))

    def get_hazard_level(self, category: str) -> str:
        """Get hazard level mapping"""
        return self.hazard_mapping.get(category, 'Low')

# --------------------------
# MODEL LOADING
# --------------------------
CONFIG = {
    'model_path': Path('waste_classifier_rf.pkl'),
    'hazard_colors': {
        'High': '#ff4444',
        'Medium': '#ffd700',
        'Low': '#77dd77'
    }
}

# Load trained model components
with open(CONFIG['model_path'], 'rb') as f:
    model_package = pickle.load(f)

# Reconstruct classifier from saved components
classifier = WasteClassifier()
classifier.model = model_package['model']
classifier.vectorizer = model_package['vectorizer']
classifier.preprocess_text = model_package['preprocessor']
classifier.keyword_patterns = model_package['keyword_patterns']
classifier.hazard_mapping = model_package['hazard_mapping']

# --------------------------
# WEB VISUALIZATION FUNCTIONS
# --------------------------
def plot_confidence(probs, classes):
    """Generate confidence distribution plot"""
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = [CONFIG['hazard_colors'][classifier.get_hazard_level(c)] for c in classes]
    bars = ax.barh(classes, probs, color=colors)
    ax.set_xlabel('Confidence')
    ax.set_title('Classification Confidence')
    ax.bar_label(bars, fmt='%.2f%%')
    plt.tight_layout()
    
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    return base64.b64encode(buf.getbuffer()).decode('ascii')

def plot_hazard_indicator(hazard_level):
    """Generate hazard level indicator"""
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.add_patch(plt.Circle((0.5, 0.5), 0.4, color=CONFIG['hazard_colors'][hazard_level]))
    ax.text(0.5, 0.5, hazard_level[0], ha='center', va='center', 
            fontsize=24, color='white', weight='bold')
    ax.axis('off')
    
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    return base64.b64encode(buf.getbuffer()).decode('ascii')

def plot_disposal_method(disposal_text: str) -> str:
    """Generate disposal method text visualization"""
    fig, ax = plt.subplots(figsize=(6, 2))
    ax.text(0.5, 0.5, disposal_text, 
            ha='center', va='center', 
            wrap=True,
            fontsize=12,
            color='#2c3e50')
    ax.axis('off')
    
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    return base64.b64encode(buf.getbuffer()).decode('ascii')

def get_disposal(category: str) -> str:
    """Get disposal instructions for predicted category"""
    disposal_methods = {
        'Chemical': "Yellow chemical container - Use PPE and ensure proper ventilation",
        'Sharps': "Sharps bin - Do not recap needles, handle with puncture-resistant gloves",
        'Pharmaceutical': "Blue pharma bin - Separate liquids from solids, check expiration dates",
        'General': "Black general waste - Standard clinical disposal procedures",
    }
    return disposal_methods.get(category, "Consult safety guidelines - Unknown waste category")

# --------------------------
# FLASK ROUTES
# --------------------------
@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    error = None
    
    if request.method == 'POST':
        text = request.form['text']
        try:
            features = classifier.create_features(text)
            probs = classifier.model.predict_proba(features)[0]
            prediction = classifier.model.predict(features)[0]
            hazard_level = classifier.get_hazard_level(prediction)
            disposal_text = get_disposal(prediction)  # Now using the defined function
            
            result = {
                'text': text,
                'prediction': prediction,
                'hazard_level': hazard_level,
                'confidence': f"{max(probs)*100:.2f}%",
                'confidence_plot': plot_confidence(probs*100, classifier.model.classes_),
                'hazard_plot': plot_hazard_indicator(hazard_level),
                'disposal_plot': plot_disposal_method(disposal_text)
            }
        except Exception as e:
            return render_template('error.html', error=str(e))
    
    return render_template('index.html', result=result)

# --------------------------
# INITIALIZATION
# --------------------------
def initialize_nltk():
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)

if __name__ == '__main__':
    initialize_nltk()
    app.run(debug=True)
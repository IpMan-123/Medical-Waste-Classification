"""
Medical Waste Classification System
Categories: Chemical, Sharps, Pharmaceutical, General
Hazard Levels: High, Medium, Low
"""

# --------------------------
# IMPORTS & CONFIGURATION
# --------------------------
import pandas as pd
import re
import numpy as np
import pickle
import ssl
import nltk
from typing import Dict, List, Tuple
from pathlib import Path
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Configuration Constants
CONFIG = {
    'data_path': Path('medical_waste_dataset.csv'),
    'model_save_path': Path('waste_classifier_rf.pkl'),
    'test_size': 0.2,
    'random_state': 42,
    'confidence_threshold': 0.75,
    'rf_params': {
        'n_estimators': 200,
        'max_depth': None,
        'min_samples_split': 2,
        'random_state': 42
    },
    'tfidf_params': {
        'ngram_range': (1, 2),
        'max_features': 1000,
        'min_df': 2
    },
    'hazard_mapping': {
        'Chemical': 'High',
        'Sharps': 'High',
        'Pharmaceutical': 'Medium',
        'General': 'Low'
    }
}

# --------------------------
# CORE CLASSIFIER CLASS
# --------------------------
class WasteClassifier:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(**CONFIG['tfidf_params'])
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.model = RandomForestClassifier(**CONFIG['rf_params'])
        self.keyword_patterns = {
            'has_sharp': r'needle|scalpel|blade|sharp',
            'has_chemical': r'chemical|reagent|toxic|solvent',
            'has_pharma': r'medication|drug|pill|pharmaceutical'
        }

    def preprocess_text(self, text: str) -> str:
        """Clean and normalize input text"""
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        tokens = [
            self.lemmatizer.lemmatize(token)
            for token in text.split()
            if token not in self.stop_words
        ]
        return ' '.join(tokens)

    def create_features(self, text: str) -> np.ndarray:
        """Generate combined TF-IDF and domain features"""
        processed_text = self.preprocess_text(text)
        tfidf_features = self.vectorizer.transform([processed_text])
        domain_features = np.array([
            int(bool(re.search(pattern, text.lower())))
            for pattern in self.keyword_patterns.values()
        ]).reshape(1, -1)
        return np.hstack((tfidf_features.toarray(), domain_features))

    def get_hazard_level(self, category: str) -> str:
        """Determine hazard level based on classification"""
        return CONFIG['hazard_mapping'].get(category, 'Low')

# --------------------------
# MAIN WORKFLOW
# --------------------------
def main():
    print("4-Category Medical Waste Classifier")
    print("===================================")
    
    # Initialize environment
    initialize_nltk()
    classifier = WasteClassifier()
    
    # Data processing
    df = load_and_prepare_data()
    augmented_df = augment_dataset(df)
    
    # Feature engineering
    X, y = prepare_features(augmented_df, classifier)
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Model training
    train_model(classifier, X_train, y_train)
    
    # Evaluation
    evaluate_model(classifier, X_test, y_test)
    
    # Save model
    save_model(classifier)
    launch_interface(classifier)

# --------------------------
# DATA PROCESSING FUNCTIONS
# --------------------------
def initialize_nltk() -> None:
    """Handle NLTK resource setup"""
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context
    
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)

def load_and_prepare_data() -> pd.DataFrame:
    """Load and validate dataset"""
    df = pd.read_csv(CONFIG['data_path'])
    valid_categories = ['Chemical', 'Sharps', 'Pharmaceutical', 'General']
    if not set(df['Category']).issubset(valid_categories):
        raise ValueError("Dataset contains invalid categories")
    return df

def augment_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Create augmented dataset variations"""
    augmented = []
    for desc, cat in zip(df['Description'], df['Category']):
        augmented.append((desc, cat))
        if len(desc.split()) > 3:
            words = desc.split()
            augmented.append((' '.join(words[1:] + [words[0]]), cat))
    return pd.DataFrame(augmented, columns=['Description', 'Category'])

# --------------------------
# MODEL FUNCTIONS
# --------------------------
def prepare_features(df: pd.DataFrame, classifier: WasteClassifier) -> Tuple[np.ndarray, pd.Series]:
    """Create feature matrix and labels"""
    df['Processed'] = df['Description'].apply(classifier.preprocess_text)
    X_text = df['Processed']
    classifier.vectorizer.fit(X_text)
    
    domain_features = df['Description'].apply(
        lambda x: [int(bool(re.search(p, x.lower()))) 
        for p in classifier.keyword_patterns.values()]
    ).apply(pd.Series)
    
    X_tfidf = classifier.vectorizer.transform(X_text)
    return np.hstack((X_tfidf.toarray(), domain_features)), df['Category']

def split_data(X: np.ndarray, y: pd.Series) -> tuple:
    """Split dataset into train/test sets"""
    return train_test_split(
        X, y,
        test_size=CONFIG['test_size'],
        random_state=CONFIG['random_state'],
        stratify=y
    )

def train_model(classifier: WasteClassifier, X_train: np.ndarray, y_train: pd.Series) -> None:
    """Train the classification model"""
    print("\nTraining model...")
    classifier.model.fit(X_train, y_train)

def evaluate_model(classifier: WasteClassifier, X_test: np.ndarray, y_test: pd.Series) -> None:
    """Evaluate model performance"""
    y_pred = classifier.model.predict(X_test)
    print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

def save_model(classifier: WasteClassifier) -> None:
    """Serialize trained model"""
    model_package = {
        'model': classifier.model,
        'vectorizer': classifier.vectorizer,
        'preprocessor': classifier.preprocess_text,
        'keyword_patterns': classifier.keyword_patterns,
        'hazard_mapping': CONFIG['hazard_mapping']
    }
    with open(CONFIG['model_save_path'], 'wb') as f:
        pickle.dump(model_package, f)
    print(f"\nModel saved to {CONFIG['model_save_path']}")

# --------------------------
# USER INTERFACE
# --------------------------
def launch_interface(classifier: WasteClassifier) -> None:
    """Interactive classification interface"""
    print("\nClassification Ready")
    print("Enter waste descriptions (Chemical/Sharps/Pharmaceutical/General)")
    print("Type 'quit' to exit\n" + "="*50)
    
    while True:
        user_input = input("\nWaste description: ").strip()
        if user_input.lower() == 'quit':
            break
            
        try:
            features = classifier.create_features(user_input)
            proba = classifier.model.predict_proba(features)[0]
            prediction = classifier.model.predict(features)[0]
            confidence = max(proba) * 100
            hazard = classifier.get_hazard_level(prediction)
            
            display_results(prediction, hazard, confidence, proba, classifier)
            
        except Exception as e:
            print(f"\nError: {str(e)}")

def display_results(prediction: str, hazard: str, confidence: float, 
                   proba: np.ndarray, classifier: WasteClassifier) -> None:
    """Display classification results"""
    print(f"\n{' RESULT '.center(50, '=')}")
    print(f"Category: {prediction}")
    print(f"Hazard Level: {hazard} ({get_hazard_symbol(hazard)})")
    print(f"Confidence: {confidence:.2f}%")
    print(f"\nDisposal: {get_disposal(prediction)}")
    
    if confidence < CONFIG['confidence_threshold'] * 100:
        print("\n[Low Confidence] Possible alternatives:")
        sorted_probs = sorted(zip(classifier.model.classes_, proba),
                            key=lambda x: x[1], reverse=True)[1:3]
        for cat, prob in sorted_probs:
            print(f"- {cat}: {prob*100:.2f}%")
    
    print("="*50)

def get_hazard_symbol(hazard: str) -> str:
    """Return visual hazard indicator"""
    return {
        'High': '🔴',
        'Medium': '🟡',
        'Low': '🟢'
    }.get(hazard, '⚪')

def get_disposal(category: str) -> str:
    """Get disposal recommendations"""
    protocols = {
        'Chemical': "Yellow chemical container - Use PPE and ventilate area",
        'Sharps': "Sharps bin - Do not recap needles",
        'Pharmaceutical': "Blue pharma bin - Separate liquids from solids",
        'General': "Black general waste - Standard disposal procedures"
    }
    return protocols.get(category, "Consult safety guidelines - Unknown category")

if __name__ == "__main__":
    main()
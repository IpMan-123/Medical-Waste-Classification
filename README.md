# Medical Waste Classification System

## Overview
This web application classifies medical waste descriptions into appropriate categories and determines their hazard levels. Using natural language processing (NLP) and machine learning techniques, it provides healthcare professionals with quick, accurate waste disposal guidance to ensure compliance with medical waste regulations and safety protocols.

## Features

- **Text-Based Classification**: Analyze waste descriptions and classify into categories (Chemical, Sharps, Pharmaceutical, General)
- **Hazard Level Assessment**: Automatic evaluation of waste descriptions for hazard levels (Low, Medium, High)
- **Visualization**:
  - Confidence distribution chart showing prediction probabilities
  - Animated vertical hazard level indicator that moves from position 0 to the appropriate level (L, M, H)
  - Disposal method recommendations with color-coded guidance
- **User-Friendly Interface**: Clean, responsive design for ease of use in clinical settings
- **Real-Time Processing**: Instant classification results with visual feedback

## Requirements

- Python 3.12+
- Flask 3.1.0
- scikit-learn 1.6.1
- NLTK 3.9.1
- pandas 2.2.3
- NumPy 2.2.4
- Matplotlib 3.10.1

## Installation

1. **Clone the repository**
   ```
   git clone [repository-url]
   cd medical-waste-classification
   ```

2. **Create and activate a virtual environment**
   ```
   python -m venv medical_waste_env
   
   # On Windows
   medical_waste_env\Scripts\activate.bat
   
   # On Unix or MacOS
   source medical_waste_env/bin/activate
   ```

3. **Install the dependencies**
   ```
   pip install -r requirements.txt
   ```

4. **Initialize NLTK data (if not already done)**
   ```python
   python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet')"
   ```

## Usage

1. **Start the Flask application**
   ```
   python app.py
   ```

2. **Access the web interface**
   - Open your browser and go to `http://127.0.0.1:5000/`
   - Enter a medical waste description in the text field
   - Click "Classify" to process the input
   - View the classification result, hazard level, and disposal recommendations

## Project Structure

```
.
├── app.py                        # Main Flask application
├── medical_waste_dataset.csv     # Training dataset
├── medical_waste_dataset.py      # Dataset processing script
├── waste_classifier_model.py     # ML model training script
├── waste_classifier_rf.pkl       # Serialized Random Forest model
├── requirements.txt              # Project dependencies
├── README.md                     # Project documentation
└── templates/                    # HTML templates
    ├── error.html                # Error page template
    ├── index.html                # Main application interface
    └── result.html               # Results display template
```

## Technical Details

### Machine Learning Model

- **Algorithm**: Random Forest Classifier
- **Text Processing**: TF-IDF Vectorization with custom preprocessing
- **Feature Engineering**: Combines TF-IDF features with domain-specific keyword patterns
- **Model Performance**: Trained and evaluated on medical waste description dataset

### Natural Language Processing

- Text normalization includes:
  - Lowercasing
  - Punctuation removal
  - Stop word filtering
  - Lemmatization
- Domain-specific keyword pattern matching enhances classification accuracy

### Web Application

- **Backend**: Flask web framework
- **Visualization**: Matplotlib for generating visualizations
- **Frontend**: HTML/CSS with JavaScript for animations
- **Data Flow**:
  1. User inputs waste description
  2. Text is preprocessed and features extracted
  3. Model predicts waste category and hazard level
  4. Results and visualizations are generated and displayed

## Hazard Level Indicator

The application features an animated vertical hazard level indicator that provides visual feedback about the waste classification:

- **Low Risk (L)**: Green indicator at the bottom position
- **Medium Risk (M)**: Yellow indicator at the middle position
- **High Risk (H)**: Red indicator at the top position

The indicator starts at position 0 and smoothly animates to the appropriate level after classification.

## Disposal Recommendations

Based on the classification result, the system provides specific disposal instructions:

- **Chemical Waste**: Yellow container with PPE and ventilation requirements
- **Sharps**: Sharps bin with handling precautions
- **Pharmaceutical**: Blue pharmaceutical waste bin with separation guidelines
- **General Waste**: Standard black waste disposal

## Future Enhancements

- Integration with barcode/QR code scanning
- Mobile application development
- Connection to hospital waste management systems
- Expanded waste categories and multi-language support
- Real-time analytics and reporting features

## Contributors

- [Your Name]
- [Additional Contributors]

## License

[License Information]

## Acknowledgments

- [Academic Advisor/Professor]
- [University/Institution]
- [Any other acknowledgments]
## **Emotion Classifier Repository Documentation**

## Overview
This repository contains two emotion detection systems:

1. **Logistic Regression-based Model** : Deployed via Gradio for real-time emotion prediction (using TF-IDF features).
2. **Hugging Face GO Emotions BERT Model** : A Jupyter Notebook for detecting 27 fine-grained emotions using a pre-trained BERT model. (EXTRA FOR BETTER PERFORMANCE)

## Installation Guide
  Prerequisites: 
  Python 3.8+
  pip or conda

## Setup Instructions
1. Install Requirements:
   ```bash
   pip install -r requirements.txt

  This installs packages like gradio, joblib, pandas, scikit-learn, and transformers.

2. Make a .env file if you want to run **go-emotion** model for better performance though not deployed:
   ```bash
   HF_TOKEN = paste_your_huggingface_read_token

## Usage
 I. **Logistic Regression Model (Gradio App)**
   ```bash
   python src/app.py
```

Access the web interface at  **http://localhost:port_number**

Input text in the Gradio UI to get emotion predictions (e.g., "happy", "sad", "angry").


**Key Components**
  Data Flow :

  1. Text input → Preprocessing (src/preprocess.py) → TF-IDF Vectorization → Logistic Regression Prediction.
    
  2. Models :

      models/tfidf_vectorizer.joblib: Trained TF-IDF vectorizer

      models/emotion_classifier.joblib: Trained logistic regression model



  II. **GO Emotions BERT Model (Jupyter Notebook)**
    

  **Run the Notebook:**

  1. Open using_go_emotions_bert.ipynb.
  2. Execute cells to:
  3. Load the pre-trained Hugging Face go-emotions BERT model.
  4. Process text data (from data/DailyDialog.csv or custom inputs).
  5. Predict 27 emotions (e.g., "admiration", "annoyance", "approval").

  **Key Features**
    1. Uses AutoTokenizer and TFAutoModelForSequenceClassification.
    2. No training required; leverages Hugging Face's pre-trained weights.


## Directory Structure

```
emotion_classifier/
├── .gradio/
│   └── flagged/                    # Stores flagged user inputs (for debugging)
│     └── Predicted Emotion/        # Output directory for predictions
│         └── dataset1.csv
├── data/
│   ├── DailyDialog.csv             # Training/data dataset
├── models/
│   ├── emotion_classifier.joblib
│   └── tfidf_vectorizer.joblib
├── src/
│   ├── __pycache__/
│   ├── app.py                      # Gradio application entry point
│   ├── config.py                   # Configuration parameters (e.g., model paths)
│   ├── confusion_matrix.png        # Model performance visualization
│   ├── emotion_classifier.joblib
│   ├── model.py                    # Core model logic
│   ├── preprocess_and_train_model_regression.ipynb   # Training script
│   ├── preprocess.py               # Text preprocessing functions
│   └── tfidf_vectorizer.joblib
├── .env                            # Environment configuration (for HF_TOKEN)
├── .gitignore
├── requirements.txt                # Dependencies
└── using_go_emotions_bert.ipynb    #(The implementation of bert based go_emotion model from huggingface)

```


## **Model Details**
  1. **Logistic Regression Model**

    Features : TF-IDF vectorized text features.

    Training : Uses data/DailyDialog.csv (dialogue snippets labeled with emotions).

    Evaluation : Confusion matrix in src/confusion_matrix.png.

  2. **GO Emotions BERT Model**

    Emotions : 27 fine-grained emotions ( **https://huggingface.co/joeddav/distilbert-base-uncased-go-emotions-student** ).

    Performance : Higher accuracy but slower inference compared to logistic regression.

## Deployment

  Gradio App : Hosted locally by default. For production, deploy using Gradio’s cloud services or a web server or on RENDER.

  Docker : Optional (not included in repo).

## Configuration (`config.py`)

This file centralizes all key settings and mappings for the emotion-classification pipeline:

- **MODEL_PATH**  
  Path to the pre-trained emotion classification model (Joblib format).

  Path to the TF-IDF vectorizer used to transform raw text into feature vectors.

  A dictionary that converts raw model output labels into user-friendly, cleaned emotion names: 

  ```python
  MODEL_PATH = "models\\emotion_classifier.joblib"
  VECTORIZER_PATH = "models\\tfidf_vectorizer.joblib"

  LABEL_MAP = {
    "joy":      "Happy",
    "sadness":  "Sad",
    "anger":    "Angry",
    "neutral":  "Neutral",
    "fear":     "Fear",
    "surprise": "Surprise"
  } ```


## Text Preprocessing (`preprocess.py`)

This module handles all steps required to clean and normalize raw text before it’s fed into the emotion-classifier pipeline:

1. **Resource Setup**  
   - Downloads and makes available the WordNet lemmatizer and English stopword list from NLTK.  
   - Initializes:
     - A **WordNetLemmatizer** for reducing words to their base form.
     - A **stopword set** to filter out common words that don’t carry significant meaning (e.g., “and”, “the”, “is”).

2. **`preprocess_text(text)`**  
   - **Lowercasing & Cleaning**  
     Converts the input string to lowercase and strips out any non-alphabetic characters (numbers, punctuation, etc.).  
   - **Tokenization & Stopword Removal**  
     Splits the cleaned text into individual words and discards any token found in the stopword list.  
   - **Lemmatization**  
     Transforms each remaining token to its lemma (base form) using the WordNet lemmatizer.  
   - **Reassembly**  
     Joins the processed tokens back into a single string, ready for vectorization.  
   - **Error Handling**  
     Catches and logs any exceptions during preprocessing, returning an empty string if an error occurs.

## Model & Training Pipeline (`model.py`)

This module orchestrates data ingestion, model training, evaluation, and persistence for the emotion-classifier.

1. **Constants & Imports**  
   - **DATASET_URL**  
     Remote CSV containing raw dialogues and their emotion labels.  
   - Key libraries:  
     - `pandas` for tabular data handling  
     - `requests` for HTTP access  
     - `scikit-learn` for vectorization, model training, and evaluation  
     - `joblib` for saving/loading fitted objects  
     - `matplotlib` & `seaborn` for plotting the confusion matrix  
     - Local imports: `MODEL_PATH`, `VECTORIZER_PATH`, `LABEL_MAP` (from `config.py`), and `preprocess_text` (from `preprocess.py`)

2. **`load_data_from_url()`**  
   - **Fetch & Read**  
     Downloads the CSV text, splits it into lines, and iterates through each to extract the dialogue and label.  
   - **Robust Parsing**  
     - Handles quoted text containing commas by detecting `","` patterns.  
     - Otherwise splits on the **last** comma to separate text from label.  
   - **Validation & Cleaning**  
     - Converts labels to lowercase and filters out any not in `LABEL_MAP`.  
     - Records (and logs) any skipped or malformed lines for debugging.  
     - Applies the `preprocess_text` function to each raw dialogue, yielding a cleaned-text series and corresponding labels.

3. **`train_model()`**  
   - **Data Split**  
     Calls `load_data_from_url()` and splits the cleaned texts and labels into training (80%) and test (20%) sets, stratified by emotion.  
   - **Vectorization**  
     Initializes a TF–IDF vectorizer (up to 5,000 features), fits it on the training texts, and transforms both train and test sets.  
   - **Model Training**  
     - Trains a Logistic Regression classifier (with balanced class weights and up to 1,000 iterations) on the vectorized training data.  
   - **Persistence**  
     Saves the trained model and vectorizer to disk as Joblib files (`emotion_classifier.joblib` and `tfidf_vectorizer.joblib`).  
   - **Evaluation & Visualization**  
     - Computes and prints accuracy, classification report, and confusion matrix on the test set.  
     - Plots and saves the confusion matrix (`confusion_matrix.png`) with human-readable emotion labels from `LABEL_MAP`.

## Web App Interface (`app.py`)

This script defines and launches a Gradio-based web interface for the emotion classifier.

1. **Imports & Initialization**  
   - Loads the pre-trained model and TF–IDF vectorizer from the paths defined in `config.py`.  
   - Imports the `preprocess_text` function for input cleaning.  
   - Defines an **emoji map** that associates each cleaned emotion label with a corresponding emoji.

2. **`predict_emotion(text)`**  
   - **Preprocessing**  
     Cleans the raw input via `preprocess_text`; if the result is empty, returns an error message.  
   - **Vectorization & Prediction**  
     Transforms the cleaned text into a TF–IDF vector and applies the loaded model to get a raw emotion label.  
   - **Label Mapping & Formatting**  
     Converts the raw label to the user-friendly name (via `LABEL_MAP`) and prefixes it with the matched emoji.  
   - **Error Handling**  
     Catches any exceptions during prediction, logs them, and returns a generic error indicator.

3. **Custom Styling & Examples**  
   - **CSS Snippet**  
     Sets maximum container width, padding, footer styling, and customizes example-button appearance.  
   - **Predefined Examples**  
     Illustrative inputs (with text and emojis) to demonstrate different emotion predictions.

4. **Gradio Interface Configuration**  
   - **Inputs**  
     A multi-line textbox for user messages, with placeholder text and custom CSS class.  
   - **Outputs**  
     A Markdown display that shows the emoji and emotion name.  
   - **Additional Settings**  
     - Title, description (centered HTML snippet), and “soft” theme.  
     - Disables flagging.  
     - Includes the example set defined above.

5. **Launch**  
   When run as a standalone script (`if __name__ == "__main__":`), calls `interface.launch()` to start the local web server.

  
## Saved Model Artifacts (`.joblib` Files)

After training, two key artifacts are serialized to disk in Joblib format for fast loading and reliable persistence:

1. **`emotion_classifier.joblib`**  
   - **Contents:** A fitted `LogisticRegression` model that maps TF–IDF–vectorized text inputs to one of the six raw emotion labels (`joy`, `sadness`, `anger`, `neutral`, `fear`, `surprise`).  
   - **Usage:**  
     ```python
     from joblib import load
     model = load("models/emotion_classifier.joblib")
     prediction = model.predict(vectorized_input)
     ```
   - **Why Joblib?**  
     Joblib is optimized for large NumPy arrays under the hood, so it loads and saves scikit-learn objects much faster than pickle.

2. **`tfidf_vectorizer.joblib`**  
   - **Contents:** A fitted `TfidfVectorizer` (configured with up to 5,000 features) that transforms raw–text strings into numeric feature vectors compatible with the classifier.  
   - **Usage:**  
     ```python
     from joblib import load
     vectorizer = load("models/tfidf_vectorizer.joblib")
     X = vectorizer.transform(["cleaned text here"])
     ```
   - **Why Joblib?**  
     Just like with the model, Joblib efficiently serializes the internal sparse matrices and vocabulary lookup structures of the TF–IDF transformer.







## Additional Resources

  1. Datasets :
    DailyDialog for training (**https://www.kaggle.com/datasets/mgmitesh/sentiment-analysis-dataset**)
    GO Emotions for BERT model (**https://huggingface.co/joeddav/distilbert-base-uncased-go-emotions-student**)


## Improvements : Add more emotions, experiment with CNN/LSTM models, or optimize BERT for speed.

## Contributing

Report bugs or suggest improvements via GitHub Issues.
Pull requests welcome!
   

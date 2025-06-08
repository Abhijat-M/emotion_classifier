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


## Additional Resources

  1. Datasets :
    DailyDialog for training (**https://www.kaggle.com/datasets/mgmitesh/sentiment-analysis-dataset**)
    GO Emotions for BERT model (**https://huggingface.co/joeddav/distilbert-base-uncased-go-emotions-student**)


## Improvements : Add more emotions, experiment with CNN/LSTM models, or optimize BERT for speed.

## Contributing

Report bugs or suggest improvements via GitHub Issues.
Pull requests welcome!
   

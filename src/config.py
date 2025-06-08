MODEL_PATH = "models\\emotion_classifier.joblib"
VECTORIZER_PATH = "models\\tfidf_vectorizer.joblib"

# Map raw emotions to cleaned labels
LABEL_MAP = {
    "joy": "Happy",
    "sadness": "Sad",
    "anger": "Angry",
    "neutral": "Neutral",
    "fear": "Fear",
    "surprise": "Surprise"
}
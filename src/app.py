import joblib
import gradio as gr
from config import MODEL_PATH, VECTORIZER_PATH, LABEL_MAP
from preprocess import preprocess_text

# Load model and vectorizer
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

# Emoji mapping for emotions
EMOJI_MAP = {
    "Happy": "😊",
    "Sad": "😢",
    "Angry": "😠",
    "Neutral": "😐"
}

def predict_emotion(text):
    """Predict emotion from input text ."""
    try:
        cleaned_text = preprocess_text(text)
        if not cleaned_text:
            return {"error": "Invalid input after preprocessing"}
        
        vec = vectorizer.transform([cleaned_text])
        raw_prediction = model.predict(vec)[0]
        emotion = LABEL_MAP.get(raw_prediction, "Unknown")
        
        # Return formatted result with emoji
        return f"{EMOJI_MAP.get(emotion, '')} **{emotion}**"
        
    except Exception as e:
        print(f"Prediction error: {e}")
        return "❌ Error: Failed to process input"

# Custom CSS for better styling
custom_css = """
.container {
    max-width: 800px;
    margin: auto;
    padding: 20px;
}
.footer {
    text-align: center;
    padding: 10px;
    opacity: 0.7;
}
.example-btn {
    background-color: #f0f0f0 !important;
    border-radius: 8px !important;
    font-weight: normal !important;
}
"""

# Enhanced examples with emojis
examples = [
    ["I'm so happy today! 😊"],
    ["This makes me really angry! 😠"],
    ["I feel nothing... 🥱"],
    ["I'm feeling down... 😢"],
    ["Just another average day 😐"],
    ["I won the lottery! 🎉"]
]

# Gradio Interface with enhanced configuration
interface = gr.Interface(
    fn=predict_emotion,
    inputs=gr.Textbox(
        lines=4,
        placeholder="Type your message here...",
        label="Input Text",
        elem_classes=["input-box"]
    ),
    outputs=gr.Markdown(
        label="Prediction Result",
        elem_classes=["output-box"]
    ),
    examples=examples,
    title=" Emotion Classifier",
    description="""
    <div style='text-align:center; font-size:1.2em;'>
        Detect emotions in text with style! <br>
        Supports Happy 😊, Sad 😢, Angry 😠, and Neutral 😐
    </div>
    """,
    theme="soft",
    css=custom_css,
    allow_flagging="never"
)

if __name__ == "__main__":
    interface.launch()
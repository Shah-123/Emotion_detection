"""FastAPI backend for the trained BiLSTM emotion model."""

import json
import threading
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, StringConstraints
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json

MODEL_DIR = Path(__file__).parent / "saved_model"
META = json.loads((MODEL_DIR / "metadata.json").read_text(encoding="utf-8"))
CLASSES = META["classes"]

tokenizer = tokenizer_from_json((MODEL_DIR / "tokenizer.json").read_text(encoding="utf-8"))
model = load_model(MODEL_DIR / "bilstm_emotion_model.keras")

# ponytail: one global lock, Keras predict isn't thread-safe. Swap for a worker
# pool or TF Serving if more than a handful of concurrent requests matter.
_lock = threading.Lock()

# Text is stripped first so whitespace-only input is rejected instead of scoring
# an all-padding sequence.
Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=5000)]

app = FastAPI(title="Emotion Detection API", version="1.0")


class TextIn(BaseModel):
    text: Text


def predict(text: str) -> dict:
    """Preprocess exactly as training did: metadata's post/post padding.

    The notebook's own predict_emotion() uses padding='pre', which does not match
    how the model was trained (cells 24-25) - metadata.json is the source of truth.
    """
    padded = pad_sequences(
        tokenizer.texts_to_sequences([text]),
        maxlen=META["maxlen"],
        padding=META["padding"],
        truncating=META["truncating"],
    )
    with _lock:
        probs = model.predict(padded, verbose=0)[0]
    scores = {label: float(p) for label, p in zip(CLASSES, probs)}
    top = max(scores, key=scores.get)
    return {"emotion": top, "confidence": scores[top], "probabilities": scores}


@app.post("/predict")
def predict_endpoint(payload: TextIn) -> dict:
    return predict(payload.text)


# Mounted last: it claims "/", so it must not shadow the routes above.
app.mount("/", StaticFiles(directory=Path(__file__).parent / "static", html=True), name="ui")

predict("warm up the graph so the first real request is not slow")

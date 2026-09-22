# Emotion Detection

An Emotion Detection project using Python, Hugging Face `datasets`, and Machine Learning / NLP models.

## Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Shah-123/Emotion_detection.git
   cd Emotion_detection
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```


3. **Run the Notebook:**
   Open `Emotion.ipynb` in VS Code, JupyterLab, or your preferred notebook runner.

## API & Web UI

Start the server (from the repo root, so `saved_model/` resolves):

```bash
uvicorn app:app --reload
```

- **UI:** http://127.0.0.1:8000
- **Interactive docs:** http://127.0.0.1:8000/docs

### `POST /predict`

```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d "{\"text\": \"i feel so grateful today\"}"
```

```json
{
  "emotion": "joy",
  "confidence": 0.9891,
  "probabilities": { "sadness": 0.004, "joy": 0.989, "love": 0.003, "anger": 0.002, "fear": 0.001, "surprise": 0.001 }
}
```

Blank or whitespace-only text returns `422`.

Run the checks with `python test_app.py`.

> **Note:** inference uses `padding='post'` from `saved_model/metadata.json`, matching how
> the model was trained. The notebook's `predict_emotion()` helper uses `'pre'`, which does
> not match training.

### Model limitation

The training data (`dair-ai/emotion`) is first-person diary text, so the model leans heavily
on "i feel …" phrasing. In-distribution input is accurate (`i feel furious that they lied
to me again` → anger 99%), but paraphrases drift (`everything feels heavy and pointless
lately` → joy 86%). Retraining or fine-tuning a transformer is the fix, not a change to
this API.

### Input preprocessing

`clean()` in `app.py` runs before tokenizing. It deliberately does **not** re-implement
what the saved tokenizer already does — its `filters` config strips punctuation and
lowercases, exactly as at training time, so `@#$%^&* i feel so loved <<>>` already works.
It closes only the two gaps those filters leave:

| input | without `clean()` | with `clean()` |
|---|---|---|
| `I'm furious...` | `im` → `<unk>` | `im` → token 17 |
| `I’m furious...` (phone keyboard) | `im` → `<unk>` | `im` → token 17 |
| `i feel élated` | `élated` → `<unk>` | `elated` folded to ASCII |

The apostrophe is the one punctuation mark missing from `filters`, and the corpus writes
contractions bare (`im` is its 17th most common word). Rewriting the test set the way a
person actually types — real apostrophes, capitals, trailing `!` — costs 0.20pp
(92.55% → 92.35%); `clean()` restores the full 92.55%.

Both transforms only move input toward the training distribution, never away:
`test_clean_is_noop_on_training_corpus` asserts `clean()` leaves all 2000 corpus rows
byte-identical, so it cannot silently cause the drift it exists to prevent.

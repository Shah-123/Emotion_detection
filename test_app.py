"""Run directly: python test_app.py"""

from fastapi.testclient import TestClient

import app

client = TestClient(app.app)


def test_padding_matches_training():
    """Training used post-padding (notebook cells 24-25); 'pre' would shift every
    short input and quietly degrade accuracy."""
    padded = app.pad_sequences(
        app.tokenizer.texts_to_sequences(["i feel happy"]),
        maxlen=app.META["maxlen"],
        padding=app.META["padding"],
        truncating=app.META["truncating"],
    )[0]
    assert padded[0] != 0, "tokens must start at index 0, not be right-aligned"
    assert padded[-1] == 0, "padding must land at the end"
    assert len(padded) == 50


def test_known_emotions():
    for text, expected in [
        ("i am so happy and thrilled about this", "joy"),
        ("i feel hopeless and empty inside", "sadness"),
        ("it makes my blood boil that they lied", "anger"),
    ]:
        got = client.post("/predict", json={"text": text}).json()["emotion"]
        assert got == expected, f"{text!r} -> {got}, expected {expected}"


def test_response_shape():
    body = client.post("/predict", json={"text": "i feel loved"}).json()
    assert set(body["probabilities"]) == set(app.CLASSES)
    assert abs(sum(body["probabilities"].values()) - 1) < 1e-4
    assert body["confidence"] == max(body["probabilities"].values())


def test_clean_is_noop_on_training_corpus():
    """The corpus is bare lowercase ASCII. If cleaning altered it, inference would
    diverge from training - this is the guard that it only ever moves input closer."""
    import csv

    with open("data/test.csv", encoding="utf-8") as f:
        texts = [row["text"] for row in csv.DictReader(f)]
    assert texts, "test.csv is empty"
    changed = [t for t in texts if app.clean(t) != t]
    assert not changed, f"{len(changed)} corpus rows altered, e.g. {changed[:2]}"


def test_clean_normalizes_user_typing():
    assert app.clean("I'm done") == "Im done"          # straight apostrophe
    assert app.clean("I’m done") == "Im done"      # phone keyboard curly quote
    assert app.clean("café élated") == "cafe elated"  # accents folded
    assert app.clean("wow!!! @#$ really?") == "wow!!! @#$ really?"  # left to the tokenizer


def test_contraction_reaches_vocab():
    """"I'm" must tokenize to the corpus's "im" (rank 17), not to <unk>."""
    oov = app.tokenizer.word_index[app.tokenizer.oov_token]
    seq = app.tokenizer.texts_to_sequences([app.clean("I'm not happy")])[0]
    assert oov not in seq, f"still contains <unk>: {seq}"
    assert seq[0] == app.tokenizer.word_index["im"]


def test_rejects_blank_input():
    for bad in ["", "   ", "\n\t"]:
        assert client.post("/predict", json={"text": bad}).status_code == 422


def test_serves_ui():
    res = client.get("/")
    assert res.status_code == 200 and "Emotion Detection" in res.text


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)

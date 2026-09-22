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

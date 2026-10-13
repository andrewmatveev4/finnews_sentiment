import json

import pytest

from finnews_sentiment.features.preprocess import ID2LABEL
from finnews_sentiment.models.predict import load_model, predict
from finnews_sentiment.models.train_improved import make_pipeline


@pytest.fixture(scope="module")
def tiny_model():
    texts = ["profit rose", "sales grew strongly", "profit fell", "sales dropped sharply",
             "meeting on tuesday", "company is based in helsinki"]
    labels = [2, 2, 0, 0, 1, 1]
    return make_pipeline().fit(texts, labels)


def test_single_string_returns_list_of_one(tiny_model):
    result = predict("profit rose", tiny_model, ID2LABEL)
    assert isinstance(result, list) and len(result) == 1


def test_output_contract(tiny_model):
    item = predict("profit rose", tiny_model, ID2LABEL)[0]
    assert set(item) == {"text", "label", "confidence", "probabilities", "unknown_words"}
    assert item["label"] in ID2LABEL.values()
    assert set(item["probabilities"]) == set(ID2LABEL.values())
    assert sum(item["probabilities"].values()) == pytest.approx(1, abs=0.01)
    assert item["confidence"] == max(item["probabilities"].values())
    json.dumps(item)


def test_preprocessing_is_inside_model(tiny_model):
    raw = predict("PROFIT <b>ROSE</b>", tiny_model, ID2LABEL)[0]
    clean = predict("profit rose", tiny_model, ID2LABEL)[0]
    assert raw["probabilities"] == clean["probabilities"]


@pytest.mark.parametrize("bad", ["", "   ", None, 42])
def test_bad_input_raises(tiny_model, bad):
    with pytest.raises(ValueError):
        predict([bad], tiny_model, ID2LABEL)


def test_order_is_preserved(tiny_model):
    texts = ["profit fell", "meeting on tuesday", "profit rose"]
    assert [r["text"] for r in predict(texts, tiny_model, ID2LABEL)] == texts


def test_missing_model_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="Сначала обучите"):
        load_model(tmp_path / "nope.joblib")


def test_unknown_words_reported(tiny_model):
    item = predict("Profit PLUMMETED", tiny_model, ID2LABEL)[0]
    assert item["unknown_words"] == ["plummeted"]

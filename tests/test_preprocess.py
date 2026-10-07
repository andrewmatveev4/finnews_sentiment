import pytest

from finnews_sentiment.features.preprocess import preprocess_text


@pytest.mark.parametrize("raw, expected", [
    ("Profit ROSE", "profit rose"),
    ("Profit <b>rose</b>", "profit rose"),
    ("EUR5 .9 m", "eur5.9 m"),
    ("  profit \t rose \n ", "profit rose"),
    ("", ""),
])
def test_preprocess_examples(raw, expected):
    assert preprocess_text(raw) == expected


@pytest.mark.parametrize("word", ["up", "down", "no", "not", "%", "45"])
def test_sentiment_signal_is_kept(word):
    assert word in preprocess_text(f"Profit went {word} this year .").split()


def test_opposite_news_stay_different():
    down = preprocess_text("Operating profit went down to EUR 2.1 mn from EUR 3.5 mn .")
    up = preprocess_text("Operating profit went up to EUR 3.5 mn from EUR 2.1 mn .")
    assert down != up


@pytest.mark.parametrize("value", [None, float("nan"), 42])
def test_non_string_returns_empty(value):
    assert preprocess_text(value) == ""


def test_idempotent():
    once = preprocess_text("Profit <b>ROSE</b>  to EUR5 .9 m")
    assert preprocess_text(once) == once

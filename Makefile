PY := python
URL := https://huggingface.co/datasets/takala/financial_phrasebank/resolve/main/data/FinancialPhraseBank-v1.0.zip
RAW_ZIP := data/raw/FinancialPhraseBank-v1.0.zip
RAW_TXT := data/raw/FinancialPhraseBank-v1.0/Sentences_50Agree.txt
INTERIM := data/interim/phrasebank_50agree.csv
PROCESSED := data/processed/clean_news.csv

.PHONY: all data eda train analyze demo test lint

all: data train analyze demo

data:
	mkdir -p data/raw
	curl -L -o $(RAW_ZIP) $(URL)
	unzip -o -q $(RAW_ZIP) -d data/raw
	$(PY) -m finnews_sentiment.data.load_data --input $(RAW_TXT) --output $(INTERIM)
	$(PY) -m finnews_sentiment.features.preprocess --input $(INTERIM) --output $(PROCESSED)

eda:
	$(PY) -m finnews_sentiment.visualization.eda

train:
	$(PY) -m finnews_sentiment.models.train_model
	$(PY) -m finnews_sentiment.models.train_improved

analyze:
	$(PY) -m finnews_sentiment.models.error_analysis > reports/day05_error_analysis_output.txt

demo:
	$(PY) -m finnews_sentiment.models.predict --input examples/new_news.csv --output reports/predictions.csv

test:
	$(PY) -m pytest -q

lint:
	flake8 --max-line-length=100 src/ tests/
install:
	pip install --upgrade pip &&\
	pip install -r requirements.txt

lint:
	pylint --disable=R,C $$(git ls-files '*.py')

test:
	python -m pytest -vv --cov=main test_*.py

format:
	black $$(git ls-files '*.py')

deploy:
	echo "deploy command goes here"

all: install lint test format deploy
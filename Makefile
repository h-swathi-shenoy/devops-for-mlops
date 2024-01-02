install:
	pip install --upgrade pip &&\
	pip install -r requirements.txt

lint:
	pylint --disable=R,C --exit-zero *.py src

# test:
# 	python -m pytest -vvv --cov=src test_*.py

format:
	black *.py src/*.py

all: install lint format #test format
install:
	pip install --upgrade pip &&\
	pip install -r requirements.txt

lint:
	pylint --disable=R,C *.py src

# test:
# 	python -m pytest -vvv --cov=src test_*.py

# format:
# 	black *.py src/*.py

all: install lint #test format
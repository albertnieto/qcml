PYTHON := python3

all: test lint

install:
	$(PYTHON) -m pip install -r requirements.txt

test:
	tox

lint:
	tox -e lint

build:
	$(PYTHON) -m build

clean:
	rm -rf build dist *.egg-info
	find . -name '__pycache__' -exec rm -rf {} +

tox:
	tox

pytest:
	pytest

format:
	black qcml tests

release: clean build
	$(PYTHON) -m twine upload dist/*

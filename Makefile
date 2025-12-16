.PHONY: install test run docker-build docker-run clean

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v

run:
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

docker-build:
	docker build -t market-prediction-agent .

docker-run:
	docker run -p 8000:8000 --env-file .env market-prediction-agent

docker-compose-up:
	docker-compose up -d

docker-compose-down:
	docker-compose down

clean:
	find . -type d -name __pycache__ -exec rm -r {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -r {} +

lint:
	flake8 src/ tests/ --max-line-length=120 --ignore=E501,W503
	mypy src/ --ignore-missing-imports

format:
	black src/ tests/ --line-length=120
	isort src/ tests/


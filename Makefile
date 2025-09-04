# Fraud Detection MLOps Makefile
.PHONY: help setup install test lint format clean build run docker-build docker-run k8s-deploy sample-data train serve

# Help
help: ## Show this help message
	@echo "Fraud Detection MLOps - Available Commands:"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# Setup
setup: ## Initial project setup
	@echo "Setting up fraud detection MLOps project..."
	python3 -m venv venv
	@echo "Virtual environment created. Activate with: source venv/bin/activate"

install: ## Install dependencies
	pip install --upgrade pip
	pip install -r requirements.txt

install-dev: ## Install development dependencies
	pip install --upgrade pip
	pip install -r requirements.txt
	pip install pre-commit
	pre-commit install

# Data
sample-data: ## Generate sample fraud detection data
	@echo "Generating sample data..."
	python -c "from src.utils.helpers import generate_sample_data; df = generate_sample_data(5000, 0.1); df.to_csv('data/raw/sample_fraud_data.csv', index=False); print('Sample data saved to data/raw/sample_fraud_data.csv')"

check-data: ## Check if credit card data exists
	@if [ -f "data/raw/creditcard.csv" ]; then \
		echo "✅ Credit card dataset found: $(shell wc -l < data/raw/creditcard.csv) records"; \
	else \
		echo "❌ Credit card dataset not found at data/raw/creditcard.csv"; \
	fi

# Model Training
train: ## Train fraud detection model with credit card data
	@echo "Training fraud detection model..."
	@if [ -f "data/raw/creditcard.csv" ]; then \
		python src/training/train_model.py --data-path data/raw/creditcard.csv --model-type random_forest --output-dir models/trained; \
	else \
		echo "❌ Credit card data not found. Using sample data..."; \
		python src/training/train_model.py --data-path data/raw/sample_fraud_data.csv --model-type random_forest --output-dir models/trained; \
	fi

train-credit-card: ## Train specifically with credit card dataset
	@echo "Training with credit card dataset..."
	@if [ -f "data/raw/creditcard.csv" ]; then \
		python src/training/train_model.py --data-path data/raw/creditcard.csv --model-type random_forest --output-dir models/trained --test-size 0.3; \
	else \
		echo "❌ Credit card dataset not found at data/raw/creditcard.csv"; \
		exit 1; \
	fi

train-all: ## Train all model types with available data
	@echo "Training all model types..."
	@DATA_PATH="data/raw/creditcard.csv"; \
	if [ ! -f "$$DATA_PATH" ]; then \
		echo "Using sample data instead..."; \
		DATA_PATH="data/raw/sample_fraud_data.csv"; \
	fi; \
	python src/training/train_model.py --data-path $$DATA_PATH --model-type random_forest --output-dir models/trained; \
	python src/training/train_model.py --data-path $$DATA_PATH --model-type logistic_regression --output-dir models/trained; \
	python src/training/train_model.py --data-path $$DATA_PATH --model-type isolation_forest --output-dir models/trained

train-sample: ## Train with sample data
	@echo "Training with sample data..."
	python src/training/train_model.py --data-path data/raw/sample_fraud_data.csv --model-type random_forest --output-dir models/trained

# Serving
serve: ## Start the fraud detection API server
	@echo "Starting fraud detection API..."
	cd src && python -m uvicorn serving.app:app --host 0.0.0.0 --port 8000 --reload

# Testing
test: ## Run tests
	@echo "Running tests..."
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term

test-api: ## Test API endpoints
	@echo "Testing API endpoints..."
	curl -X GET http://localhost:8000/health
	curl -X POST http://localhost:8000/predict \
		-H "Content-Type: application/json" \
		-d '{"amount": 1500.0, "merchant_category": "online", "hour": 14, "customer_age": 35}'

# Code Quality
lint: ## Run linting
	@echo "Running linting..."
	flake8 src tests
	mypy src

format: ## Format code
	@echo "Formatting code..."
	black src tests notebooks
	isort src tests

check: ## Check code quality
	@echo "Checking code quality..."
	black --check src tests
	isort --check-only src tests
	flake8 src tests

# Docker
docker-build: ## Build Docker image
	@echo "Building Docker image..."
	docker build -t fraud-detection:latest -f infra/docker/Dockerfile .

docker-run: ## Run Docker container
	@echo "Running Docker container..."
	docker run -p 8000:8000 -v $(PWD)/models:/app/models fraud-detection:latest

docker-compose-up: ## Start all services with docker-compose
	@echo "Starting all services..."
	docker-compose up -d

docker-compose-down: ## Stop all services
	@echo "Stopping all services..."
	docker-compose down

# Kubernetes
k8s-namespace: ## Create Kubernetes namespace
	kubectl apply -f infra/k8s/namespace.yaml

k8s-deploy: k8s-namespace ## Deploy to Kubernetes
	@echo "Deploying to Kubernetes..."
	kubectl apply -f infra/k8s/deployment.yaml -n fraud-detection

k8s-delete: ## Delete Kubernetes deployment
	@echo "Deleting Kubernetes deployment..."
	kubectl delete -f infra/k8s/deployment.yaml -n fraud-detection

k8s-status: ## Check Kubernetes deployment status
	kubectl get pods,svc,ing -n fraud-detection

# Helm
helm-install: ## Install Helm chart
	@echo "Installing Helm chart..."
	helm install fraud-detection ./charts/fraud-detection

helm-upgrade: ## Upgrade Helm chart
	@echo "Upgrading Helm chart..."
	helm upgrade fraud-detection ./charts/fraud-detection

helm-uninstall: ## Uninstall Helm chart
	@echo "Uninstalling Helm chart..."
	helm uninstall fraud-detection

# Monitoring
monitoring-up: ## Start monitoring stack
	@echo "Starting monitoring stack..."
	docker-compose -f infra/docker/docker-compose.yml up prometheus grafana -d

# Clean
clean: ## Clean up generated files
	@echo "Cleaning up..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type f -name ".coverage" -delete
	find . -type d -name "htmlcov" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +

clean-models: ## Clean trained models
	@echo "Cleaning trained models..."
	rm -rf models/trained/*
	touch models/trained/.gitkeep

# Documentation
docs: ## Generate documentation
	@echo "Generating documentation..."
	@echo "📖 Documentation will be available at: http://localhost:8000/docs (when API is running)"

# Complete workflow
setup-project: setup install sample-data ## Complete project setup
	@echo "✅ Project setup complete!"
	@echo "Next steps:"
	@echo "1. Activate virtual environment: source venv/bin/activate"
	@echo "2. Train model: make train"
	@echo "3. Start API: make serve"

quick-start: install sample-data train serve ## Quick start - install, generate data, train, and serve
	@echo "🚀 Quick start complete! API should be running at http://localhost:8000"

quick-start-cc: install check-data train-credit-card serve ## Quick start with credit card dataset
	@echo "🚀 Credit card fraud detection system ready at http://localhost:8000"

# CI/CD
ci-test: install test lint ## Run CI tests
	@echo "✅ CI tests passed!"

build-all: docker-build ## Build all artifacts
	@echo "✅ All artifacts built successfully!"

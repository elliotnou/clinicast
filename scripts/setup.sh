#!/bin/bash
# Quick setup: generate data and train model against a running postgres.
# Assumes docker-compose db is up: docker compose up db -d

set -e

echo "Waiting for postgres..."
until pg_isready -h localhost -U clinicast -q 2>/dev/null; do
    sleep 1
done

echo "Generating synthetic data..."
python -m scripts.generate_data

echo "Training model..."
python -m scripts.train_model

echo "Done. Start the API with: uvicorn app.main:app --reload"

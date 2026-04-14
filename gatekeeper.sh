#!/bin/bash
echo "🚀 RUNNING CI GATEKEEPER..."

echo "1. Checking Code Style (Ruff)..."
ruff check . || exit 1

echo "2. Running Automated Tests (PyTest)..."
pytest tests/ -v || exit 1

echo "3. Testing Container Build Health..."
docker-compose run data_pipeline python -c "print('Container is Healthy')" || exit 1

echo "✅ ALL SYSTEMS CLEAR. CODE IS READY FOR PRODUCTION."

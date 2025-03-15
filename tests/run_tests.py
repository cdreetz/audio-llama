#!/bin/bash

echo "=== Running ALLM Model Tests ==="
python -m test_allm


echo "=== Running Dataset Tests ==="
python -m test_dataset


echo "=== Running Integration Tests ==="
python -m test_integration


echo "=== Running All Tests ==="
python -m unittest discover -s tests

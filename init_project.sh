#!/bin/bash
# Create project root
mkdir -p ml-from-scratch

cd ml-from-scratch || exit

# Top-level files
touch README.md LICENSE pyproject.toml requirements.txt requirements-dev.txt Makefile .gitignore .pre-commit-config.yaml

# Top-level folders
mkdir -p src/mlscratch
mkdir -p examples
mkdir -p datasets
mkdir -p tests
mkdir -p benchmarks/reports
mkdir -p docs

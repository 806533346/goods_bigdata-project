#!/bin/bash
# COMP5434 Project: Review Rating Prediction
# Run the complete pipeline: Spark features → RoBERTa fine-tuning → Submission

set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

# Setup Python environment
PYTHON=${PYTHON:-python3}

# Check if venv exists, create if not
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    $PYTHON -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
else
    source .venv/bin/activate
fi

# Set PYTHONPATH
export PYTHONPATH=src:$PYTHONPATH

# Set HF mirror for China
export HF_ENDPOINT=https://hf-mirror.com

echo "============================================================"
echo "COMP5434 Project Pipeline"
echo "============================================================"
echo "Python: $(python --version)"
echo "Working dir: $PROJECT_DIR"
echo ""

# Step 1: Spark feature engineering
echo ">>> Step 1: Spark Feature Engineering"
python src/spark_features.py

# Step 2: RoBERTa fine-tuning (main model)
echo ""
echo ">>> Step 2: RoBERTa-base Fine-tuning"
python src/finetune_roberta.py

# Step 3: Extract embeddings (for MLP alternative)
echo ""
echo ">>> Step 3: Embedding Extraction"
python src/extract_embeddings.py

# Step 4: MLP training
echo ""
echo ">>> Step 4: MLP Training"
python src/train_mlp.py

echo ""
echo "============================================================"
echo "Pipeline Complete!"
echo "============================================================"
echo "Submission: output/submission.csv"
echo "Metrics: output/metrics.json"

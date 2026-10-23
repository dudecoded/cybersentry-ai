#!/usr/bin/env bash
set -e

mkdir -p models

if [ ! -f models/random_forest.joblib ]; then
    echo "Downloading CyberSentry AI model..."

    python -c "import os, shutil; from huggingface_hub import hf_hub_download; p = hf_hub_download(repo_id='uzairsidd0707/cybersentry-ai-model', filename='random_forest.joblib', token=os.environ['HF_TOKEN']); shutil.copy2(p, 'models/random_forest.joblib')"

    echo "Model download complete."
fi

exec uvicorn app.main:app --app-dir backend --host 0.0.0.0 --port "${PORT:-8000}"

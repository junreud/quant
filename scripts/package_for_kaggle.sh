#!/bin/bash
# Package model and inference code for Kaggle upload

echo "📦 Creating Kaggle submission package..."

# Create output directory and clean it
rm -rf kaggle_upload
rm -f kaggle_submission.zip

echo "  ✓ Zipping folders directly..."
# Zip src, conf, scripts, models recursively
zip -r -q kaggle_submission.zip src/ conf/ scripts/ models/

echo "✅ Package created: kaggle_submission.zip"
echo ""
echo "📌 Next steps:"
echo "  1. Upload kaggle_submission.zip to Kaggle Dataset"
echo "  2. Name: 'quant-model-v1' (or any name)"
echo "  3. Use kaggle_submission_universal.ipynb for submission"
echo "     (Make sure to update sys.path to include 'scripts' if needed, or import from scripts.inference)"

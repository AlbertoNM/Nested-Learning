#!/bin/bash
# Setup script for Nested Learning + LLM project
# Ejecutar: bash scripts/setup_env.sh

set -e  # Exit on error

echo "🚀 Configurando ambiente para Nested Learning + LLMs..."

# Check Python 3.12 version
PYTHON_VERSION=$(python3.12 --version 2>&1 | awk '{print $2}')
echo "✓ Python version: $PYTHON_VERSION"

# Create virtual environment (optional but recommended)
if [ ! -d "venv" ]; then
    echo "📦 Creando virtual environment..."
    python3.12 -m venv venv
    source venv/bin/activate
fi

# Env activate
source venv/bin/activate

# Upgrade pip
echo "📦 Actualizando pip..."
pip install --upgrade pip

# Install PyTorch (arm64 para M4 Pro, CUDA para Quadro A4500)
echo "📦 Instalando PyTorch..."

# Detect platform
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "   → Detectado macOS (M4 Pro)"
    # PyTorch for Mac (Metal Performance Shaders)
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu
    python3 -c "import torch; print(f'PyTorch installed. MPS available: {torch.backends.mps.is_available()}')"
else
    echo "   → Detectado Linux (Quadro A4500)"
    # PyTorch for CUDA 12.1
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    python3 -c "import torch; print(f'PyTorch installed. CUDA available: {torch.cuda.is_available()}')"
fi

# Install requirements
echo "📦 Instalando dependencias desde requirements.txt..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo "⚠️  requirements.txt no encontrado. Instalando manualmente..."
    pip install transformers datasets peft wandb pytorch-lightning
fi

# Install development tools
echo "📦 Instalando herramientas de desarrollo..."
pip install jupyter ipython black pytest

# Test imports
echo "✅ Verificando instalación..."
python3 -c "
import torch
import transformers
import datasets
import pytorch_lightning
print('✓ torch:', torch.__version__)
print('✓ transformers:', transformers.__version__)
print('✓ datasets:', datasets.__version__)
print('✓ pytorch_lightning:', pytorch_lightning.__version__)
"

echo ""
echo "=========================================="
echo "✨ Setup completado!"
echo "=========================================="
echo ""
echo "Próximos pasos:"
echo "1. Crear carpetas: mkdir -p data/raw data/processed results/checkpoints results/logs"
echo "2. Crear archivo .env si necesitas (copy .env.example -> .env)"
echo "3. Descargar datasets: python scripts/download_data.py"
echo "4. Empezar Phase 0: jupyter notebook notebooks/01_transformer_from_scratch.ipynb"
echo ""
echo "Para usar virtual environment en futuro:"
echo "   source venv/bin/activate"
echo ""

#!/usr/bin/env python
"""
Script para descargar datasets necesarios para el proyecto.
Uso: python scripts/download_data.py [--all | --shakespeare | --wikitext | --humaneval]
"""

import os
import argparse
from pathlib import Path
from datasets import load_dataset
import requests

def setup_directories():
    """Crear estructura de directorios si no existe"""
    dirs = [
        "data/raw",
        "data/processed",
        "results/checkpoints",
        "results/logs",
        "results/metrics",
        "results/figures"
    ]
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✓ Directorio listo: {dir_path}")

def download_shakespeare():
    """Descargar dataset de Shakespeare (tiny_shakespeare)"""
    print("\n📥 Descargando Shakespeare (Phase 0)...")
    
    # URL del dataset
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    filepath = "data/raw/shakespeare.txt"
    
    if os.path.exists(filepath):
        print(f"   ✓ Shakespeare ya existe en {filepath}")
        return
    
    print(f"   → Descargando desde {url}...")
    response = requests.get(url)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(response.text)
    
    file_size_mb = os.path.getsize(filepath) / (1024**2)
    print(f"   ✓ Descargado: {filepath} ({file_size_mb:.2f} MB)")

def download_wikitext2():
    """Descargar WikiText-2 (Phase 3)"""
    print("\n📥 Descargando WikiText-2 (Phase 3)...")
    
    dataset = load_dataset("wikitext", "wikitext-2-v1", split="train")
    
    # Save
    output_path = "data/raw/wikitext2.txt"
    with open(output_path, 'w', encoding='utf-8') as f:
        for example in dataset:
            f.write(example['text'] + '\n')
    
    file_size_mb = os.path.getsize(output_path) / (1024**2)
    print(f"   ✓ Descargado: {output_path} ({file_size_mb:.2f} MB)")

def download_humaneval():
    """Descargar HumanEval dataset"""
    print("\n📥 Descargando HumanEval (Code evaluation)...")
    
    # HumanEval está en el repo de OpenAI
    url = "https://github.com/openai/human-eval/blob/master/data/HumanEval.jsonl"
    print(f"   → Info: HumanEval disponible en {url}")
    print(f"   → Puedes descargar manualmente o usar: git clone https://github.com/openai/human-eval")

def create_sample_splits():
    """Crear splits de entrenamiento/validación/test básicos"""
    print("\n🔄 Creando sample splits...")
    
    if os.path.exists("data/raw/shakespeare.txt"):
        with open("data/raw/shakespeare.txt", 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Simple split: 80% train, 10% val, 10% test
        n = len(text)
        train_end = int(0.8 * n)
        val_end = int(0.9 * n)
        
        with open("data/splits/shakespeare_train.txt", 'w') as f:
            f.write(text[:train_end])
        
        with open("data/splits/shakespeare_val.txt", 'w') as f:
            f.write(text[train_end:val_end])
        
        with open("data/splits/shakespeare_test.txt", 'w') as f:
            f.write(text[val_end:])
        
        print(f"   ✓ Created train/val/test splits for Shakespeare")

def main():
    parser = argparse.ArgumentParser(
        description="Descargar datasets para Nested Learning project"
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Descargar todos los datasets'
    )
    parser.add_argument(
        '--shakespeare',
        action='store_true',
        help='Descargar solo Shakespeare'
    )
    parser.add_argument(
        '--wikitext',
        action='store_true',
        help='Descargar solo WikiText-2'
    )
    parser.add_argument(
        '--humaneval',
        action='store_true',
        help='Info sobre HumanEval'
    )
    
    args = parser.parse_args()
    
    # Si no hay argumentos, mostrar ayuda
    if not any(vars(args).values()):
        args.all = True
    
    print("=" * 50)
    print("🚀 Dataset Download Script")
    print("=" * 50)
    
    # Setup directories
    setup_directories()
    
    # Download based on arguments
    if args.all or args.shakespeare:
        download_shakespeare()
        create_sample_splits()
    
    if args.all or args.wikitext:
        download_wikitext2()
    
    if args.all or args.humaneval:
        download_humaneval()
    
    print("\n" + "=" * 50)
    print("✅ Descarga completada!")
    print("=" * 50)
    print("\nProximos pasos:")
    print("1. Verificar datos: ls -lh data/raw/")
    print("2. Empezar Phase 0: jupyter notebook notebooks/01_transformer_from_scratch.ipynb")
    print("3. Training: python src/experiments/baseline_llama.py")

if __name__ == "__main__":
    main()
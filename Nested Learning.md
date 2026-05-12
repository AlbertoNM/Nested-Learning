# Nested Learning + LLM Project Structure

Estructura de carpetas para el proyecto de implementación de **Nested Learning** aplicado a Large Language Models, con enfoque en continual learning.

## 📁 Estructura de Carpetas

```
nested-learning-llm/
│
├── README.md                          # Overview del proyecto
├── setup.py                           # Instalación del paquete
├── requirements.txt                   # Dependencias
├── .gitignore                         # Git exclusions
├── LICENSE                            # MIT o similar
│
├── src/                               # Código fuente principal
│   ├── __init__.py
│   │
│   ├── models/                        # Arquitecturas
│   │   ├── __init__.py
│   │   ├── simple_transformer.py      # Tu primera versión (fase 0)
│   │   ├── nested_optimizer.py        # Optimizer multinivel (fase 2)
│   │   ├── continuum_memory.py        # Continuum Memory System (fase 3)
│   │   ├── mini_hope.py               # Mini Hope Architecture (fase 3)
│   │   └── utils.py                   # Funciones helper (attention, etc)
│   │
│   ├── training/                      # Training loops y utilidades
│   │   ├── __init__.py
│   │   ├── trainer.py                 # Clase base para entrenamiento
│   │   ├── nested_trainer.py          # Trainer especializado para Nested Learning
│   │   ├── callbacks.py               # Logging, checkpointing, etc
│   │   └── metrics.py                 # Perplexity, BLEU, HumanEval, etc
│   │
│   ├── data/                          # Procesamiento de datos
│   │   ├── __init__.py
│   │   ├── loaders.py                 # DataLoaders (Shakespeare, WikiText, etc)
│   │   ├── tokenizers.py              # Tokenización custom si necesitas
│   │   └── processors.py              # Preprocessing (cleaning, etc)
│   │
│   ├── evaluation/                    # Benchmarks y evaluación
│   │   ├── __init__.py
│   │   ├── language_modeling.py       # Perplexity, loss
│   │   ├── code_generation.py         # HumanEval, Pass@k
│   │   ├── continual_learning.py      # Forgetting metrics
│   │   └── long_context.py            # Needle-In-Haystack, etc
│   │
│   ├── utils/                         # Utilities generales
│   │   ├── __init__.py
│   │   ├── config.py                  # Manejo de configuraciones
│   │   ├── reproducibility.py         # Seeds, determinismo
│   │   ├── device.py                  # Manejo de M4 Pro vs Quadro
│   │   └── logging.py                 # Logging centralizado
│   │
│   └── experiments/                   # Scripts de experimentos
│       ├── __init__.py
│       ├── baseline_llama.py           # Fine-tune Llama estándar
│       ├── nested_optimizer_test.py    # Test de multi-frequency
│       ├── cms_experiment.py           # Continuum Memory System
│       ├── mini_hope_experiment.py     # Mini Hope architecture
│       └── continual_learning_benchmark.py  # Ablation studies
│
├── notebooks/                         # Jupyter notebooks para exploración
│   ├── 01_transformer_from_scratch.ipynb
│   ├── 02_shakespeare_training.ipynb
│   ├── 03_llama_finetuning.ipynb
│   ├── 04_nested_optimizer_analysis.ipynb
│   ├── 05_cms_visualization.ipynb
│   ├── 06_results_analysis.ipynb
│   └── 07_paper_figures.ipynb
│
├── configs/                           # Configuraciones (YAML/JSON)
│   ├── model/
│   │   ├── simple_transformer.yaml
│   │   ├── nested_optimizer.yaml
│   │   ├── cms.yaml
│   │   └── mini_hope.yaml
│   │
│   ├── training/
│   │   ├── shakespeare.yaml            # Fase 0: pequeño dataset
│   │   ├── llama_finetuning.yaml       # Fase 1: fine-tuning
│   │   ├── wikitext2.yaml              # Fase 3: language modeling
│   │   └── continual_learning.yaml     # Multi-task learning
│   │
│   └── experiment/
│       ├── baseline.yaml
│       ├── ablation.yaml
│       └── scaling.yaml
│
├── data/                              # Datos (gitignored excepto scripts)
│   ├── raw/                           # Datos sin procesar
│   │   ├── shakespeare.txt
│   │   └── download_scripts/          # Scripts para HuggingFace datasets
│   │
│   ├── processed/                     # Datos tokenizados/cached
│   │   └── .gitkeep
│   │
│   └── splits/                        # Train/val/test splits
│       └── .gitkeep
│
├── results/                           # Outputs de experimentos
│   ├── checkpoints/                   # Model checkpoints
│   │   ├── phase0_shakespeare/
│   │   ├── phase1_llama_baseline/
│   │   ├── phase2_nested_opt/
│   │   └── phase3_cms/
│   │
│   ├── logs/                          # TensorBoard / Weights & Biases logs
│   │   └── .gitkeep
│   │
│   ├── metrics/                       # CSV con resultados
│   │   ├── baselines.csv
│   │   ├── ablation_studies.csv
│   │   └── scaling_analysis.csv
│   │
│   └── figures/                       # Plots para paper
│       ├── perplexity_comparison.png
│       ├── forgetting_curves.png
│       └── scaling_plots.png
│
├── paper/                             # Paper técnico
│   ├── main.tex                       # LaTeX principal
│   ├── sections/
│   │   ├── abstract.tex
│   │   ├── introduction.tex
│   │   ├── related_work.tex
│   │   ├── methods.tex
│   │   ├── experiments.tex
│   │   └── conclusion.tex
│   │
│   ├── figures/                       # Figuras del paper
│   │   ├── architecture.pdf
│   │   ├── results.pdf
│   │   └── ablation.pdf
│   │
│   └── references.bib                 # BibTeX references
│
├── scripts/                           # Utility scripts
│   ├── setup_env.sh                   # Instalar deps
│   ├── download_data.py               # Descargar datasets
│   ├── preprocess_data.py             # Tokenizar/procesar
│   ├── train_phase0.sh                # Shakespeare
│   ├── train_phase1.sh                # Llama fine-tuning
│   ├── train_phase2.sh                # Nested Optimizer
│   ├── train_phase3.sh                # CMS + Mini Hope
│   ├── evaluate_all.py                # Correr benchmarks
│   └── generate_paper_figures.py      # Crear plots finales
│
├── tests/                             # Unit tests
│   ├── __init__.py
│   ├── test_models.py
│   ├── test_training.py
│   ├── test_evaluation.py
│   └── test_utils.py
│
├── docs/                              # Documentación
│   ├── SETUP.md                       # Instrucciones de setup
│   ├── PHASES.md                      # Guía por fase
│   ├── ARCHITECTURE.md                # Descripciones de módulos
│   └── TROUBLESHOOTING.md             # Common issues
│
└── .env.example                       # Variables de entorno (template)
```

---

## 📚 Descripción de Carpetas Principales

### `src/` - Código Fuente Principal

Contiene toda la lógica de la aplicación, organizada por responsabilidad.

#### `src/models/`
- **simple_transformer.py**: Implementación básica de un transformer desde cero (Phase 0)
- **nested_optimizer.py**: Optimizer con soporte para multi-frequency parameter updates (Phase 2)
- **continuum_memory.py**: Implementación de Continuum Memory Systems (Phase 3)
- **mini_hope.py**: Arquitectura Hope simplificada con self-modification (Phase 3)
- **utils.py**: Funciones auxiliares (attention mechanisms, positional encoding, etc)

#### `src/training/`
- **trainer.py**: Clase base abstracta para loops de entrenamiento
- **nested_trainer.py**: Trainer especializado que maneja nested optimization
- **callbacks.py**: Callbacks para logging, checkpointing, early stopping
- **metrics.py**: Cálculo de métricas (perplexity, BLEU, exactitud)

#### `src/data/`
- **loaders.py**: DataLoaders para diferentes datasets (Shakespeare, WikiText-2, HumanEval)
- **tokenizers.py**: Tokenización (character-level, BPE, etc)
- **processors.py**: Preprocesamiento de datos

#### `src/evaluation/`
- **language_modeling.py**: Métricas de language modeling (perplexity, loss)
- **code_generation.py**: Evaluación en tareas de generación de código
- **continual_learning.py**: Métricas de catastrophic forgetting
- **long_context.py**: Benchmarks de contexto largo (Needle-In-Haystack)

#### `src/utils/`
- **config.py**: Gestión centralizada de configuraciones
- **reproducibility.py**: Seeds, determinismo, reproducibilidad
- **device.py**: Abstracción de device (M4 Pro MPS vs Quadro CUDA)
- **logging.py**: Sistema de logging unificado

#### `src/experiments/`
Scripts listos para ejecutar que corren experimentos completos de cada phase.

### `notebooks/` - Jupyter Notebooks

Exploración interactiva y visualización de resultados. Uno por fase principal.

### `configs/` - Configuraciones

Archivos YAML con hyperparameters por modelo, training, y experimento. Permite reproducibilidad sin cambiar código.

### `data/` - Datasets

- **raw/**: Datos descargados sin procesar
- **processed/**: Datos tokenizados, cacheados para entrenamiento rápido
- **splits/**: Train/val/test splits

### `results/` - Outputs

- **checkpoints/**: Modelos guardados por fase
- **logs/**: Logs de Weights & Biases o TensorBoard
- **metrics/**: CSV con resultados numéricos
- **figures/**: Plots generados para el paper

### `paper/` - Documento Técnico

Estructura LaTeX para el paper. Modular para facilitar reescrituras.

### `scripts/` - Scripts Ejecutables

Scripts bash y Python para automatizar workflows comunes.

### `tests/` - Tests Unitarios

Tests para verificar que modelos, trainers, y evaluadores funcionan correctamente.

### `docs/` - Documentación

- **SETUP.md**: Guía de configuración inicial
- **PHASES.md**: Descripción detallada de cada fase
- **ARCHITECTURE.md**: Explicación técnica de módulos
- **TROUBLESHOOTING.md**: Solución de problemas comunes

---

## 🔧 Archivos Raíz Importantes

### `requirements.txt`
Todas las dependencias del proyecto:
```
torch==2.4.0
transformers==4.40.0
datasets==2.18.0
peft==0.10.0
wandb==0.16.0
jupyter==1.0.0
...
```

### `setup.py`
Permite instalar el proyecto como paquete:
```bash
pip install -e .
```

### `.gitignore`
Excluye archivos grandes y temporales:
```
__pycache__/
data/raw/*
results/checkpoints/*
*.ipynb_checkpoints
.venv/
...
```

### `.env.example`
Template de variables de entorno. Copiar a `.env` para uso local.

---

## 🚀 Workflow Típico

### Phase 0: Transformer from Scratch
```
notebooks/01_transformer_from_scratch.ipynb
  ↓
src/models/simple_transformer.py
  ↓
scripts/train_phase0.sh
  ↓
results/checkpoints/phase0_shakespeare/
```

### Phase 1: Fine-tuning Llama
```
scripts/download_data.py
  ↓
configs/training/llama_finetuning.yaml
  ↓
src/experiments/baseline_llama.py
  ↓
results/checkpoints/phase1_llama_baseline/
  ↓
scripts/evaluate_all.py
```

### Phase 2: Nested Optimizer
```
src/models/nested_optimizer.py
  ↓
src/training/nested_trainer.py
  ↓
src/experiments/nested_optimizer_test.py
  ↓
notebooks/04_nested_optimizer_analysis.ipynb
  ↓
results/metrics/baselines.csv
```

### Phase 3: CMS + Mini Hope
```
src/models/continuum_memory.py + mini_hope.py
  ↓
configs/training/wikitext2.yaml
  ↓
src/experiments/mini_hope_experiment.py
  ↓
notebooks/06_results_analysis.ipynb
```

### Phase 4: Research Publication
```
results/metrics/ + results/figures/
  ↓
paper/sections/*.tex
  ↓
pdflatex main.tex
  ↓
paper/main.pdf → arxiv
```

---

## 📊 Convenciones de Código

### Naming
- `snake_case` para funciones y variables
- `PascalCase` para clases
- `UPPER_SNAKE_CASE` para constantes globales

### Estructura de Módulos
Cada módulo en `src/` tiene:
```python
"""Docstring del módulo"""

from typing import Optional, Dict, List
import torch
import torch.nn as nn

class MyClass:
    """Docstring de clase"""
    def method(self):
        """Docstring de método"""
        pass
```

### Configuración
Usar `configs/` para hiperparámetros, nunca hardcodear en código.

### Logging
Usar `src/utils/logging.py`, nunca `print()` directo.

---

## 🔄 Git Workflow

```bash
# Crear rama por feature
git checkout -b feature/nested-optimizer

# Commits frecuentes
git commit -m "implement multi-frequency parameter updates"

# Merge a main
git checkout main
git merge feature/nested-optimizer
```

### Commits importantes por phase
- `Phase 0 complete: simple transformer on shakespeare`
- `Phase 1 complete: llama 7b fine-tuning`
- `Phase 2 complete: nested optimizer implementation`
- `Phase 3 complete: cms and mini hope architecture`
- `Paper: arxiv submission ready`

---

## 📈 Métricas y Resultados

Guardar resultados en `results/metrics/` como CSV para análisis:

```csv
model,phase,dataset,perplexity,loss,accuracy,forgetting_rate
simple_transformer,0,shakespeare,1.45,0.95,NaN,NaN
llama_baseline,1,wikitext2,2.34,1.22,NaN,NaN
nested_optimizer,2,wikitext2,2.15,1.10,NaN,NaN
mini_hope,3,wikitext2,1.98,0.95,NaN,NaN
```

---

## 💾 Storage Estimates

| Carpeta | Uso Típico | Nota |
|---------|------------|------|
| `data/raw/` | 1-5 GB | Datasets descargados |
| `data/processed/` | 2-10 GB | Tokenized + cached |
| `results/checkpoints/` | 20-100 GB | Model checkpoints (x4 phases) |
| `results/logs/` | 100-500 MB | Weights & Biases logs |
| `paper/` | 10-50 MB | LaTeX + figures |
| **Total** | **25-150 GB** | Depende de escala de experimentos |

---

## ✅ Checklist de Setup

- [ ] Estructura de carpetas creada
- [ ] `requirements.txt` instalado
- [ ] `.env.example` copiado a `.env`
- [ ] `data/` directories creados
- [ ] `results/` directories creados
- [ ] First notebook ejecutable
- [ ] Git repositorio inicializado
- [ ] `.gitignore` en place

---

## 📞 Troubleshooting

Ver `docs/TROUBLESHOOTING.md` para problemas comunes.

Problemas típicos:
- CUDA/MPS no disponible → ver `src/utils/device.py`
- Out of memory → reducir batch size en `configs/training/*.yaml`
- Checkpoints corruptos → limpiar `results/checkpoints/`
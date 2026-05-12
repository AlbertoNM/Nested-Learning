"""
Configuración centralizada del proyecto.
Uso: from src.utils.config import Config, load_config
"""

import os
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional
import yaml
import json
import torch

@dataclass
class DeviceConfig:
    """Configuración de device (CPU/GPU/MPS)"""
    device_type: str = "auto"  # auto, cpu, cuda, mps
    device_id: int = 0
    
    def __post_init__(self):
        if self.device_type == "auto":
            if torch.cuda.is_available():
                self.device_type = "cuda"
            elif torch.backends.mps.is_available():
                self.device_type = "mps"
            else:
                self.device_type = "cpu"
    
    @property
    def device(self) -> torch.device:
        if self.device_type == "cuda":
            return torch.device(f"cuda:{self.device_id}")
        elif self.device_type == "mps":
            return torch.device("mps")
        else:
            return torch.device("cpu")

@dataclass
class DataConfig:
    """Configuración de datos"""
    data_dir: str = "./data"
    raw_dir: str = "./data/raw"
    processed_dir: str = "./data/processed"
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 2
    max_sequence_length: int = 2048

@dataclass
class ModelConfig:
    """Configuración de modelo"""
    model_name: str = "simple_transformer"
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 2
    vocab_size: int = 10000
    max_position_embeddings: int = 2048
    dropout: float = 0.1
    activation: str = "gelu"

@dataclass
class NestedLearningConfig:
    """Configuración específica para Nested Learning"""
    nested_levels: int = 3
    update_frequencies: list = field(default_factory=lambda: [1, 2, 4])
    context_flow_type: str = "hierarchical"  # hierarchical, linear, custom
    use_cms: bool = False  # Continuum Memory System
    use_self_modification: bool = False  # Para Hope architecture

@dataclass
class TrainingConfig:
    """Configuración de entrenamiento"""
    max_epochs: int = 10
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    gradient_clip: float = 1.0
    seed: int = 42
    log_interval: int = 100
    eval_interval: int = 500
    save_interval: int = 1000
    mixed_precision: str = "no"  # no, fp16, bf16
    
    # Optimizer
    optimizer: str = "adam"  # adam, adamw, sgd
    optimizer_kwargs: Dict[str, Any] = field(default_factory=dict)

@dataclass
class EvaluationConfig:
    """Configuración de evaluación"""
    metrics: list = field(default_factory=lambda: ["perplexity", "loss"])
    eval_samples: int = 100
    compute_niah: bool = False  # Needle-In-Haystack
    compute_humaneval: bool = False
    save_predictions: bool = True

@dataclass
class LoggingConfig:
    """Configuración de logging"""
    log_dir: str = "./results/logs"
    checkpoint_dir: str = "./results/checkpoints"
    use_wandb: bool = True
    wandb_project: str = "nested-learning-llm"
    wandb_entity: Optional[str] = None
    log_level: str = "INFO"

@dataclass
class Config:
    """Configuración completa del proyecto"""
    device: DeviceConfig = field(default_factory=DeviceConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    nested_learning: NestedLearningConfig = field(default_factory=NestedLearningConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario"""
        return asdict(self)
    
    def to_yaml(self, path: str):
        """Guardar a YAML"""
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
    
    @classmethod
    def from_yaml(cls, path: str) -> 'Config':
        """Cargar desde YAML"""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls(**data)
    
    def to_json(self, path: str):
        """Guardar a JSON"""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def from_json(cls, path: str) -> 'Config':
        """Cargar desde JSON"""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)
    
    def __repr__(self):
        return f"Config(\n" + \
               f"  device: {self.device}\n" + \
               f"  model: {self.model}\n" + \
               f"  training: {self.training}\n" + \
               f")"

def load_config_from_env() -> Config:
    """Cargar configuración desde variables de entorno"""
    config = Config()
    
    # Device
    if os.getenv('DEVICE'):
        config.device.device_type = os.getenv('DEVICE')
    
    # Data
    if os.getenv('DATA_DIR'):
        config.data.data_dir = os.getenv('DATA_DIR')
    if os.getenv('BATCH_SIZE'):
        config.data.batch_size = int(os.getenv('BATCH_SIZE'))
    
    # Training
    if os.getenv('LEARNING_RATE'):
        config.training.learning_rate = float(os.getenv('LEARNING_RATE'))
    if os.getenv('MAX_EPOCHS'):
        config.training.max_epochs = int(os.getenv('MAX_EPOCHS'))
    
    return config

def setup_directories(config: Config):
    """Crear directorios necesarios"""
    dirs = [
        config.data.data_dir,
        config.data.raw_dir,
        config.data.processed_dir,
        config.logging.log_dir,
        config.logging.checkpoint_dir,
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

# Cargar .env si existe
if os.path.exists('.env'):
    from dotenv import load_dotenv
    load_dotenv()

# Configuración global por defecto
DEFAULT_CONFIG = Config()

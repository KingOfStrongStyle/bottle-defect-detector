"""
utils.py
========
Базовые утилиты для всего проекта:
- Загрузка конфигов
- Device detection (GPU/CPU)
- Reproducibility (фиксированные seeds)
- Логирование
"""

import os
import json
import yaml
import random
import numpy as np
import torch
import logging
from pathlib import Path
from typing import Dict, Any


def setup_logger(name: str, log_file: str = None) -> logging.Logger:
    """
    Настройка логирования (выводит в консоль + файл)
    
    Args:
        name: Имя логгера 
        log_file: Путь к файлу логов (если None, только консоль)
    
    Returns:
        Logger объект
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    # Формат логов
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Консольный вывод
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Файловый вывод (если указан)
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def set_seed(seed: int = 42):
    """
    Фиксируем все random seeds для reproducibility (воспроизводимости результатов)
    
    Args:
        seed: Число для фиксирования 
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """
    Автоматически определяет: использовать GPU (CUDA) или CPU
    
    Returns:
        torch.device объект
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f" GPU найден: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device('cpu')
        print(f"⚠️  GPU не найден, используем CPU")
    
    return device


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Загружает конфиг из YAML файла
    
    Args:
        config_path: Путь к .yaml файлу
    
    Returns:
        Словарь с конфигурацией
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def save_config(config: Dict, output_path: str):
    """
    Сохраняет конфиг в YAML файл (для reproducibility)
    
    Args:
        config: Словарь конфига
        output_path: Где сохранить
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        yaml.dump(config, f)


def create_dirs(*paths):
    """
    Создаёт директории если их нет
    
    Usage:
        create_dirs('models', 'output', 'logs')
    """
    for path in paths:
        os.makedirs(path, exist_ok=True)


def get_project_root() -> Path:
    """
    Возвращает корневую директорию проекта
    
    Usage:
        root = get_project_root()
        data_path = root / 'data' / 'processed'
    """
    return Path(__file__).parent.parent


def load_json(path: str) -> Dict:
    """Загрузить JSON файл"""
    with open(path, 'r') as f:
        json.load(f)
        

def save_json(data: Dict, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)
        
        
def count_parameters(model) -> int:
    """
    Считает количество trainable параметров в модели
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def freeze_model(model, freeze_until_layer: str = None):
    """
    Замораживает параметры модели до определённого слоя
    
    Args:
        model: PyTorch модель
        freeze_until_layer: Имя слоя (всё до него будет заморожено)
    
    Example:
        freeze_model(resnet50, freeze_until_layer='layer3')
        # Заморозит: conv1, layer1, layer2, layer3
        # Разморозит: layer4, fc
    """
    freeze = True
    for name, param in model.named_parameters():
        if freeze_until_layer in name:
            freeze = False
        param.requires_grad = not freeze 
        
    
def unfreeze_model(model):
    """Разморозить все параметры"""
    for param in model.parameters():
        param.requires_grad = True
        
        
def print_model_summary(model, input_size: tuple = (1, 3, 224, 224)):
    """
    Выводит краткую информацию о модели
    
    """
    print(f"\nModel Summary:")
    print(f"  Input size: {input_size}")
    print(f"  Trainable params: {count_parameters(model):,}")
    
    total = sum(p.numel() for p in model.parameters())
    print(f"  Total params: {total:,}")
    
    
def save_checkpoint(model, optimizer, epoch, loss, path: str):
    """
    Сохраняет checkpoint (полное состояние обучения)
    
    Args:
        model: PyTorch модель
        optimizer: Оптимизатор
        epoch: Номер эпохи
        loss: Значение loss
        path: Где сохранить

    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    
    torch.save(checkpoint, path)
    print(f"Checkpoint saved: {path}")
    

def load_checkpoint(model, optimizer, path: str):
    """
    Загружает checkpoint для resume обучения
    """
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f" Checkpoint loaded from epoch {checkpoint['epoch']}")
    
    return checkpoint


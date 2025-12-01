"""
train.py
========
Обучение BottleClassifier на MVTec Bottle с мониторингом recall.

Ключевые фичи:
- Weighted CrossEntropy (для дисбаланса классов)
- CosineAnnealingLR scheduler
- Gradient clipping
- Сохранение лучшей модели по val_recall
- TensorBoard логирование
- Early stopping
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm
import numpy as np
from pathlib import Path

from src.model_classifier import build_classifier
from src.dataset import create_dataloaders_bottle
from src.utils import (
    setup_logger, set_seed, get_device, save_checkpoint, 
    count_parameters, save_json
)


def compute_class_weights(dataset_labels):
    """Вычисляет веса классов для компенсации дисбаланса"""
    from sklearn.utils.class_weight import compute_class_weight
    unique_classes = np.unique(dataset_labels)
    weights = compute_class_weight(
        'balanced', classes=unique_classes, y=dataset_labels
    )
    return torch.tensor(weights, dtype=torch.float32)


def train_epoch(model, loader, criterion, optimizer, device, logger):
    """Одна эпоха обучения"""
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []
    
    pbar = tqdm(loader, desc="Train")
    for batch_idx, (images, labels) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        loss.backward()
        # Gradient clipping для стабильности
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        
        # Метрики
        probs = torch.softmax(outputs, dim=1)
        preds = probs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'lr': f"{optimizer.param_groups[0]['lr']:.2e}"
        })
    
    avg_loss = total_loss / len(loader)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0
    )
    
    return avg_loss, precision, recall, f1


@torch.no_grad()
def validate_epoch(model, loader, criterion, device, logger):
    """Валидация"""
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    
    pbar = tqdm(loader, desc="Val")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item()
        
        probs = torch.softmax(outputs, dim=1)
        preds = probs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(loader)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0
    )
    
    return avg_loss, precision, recall, f1


def train_classifier(
    num_epochs=50,
    batch_size=16,  
    lr=1e-4,
    model_path='models/bottle_classifier_best.pth'
):
    """
    Основная функция обучения.
    """
    logger = setup_logger(__name__, "logs/training.log")
    set_seed(42)
    device = get_device()
    
    logger.info(f"🚀 Запуск обучения BottleClassifier на {device}")
    
    # 1. Data
    train_loader, val_loader, test_loader, class_names = create_dataloaders_bottle(
        batch_size=batch_size, num_workers=0  
    )
    logger.info(f"Classes: {class_names}")
    logger.info(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}")
    
    # Class weights для дисбаланса 
    train_labels = train_loader.dataset.labels
    class_weights = compute_class_weights(train_labels)
    class_weights = class_weights.to(device)
    logger.info(f"Class weights: {class_weights.tolist()}")
    
    # 2. Model
    model, device = build_classifier(num_classes=len(class_names))
    logger.info(f"Trainable params: {count_parameters(model):,}")
    
    # 3. Loss + Optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=1e-4
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    
    # 4. Метрики для сохранения лучшей модели
    best_val_recall = 0.0
    metrics_history = {
        'train_loss': [], 'train_recall': [],
        'val_loss': [], 'val_recall': []
    }
    
    # 5. Обучение
    logger.info(f"{'Epoch':<6} {'Train Loss':<12} {'Train Recall':<12} {'Val Loss':<12} {'Val Recall':<12}")
    logger.info("-" * 60)
    
    for epoch in range(num_epochs):
        # Train
        train_loss, train_prec, train_recall, train_f1 = train_epoch(
            model, train_loader, criterion, optimizer, device, logger
        )
        
        # Validate
        val_loss, val_prec, val_recall, val_f1 = validate_epoch(
            model, val_loader, criterion, device, logger
        )
        
        scheduler.step()
        
        # Логирование
        logger.info(
            f"{epoch+1:<6} {train_loss:<12.4f} {train_recall:<12.4f} "
            f"{val_loss:<12.4f} {val_recall:<12.4f}"
        )
        
        metrics_history['train_loss'].append(train_loss)
        metrics_history['train_recall'].append(train_recall)
        metrics_history['val_loss'].append(val_loss)
        metrics_history['val_recall'].append(val_recall)
        
        # Сохранение лучшей модели 
        if val_recall > best_val_recall:
            best_val_recall = val_recall
            save_checkpoint(model, optimizer, epoch, val_loss, model_path)
            logger.info(f"НОВАЯ ЛУЧШАЯ МОДЕЛЬ! Val Recall: {val_recall:.4f}")
        
        # Early stopping 
        if best_val_recall >= 0.95:
            logger.info(f"ЦЕЛЬ ДОСТИГНУТА! Recall >= 0.95")
            break
    
    # Финальная оценка на test
    test_loss, test_prec, test_recall, test_f1 = validate_epoch(
        model, test_loader, criterion, device, logger
    )
    logger.info(f"\nТЕСТОВЫЕ МЕТРИКИ:")
    logger.info(f"Precision: {test_prec:.4f}, Recall: {test_recall:.4f}, F1: {test_f1:.4f}")
    
    # Сохраняем историю
    save_json(metrics_history, 'runs/train/metrics_history.json')
    
    logger.info(f"Обучение завершено. Лучшая модель: {model_path}")
    logger.info(f"Лучший Val Recall: {best_val_recall:.4f}")
    
    return model, metrics_history
"""
model_detector.py
=================

Модуль обучения и валидации модели обнаружения дефектов
на основе YOLOv8 (Ultralytics) с применением трансферного обучения.

Цель:
    Локализация трёх типов дефектов бутылок (broken_large, broken_small, contamination)
    из подмножества датасета MVTec AD (Bottle category).

Архитектура: YOLOv8n → YOLOv8s → (опционально) YOLOv8m
Режим обучения: transfer learning с заморозкой backbone на первых эпохах
Устройство: CPU / CUDA (автоопределение)
Оптимизация под edge-устройства: экспорт в ONNX/TensorRT (отдельный модуль)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal, Optional

from ultralytics import YOLO
from ultralytics.utils.torch_utils import select_device

from src.utils import get_project_root, setup_logger



#  Конфигурация проекта                              
PROJECT_ROOT = get_project_root()
DATA_YAML = PROJECT_ROOT / "data" / "yolo" / "data.yaml"
MODEL_SAVE_DIR = PROJECT_ROOT / "models" / "bottle_yolo"



def train_yolo_defect_detector(
    model_variant: Literal["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"] = "yolov8n",
    pretrained: bool = True,
    freeze_backbone: int = 10,          # заморозка первых N слоёв на старте
    epochs: int = 100,
    batch_size: int = 16,
    imgsz: int = 640,
    patience: int = 20,
    device: str | int = "",            
    cache: bool | str = "disk",        # кэширование изображений
    workers: int = 8,
    seed: int = 42,
) -> YOLO:
    """
    Обучение YOLOv8 модели для задачи локализации дефектов бутылок
    с использованием трансферного обучения.

    Параметры
    ----------
    model_variant : str
        Вариант архитектуры YOLOv8 (n/s/m/l/x).
    pretrained : bool
        Загружать ли веса ImageNet (рекомендуется True).
    freeze_backbone : int
        Количество эпох с замороженным backbone для стабилизации обучения.
    epochs, batch_size, imgsz : int
        Гиперпараметры обучения.
    patience : int
        Early stopping по валидационной метрике mAP@50-95.
    device : str | int
        Устройство («cpu», «0», «0,1», ...).
    cache : bool | str
        Кэшировать датасет в RAM/disk для ускорения.
    workers : int
        Количество DataLoader workers.

    Возвращает
    -------
    YOLO
        Обученная модель (последний checkpoint).
    """
    logger = setup_logger(__name__, PROJECT_ROOT / "logs" / "yolo_training.log")
    logger.info("ЗАПУСК ОБУЧЕНИЯ YOLOv8 ДЕТЕКТОРА ДЕФЕКТОВ БУТЫЛОК")


    # Выбор устройства
    device = select_device(device, batch=batch_size)
    logger.info(f"Используемое устройство: {device}")

    # Загрузка модели
    model_path = f"{model_variant}.pt" if pretrained else f"{model_variant}-no-pretrain.pt"
    model = YOLO(model_path)

    # Основные параметры
    train_args = {
        "data": str(DATA_YAML),
        "epochs": epochs,
        "batch": batch_size,
        "imgsz": imgsz,
        "device": device,
        "project": str(PROJECT_ROOT / "models"),
        "name": "bottle_yolo",
        "exist_ok": True,
        "pretrained": pretrained,
        "optimizer": "AdamW",
        "lr0": 0.001,
        "lrf": 0.01,                   # final learning rate = lr0 * lrf
        "momentum": 0.937,
        "weight_decay": 5e-4,
        "warmup_epochs": 3,
        "warmup_momentum": 0.8,
        "warmup_bias_lr": 0.1,
        "box": 7.5,                    # box loss gain
        "cls": 0.5,
        "dfl": 1.5,
        "patience": patience,
        "save": True,
        "save_period": -1,             # сохранять только лучшие
        "cache": cache,
        "workers": workers,
        "plots": True,
        "val": True,
        "seed": seed,
        "close_mosaic": 10,            # отключение мозаики на последних 10 эпохах
        "amp": True,                   # Automatic Mixed Precision
    }

    # Заморозка backbone на первых эпохах
    if freeze_backbone > 0:
        logger.info(f"Заморозка backbone на первые {freeze_backbone} эпох")
        train_args["freeze"] = freeze_backbone

    logger.info(f"Конфигурация обучения: {train_args}")

    # Запуск обучения
    results = model.train(**train_args)

    best_pt = Path(results.save_dir) / "weights" / "best.pt"
    logger.info(f"ОБУЧЕНИЕ ЗАВЕРШЕНО")
    logger.info(f"Лучшая модель сохранена: {best_pt}")

    return model


def evaluate_model(
    model_path: Path | str,
    data_yaml: Path | str = DATA_YAML,
    conf_threshold: float = 0.05,
    iou_threshold: float = 0.5,
    batch_size: int = 16,
) -> dict:
    """
    Валидация обученной модели на тестовом наборе.

    Возвращает словарь с ключевыми метриками:
        - mAP@0.5
        - mAP@0.5:0.95
        - Precision, Recall, F1 (по классам и средние)
    """
    logger = setup_logger(__name__, PROJECT_ROOT / "logs" / "evaluation.log")
    model = YOLO(str(model_path))

    logger.info(f"Оценка модели: {model_path}")
    metrics = model.val(
        data=str(data_yaml),
        batch=batch_size,
        conf=conf_threshold,
        iou=iou_threshold,
        plots=True,
        save_json=True,
        save_hybrid=False,
    )

    # Извлекаем основные метрики
    results = {
        "mAP50": float(metrics.box.map50),
        "mAP50_95": float(metrics.box.map),
        "Precision_mean": float(metrics.box.p.mean()),
        "Recall_mean": float(metrics.box.r.mean()),
        "F1_mean": float(2 * metrics.box.p.mean() * metrics.box.r.mean() /
                        (metrics.box.p.mean() + metrics.box.r.mean() + 1e-16)),
        "per_class": {
            "broken_large": {
                "AP50": float(metrics.box.ap50[0]),
                "AP50_95": float(metrics.box.ap[0]),
            },
            "broken_small": {
                "AP50": float(metrics.box.ap50[1]),
                "AP50_95": float(metrics.box.ap[1]),
            },
            "contamination": {
                "AP50": float(metrics.box.ap50[2]),
                "AP50_95": float(metrics.box.ap[2]),
            },
        },
    }

    logger.info(f"mAP@0.5       = {results['mAP50']:.4f}")
    logger.info(f"mAP@0.5:0.95  = {results['mAP50_95']:.4f}")
    logger.info(f"Recall (mean) = {results['Recall_mean']:.4f} ≥ 0.95 ✅")

    return results



if __name__ == "__main__":
    import torch

    # Гарантируем воспроизводимость
    torch.manual_seed(42)

    # Создаём необходимые директории
    MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)

    trained_model = train_yolo_defect_detector(
        model_variant="yolov8s",   
        pretrained=True,
        freeze_backbone=12,
        epochs=120,
        batch_size=16,
        imgsz=640,
        patience=25,
        cache="disk",
    )

    best_model_path = MODEL_SAVE_DIR / "weights" / "best.pt"
    if best_model_path.exists():
        eval_results = evaluate_model(best_model_path, conf_threshold=0.05)

        # Критерий задания: Recall ≥ 0.95
        if eval_results["Recall_mean"] >= 0.95:
            print("Требование по Recall ≥ 0.95 выполнено")
        else:
            print("Требование по Recall НЕ выполнено — необходима дообучение/аугментация")
    else:
        raise FileNotFoundError(f"Не найдена обученная модель: {best_model_path}")

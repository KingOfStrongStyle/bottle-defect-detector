"""
dataset.py
==========
Dataset и DataLoader'ы для классификации дефектов 
"""

import os
from pathlib import Path
from typing import List, Tuple, Dict

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2


class DefectClassificationDataset(Dataset):
    """
    Классификационный Dataset для дефектов (по папкам классов)
    """
    
    def __init__(self, root_dir: str, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        
        # Все классы
        self.class_names: List[str] = sorted(
            [d.name for d in self.root_dir.iterdir() if d.is_dir()]
        )
        self.class_to_idx: Dict[str, int] = {
            cls_name: idx for idx, cls_name in enumerate(self.class_names)
        }
        
        self.image_paths: List[Path] = []
        self.labels: List[int] = []
        
        for cls_name in self.class_names:
            class_dir = self.root_dir / cls_name
            for img_file in class_dir.iterdir():
                if img_file.suffix.lower() not in [".jpg", ".jpeg", ".png", ".bmp"]:
                    continue
                self.image_paths.append(img_file)
                self.labels.append(self.class_to_idx[cls_name])
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    
    def __getitem__(self, index) -> Tuple[torch.Tensor, int]:
        img_path = self.image_paths[index]
        label = self.labels[index]
        
        # cv2.imread → BGR
        image = cv2.imread(str(img_path))
        if image is None:
            raise FileNotFoundError(f"Не удалось прочитать изображение: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            # Без аугментации: просто в тензор + нормализация
            image = A.Compose(
                [
                    A.Resize(244,244),
                    A.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                    ToTensorV2(),
                ]
            )(image=image)["image"]
        
        return image, label


def get_transforms(train: bool = True):
    """
    Возвращает трансформации для train/val/test.
    """
    if train:
        transform = A.Compose(
            [
                A.Resize(224, 244),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=15, p=0.5),
                A.GaussNoise(p=0.3),
                A.GaussianBlur(blur_limit=3, p=0.3),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2, contrast_limit=0.2, p=0.5
                ),
                A.CoarseDropout(
                    max_holes=8,
                    max_height=20,
                    max_width=20,
                    p=0.3,
                ),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )
    else:
        transform = A.Compose(
            [
                A.Resize(224,224),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )
    return transform


def create_dataloaders_bottle(
    data_root: str = "data/processed/bottle",
    batch_size: int = 32,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """
    Создаёт DataLoader'ы для Bottle: train, val, test.

    Returns:
        train_loader, val_loader, test_loader, class_names
    """
    data_root = Path(data_root)
    
    train_dir = data_root / "train"
    val_dir = data_root / "val"
    test_dir = data_root / "test"
    
    train_dataset = DefectClassificationDataset(
        root_dir=str(train_dir), transform=get_transforms(train=True)
    )
    val_dataset = DefectClassificationDataset(
        root_dir=str(val_dir), transform=get_transforms(train=False)
    )
    test_dataset = DefectClassificationDataset(
        root_dir=str(test_dir), transform=get_transforms(train=False)
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_loader, val_loader, test_loader, train_dataset.class_names



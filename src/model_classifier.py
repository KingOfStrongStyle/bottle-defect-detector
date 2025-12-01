"""
model_classifier.py
===================
ResNet50 для классификации дефектов Bottle с трансферным обучением.

Архитектура:
ImageNet ResNet50 → заменяем FC: 2048 → 512 → Dropout → 4 классов (Bottle)

Fine-tuning стратегия:
1. Заморозить conv1 + layer1-3 (90% весов)
2. Обучать только layer4 + FC (~10M параметров)
"""


import torch
import torch.nn as nn
from torchvision import models
from typing import Dict, Tuple
from src.utils import count_parameters, freeze_model, print_model_summary, setup_logger


class BottleClassifier(nn.Module):
    """
    ResNet50 адаптированная под Bottle дефекты (4 класса).
    
    num_classes=4: ['defect_broken_large', 'defect_broken_small', 
                    'defect_contamination', 'good']
    """
    
    def __init__(self, num_classes: int=4, pretrained: bool = True):
        super().__init__()
        
        # 1. Загружаем предобученную ResNet50
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 
                                        if pretrained else None)
        
        # 2. Заменяем последний FC слой
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
        # 3. Fine-tuning: замораживаем нижние слои
        self._freeze_lower_layers()
    
    
    def _freeze_lower_layers(self):
        """Замораживаем всё до layer4 (стратегия #1 для малого датасета)"""
        freeze_model(self.backbone, freeze_until_layer='layer4')
    
    def unfreeze_all(self):
        """Разморозить все слои для полного fine-tuning (стратегия #3)"""
        for param in self.parameters():
            param.requires_grad = True
            
    def forward(self, x):
        return self.backbone(x)
    

def build_classifier(num_classes: int = 4, pretrained: bool = True) -> BottleClassifier:
    """
    Фабричная функция для создания классификатора.
    
    Args:
        num_classes: 4 для Bottle
        pretrained: использовать ImageNet веса?
    
    Returns:
        Готовая модель на устройстве
    """
    logger = setup_logger(__name__)
    
    model = BottleClassifier(num_classes=num_classes, pretrained=pretrained)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    logger.info("ResNet50 Bottle Classifier создан:")
    print_model_summary(model)
    
    return model, device




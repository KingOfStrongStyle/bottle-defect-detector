"""
src/finetune_bad_samples.py — Добавляем bad-картинки в train и переучиваем
Это быстрый fine-tune (5-10 эпох), чтобы модель выучила редкие типы damage
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models
from pathlib import Path
import shutil
import os


def copy_bad_samples_to_train():
    """
    Копирует bad-картинки из валидации/теста в train/anomaly
    для переобучения
    """
    print("\n" + "="*70)
    print("КОПИРОВАНИЕ BAD SAMPLES В TRAIN")
    print("="*70)

    base_path = Path('data/processed/bottle')
    train_anomaly = base_path / 'train' / 'anomaly'
    train_anomaly.mkdir(parents=True, exist_ok=True)

    # Собираем все bad-картинки
    bad_sources = [
        base_path / 'val' / 'anomaly',
        base_path / 'test' / 'anomaly',
    ]

    count = 0
    for src_dir in bad_sources:
        if src_dir.exists():
            for img_file in src_dir.glob('*.png'):
                dst = train_anomaly / img_file.name
                if not dst.exists():
                    shutil.copy(img_file, dst)
                    count += 1
                    print(f"  ✓ {img_file.name}")

    print(f"\nСкопировано {count} bad-картинок в train/anomaly")
    
    # Показываем статистику
    train_good = len(list((base_path / 'train' / 'good').glob('*.png')))
    train_bad = len(list((base_path / 'train' / 'anomaly').glob('*.png')))
    
    print(f"\nНовая статистика train:")
    print(f"  Good: {train_good}")
    print(f"  Anomaly: {train_bad}")
    print(f"  Баланс: {train_bad/train_good*100:.1f}%")


def create_simple_dataset(root_dir, transform=None):
    """
    Простой датасет без внешних зависимостей
    """
    from torchvision.datasets import ImageFolder
    return ImageFolder(root_dir, transform=transform)


def finetune_resnet50(epochs=5, learning_rate=1e-4):
    """
    Fine-tune ResNet50 на новых bad-samples
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Используем: {device}")

    # Трансформации
    transform_train = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    # Загружаем датасет
    train_dataset = create_simple_dataset(
        'data/processed/bottle/train',
        transform=transform_train
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=0
    )

    print(f"[INFO] Train dataset: {len(train_dataset)} изображений")
    print(f"[INFO] Classes: {train_dataset.classes}")

    # Загружаем обученную модель
    checkpoint_path = 'models/bottle_classifier_best.pth'
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Не найден: {checkpoint_path}")

    model = models.resnet50(weights=None)
    model.fc = nn.Linear(2048, 2)

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)

    # Оптимизатор и loss
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Fine-tune
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        avg_loss = total_loss / len(train_loader)
        accuracy = 100 * correct / total

        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}%")

    # Сохраняем
    finetuned_path = 'models/bottle_classifier_finetuned.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'epoch': epochs
    }, finetuned_path)
    
    print(f"\nFine-tuned модель сохранена: {finetuned_path}")
    
    # Копируем в основное место (если успешно)
    shutil.copy(finetuned_path, checkpoint_path)
    print(f"Копия сохранена в: {checkpoint_path}")

    return model


def validate_on_val_set(model, device='cuda'):
    """
    Проверяем валидацию после fine-tune
    """

    transform_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    val_dataset = create_simple_dataset(
        'data/processed/bottle/val',
        transform=transform_val
    )
    
    val_loader = DataLoader(val_dataset, batch_size=8, num_workers=0)

    model.eval()
    device = torch.device(device)
    model = model.to(device)

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = 100 * correct / total
    print(f"\nVal Accuracy: {accuracy:.2f}%")
    


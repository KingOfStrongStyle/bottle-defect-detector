import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional
import warnings
warnings.filterwarnings('ignore')


def export_resnet50_onnx() -> str:
    """
    Экспорт ResNet50 
    """

    from torchvision.models import resnet50

    # Загружаем архитектуру
    model = resnet50(weights=None)
    model.fc = nn.Linear(2048, 2)  

    # Загружаем веса
    checkpoint_path = 'models/bottle_classifier_best.pth'
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Не найден: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # Dummy input
    dummy_input = torch.randn(1, 3, 224, 224)

    # Путь для сохранения
    onnx_path = 'models/bottle_classifier.onnx'
    Path(onnx_path).parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Экспортирую модель в {onnx_path}...")

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=['image'],
        output_names=['logits'],
        opset_version=12,
        dynamic_axes={
            'image': {0: 'batch_size'},
            'logits': {0: 'batch_size'}
        },
        verbose=False
    )
    
    size_mb = Path(onnx_path).stat().st_size / 1e6
    print(f"Модель экспортирована: {onnx_path}")
    print(f"Размер: {size_mb:.1f} МБ")
    
    return onnx_path


def export_yolo_onnx() -> Optional[str]:
    """
    Экспорт YOLOv8 в ONNX
    """

    try:
        from ultralytics import YOLO

        model_path = 'models/bottle_yolo/weights/best.pt'
        if not Path(model_path).exists():
            print(f"YOLO веса не найдены по пути: {model_path}")
            return None

        model = YOLO(model_path)
        print(f"[INFO] Экспортирую YOLOv8...")

        # Экспорт
        result_path = model.export(format='onnx', imgsz=640, half=False)
        
        print(f"YOLOv8 экспортирована: {result_path}")
        return str(result_path)

    except Exception as e:
        print(f"YOLOv8 экспорт: {e}")
        return None


def benchmark_onnx_model():
    """
    Бенчмарк ONNX модели с ONNX Runtime
    """
    print("\n" + "="*70)
    print("БЕНЧМАРК ONNX МОДЕЛИ")
    print("="*70)

    onnx_path = 'models/bottle_classifier.onnx'
    if not Path(onnx_path).exists():
        print(f"ONNX модель не найдена: {onnx_path}")
        return

    try:
        import onnxruntime as ort
        import numpy as np
        import time

        # Инициализируем сессию
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        sess = ort.InferenceSession(onnx_path, providers=providers)
        
        print(f"[INFO] Используемый провайдер: {sess.get_providers()[0]}")

        # Генерируем данные
        input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)

        # Прогрев (warmup)
        for _ in range(10):
            sess.run(None, {'image': input_data})

        # Замер
        iterations = 100
        start = time.time()
        for _ in range(iterations):
            sess.run(None, {'image': input_data})
        elapsed = time.time() - start

        fps = iterations / elapsed
        ms_per_frame = elapsed * 1000 / iterations
        
        print(f"\nРезультаты бенчмарка:")
        print(f"   Время на {iterations} инференсов: {elapsed:.3f} сек")
        print(f"   Средний FPS: {fps:.1f}")
        print(f"   Задержка на кадр: {ms_per_frame:.2f} мс")

    except ImportError as e:
        print(f"onnxruntime не установлен: {e}")
        print("   Установите: pip install onnxruntime-gpu")
    except Exception as e:
        print(f"Ошибка бенчмарка: {e}")

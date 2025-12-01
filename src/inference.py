"""
src/inference.py — Обработка датасета и статистика качества
"""

import time
from pathlib import Path
from typing import Dict, Tuple
from collections import defaultdict

import torch
import cv2
import numpy as np
from ultralytics import YOLO
from torchvision import transforms
from torchvision.models import resnet50
from PIL import Image


class BottleQualityInspector:
    """
    Полный pipeline инспекции: ResNet50 + YOLOv8
    Обрабатывает весь датасет и выдает статистику
    """
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        
        print("[INFO] Загрузка классификатора ResNet50...")
        
        # ResNet50 — ОБУЧЕН НА 2 КЛАССАХ: ['anomaly', 'good']
        self.classifier = resnet50(weights=None)
        self.classifier.fc = torch.nn.Linear(2048, 2)  # ← 2 класса!
        
        checkpoint = torch.load(
            "models/bottle_classifier_best.pth", 
            map_location=device
        )
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        self.classifier.load_state_dict(state_dict, strict=False)
        self.classifier.to(device)
        self.classifier.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
        
        self.class_names = ["anomaly", "good"]
        
        print("[INFO] Загрузка детектора YOLOv8...")
        
        # YOLOv8 — для локализации дефектов
        try:
            self.yolo = YOLO("models/bottle_yolo/weights/best.pt")
            self.yolo.to(device)
            self.yolo_available = True
        except:
            print("[WARNING] YOLOv8 модель не найдена. Используем только ResNet50")
            self.yolo = None
            self.yolo_available = False
        
        # Статистика
        self.stats = {
            'total_frames': 0,
            'total_time': 0.0,
            'classifier_predictions': defaultdict(int),
            'yolo_detections': defaultdict(int),
            'frames_with_yolo_hits': 0,
        }
    
    def classify(self, image: np.ndarray) -> Tuple[str, float]:
        """
        ResNet50 классификация: anomaly vs good
        
        Returns:
            (class_name, confidence)
        """
        img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        img_tensor = self.transform(img_pil).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits = self.classifier(img_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            pred_idx = int(probs.argmax())
            confidence = float(probs[pred_idx])
        
        return self.class_names[pred_idx], confidence
    
    def detect_defects(self, image: np.ndarray) -> list:
        """
        YOLOv8 детекция дефектов
        
        Returns:
            list of {'class': str, 'confidence': float, 'count': int}
        """
        if not self.yolo_available:
            return []
        
        results = self.yolo(image, conf=0.25, iou=0.45, verbose=False)
        detections = []
        
        if len(results) > 0 and results[0].boxes is not None:
            for box in results[0].boxes:
                class_id = int(box.cls[0])
                class_name = self.yolo.names[class_id]
                confidence = float(box.conf[0])
                
                detections.append({
                    'class': class_name,
                    'confidence': confidence
                })
        
        return detections
    
    def process_frame(self, image_path: Path) -> Dict:
        """
        Обработка одного кадра
        """
        start_time = time.time()
        
        image = cv2.imread(str(image_path))
        if image is None:
            return None
        
        # Классификация
        cls, conf = self.classify(image)
        
        # Детекция (если anomaly)
        detections = []
        if cls == "anomaly":
            detections = self.detect_defects(image)
        
        elapsed = time.time() - start_time
        
        return {
            'image_path': image_path,
            'classification': {
                'class': cls,
                'confidence': conf
            },
            'detections': detections,
            'processing_time': elapsed,
            'has_defects': len(detections) > 0
        }
    
    def scan_dataset(self, dataset_path: Path, visualize: bool = False):
        """
        Обработка всего датасета
        """
        dataset_path = Path(dataset_path)
        
        # Найти все изображения
        images = list(dataset_path.rglob("*.png")) + list(dataset_path.rglob("*.jpg"))
        images = sorted(images)
        
        print(f"\n[SCAN] Найдено {len(images)} изображений")
        print(f"[SCAN] Начинаю обработку...\n")
        
        for idx, img_path in enumerate(images):
            result = self.process_frame(img_path)
            if result is None:
                continue
            
            # Обновляем статистику
            self.stats['total_frames'] += 1
            self.stats['total_time'] += result['processing_time']
            self.stats['classifier_predictions'][result['classification']['class']] += 1
            
            if result['has_defects']:
                self.stats['frames_with_yolo_hits'] += 1
                for det in result['detections']:
                    self.stats['yolo_detections'][det['class']] += 1
            
            # Печатаем прогресс
            if (idx + 1) % max(1, len(images) // 10) == 0 or idx == 0:
                cls = result['classification']['class']
                conf = result['classification']['confidence']
                det_count = len(result['detections'])
                print(f"  [{idx+1}/{len(images)}] {cls} ({conf:.2f}) | Defects: {det_count}")
            
            # Визуализация
            if visualize:
                vis = self._draw_result(result['image_path'], result)
                cv2.imshow("Bottle Inspector", vis)
                if cv2.waitKey(1) == ord('q'):
                    break
        
        if visualize:
            cv2.destroyAllWindows()
        
        self.print_report()
    
    def _draw_result(self, image_path: Path, result: Dict) -> np.ndarray:
        """Рисует результаты на изображении"""
        image = cv2.imread(str(image_path))
        
        # Текст классификации
        cls = result['classification']['class']
        conf = result['classification']['confidence']
        color = (0, 255, 0) if cls == 'good' else (0, 0, 255)
        
        cv2.putText(image, f"{cls} ({conf:.2f})", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        # Bounding boxes (если есть дефекты)
        if result['has_defects']:
            for i, det in enumerate(result['detections']):
                y_offset = 70 + i * 30
                cv2.putText(image, 
                           f"  - {det['class']} ({det['confidence']:.2f})",
                           (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
        
        return image
    
    def print_report(self):
        """Печать финального отчета"""
        if self.stats['total_frames'] == 0:
            print("[ERROR] Не обработано изображений!")
            return
        
        fps = self.stats['total_frames'] / self.stats['total_time']
        
        print(f"\nОБЩАЯ СТАТИСТИКА:")
        print(f"  Обработано кадров: {self.stats['total_frames']}")
        print(f"  Общее время: {self.stats['total_time']:.2f} сек")
        print(f"  Средний FPS: {fps:.1f}")
        print(f"  Среднее время на кадр: {self.stats['total_time']/self.stats['total_frames']*1000:.1f} мс")
        
        print(f"\nРЕЗУЛЬТАТЫ КЛАССИФИКАЦИИ (ResNet50):")
        for cls_name in self.class_names:
            count = self.stats['classifier_predictions'][cls_name]
            pct = 100 * count / self.stats['total_frames']
            print(f"  {cls_name:15} : {count:3} ({pct:5.1f}%)")
        
        if self.yolo_available:
            print(f"\nРЕЗУЛЬТАТЫ ДЕТЕКЦИИ (YOLOv8):")
            print(f"  Кадров с дефектами: {self.stats['frames_with_yolo_hits']} / {self.stats['total_frames']}")
            
            if len(self.stats['yolo_detections']) > 0:
                for defect_class, count in sorted(self.stats['yolo_detections'].items()):
                    print(f"    - {defect_class:20} : {count:3} найдено")
            else:
                print(f"    Дефектов не найдено")
        



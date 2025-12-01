"""
src/pipeline.py
===============
Комбинированный pipeline для bottle defect detection:

- Классификатор BottleClassifier (ResNet50) определяет класс:
  ['defect_broken_large', 'defect_broken_small', 'defect_contamination', 'good']

- Если картинка "good" с высокой уверенностью — YOLO не запускается.
- Если найден дефект — запускается YOLOv8 для локализации (bounding boxes).

"""
from PIL import Image
from pathlib import Path

import cv2
import torch
import torch.nn.functional as F
from torchvision import transforms
from ultralytics import YOLO

from src.model_classifier import BottleClassifier


class BottleDefectPipeline:
    """
    Комбинированный pipeline: классификация (ResNet50) + детекция (YOLOv8).
    """

    def __init__(
        self,
        classifier_path: str = "models/bottle_classifier_best.pth",
        detector_path: str = "models/bottle_yolo/weights/best.pt",
        device: str = "cpu",
    ):
        # Устройство: CPU или CUDA
        self.device = torch.device(device)

        # === 1. Классификатор BottleClassifier (ResNet50) ===
        self.classifier = BottleClassifier(num_classes=4, pretrained=False)
        self.classifier.to(self.device)

        ckpt_path = Path(classifier_path)
        if ckpt_path.exists():
            checkpoint = torch.load(ckpt_path, map_location=self.device)

            # Поддерживаем оба варианта сохранения: чистый state_dict или словарь с ключом
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            else:
                state_dict = checkpoint

            self.classifier.load_state_dict(state_dict)
            print(f"[Pipeline] Загружен классификатор: {ckpt_path}")
        else:
            print(f"[Pipeline] ВНИМАНИЕ: классификатор {ckpt_path} не найден, модель без обученных весов")

        self.classifier.eval()

        # Порядок классов должен соответствовать обучению BottleClassifier
        self.class_names = [
            "defect_broken_large",
            "defect_broken_small",
            "defect_contamination",
            "good",
        ]

        # Трансформации под ResNet50 / ImageNet
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

        # === 2. YOLOv8 детектор дефектов ===
        self.detector = YOLO(detector_path)
        # Порядок классов YOLO — только дефекты
        self.detector_names = [
            "defect_broken_large",
            "defect_broken_small",
            "defect_contamination",
        ]

    # ---------------- Классификация ----------------

    def classify(self, image):
        """
        Классификация кадра с помощью BottleClassifier.
        Args:
           image: BGR картинка (OpenCV), np.ndarray.

        Returns:
           (class_name, confidence)
        """
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)  # numpy → PIL.Image
        img_tensor = self.transform(img_pil).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.classifier(img_tensor)
            probs = F.softmax(logits, dim=1)
            idx = int(torch.argmax(probs, dim=1).item())
            conf = float(probs[0, idx].item())
        return self.class_names[idx], conf


    # ---------------- Детекция YOLO ----------------

    def detect(self, image, conf_thres: float = 0.1):
        """
        Локализация дефектов с помощью YOLOv8.

        Args:
            image: BGR картинка (OpenCV)
            conf_thres: минимальная confidence для боксов

        Returns:
            список детекций:
            [{'class': str, 'confidence': float, 'bbox': [x1,y1,x2,y2]}, ...]
        """
        results = self.detector(image, conf=conf_thres, verbose=False)
        detections = []

        if results[0].boxes is not None:
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                score = float(box.conf.item())
                cls_id = int(box.cls.item())
                # защита от выхода за размер списка имён
                if 0 <= cls_id < len(self.detector_names):
                    cls_name = self.detector_names[cls_id]
                else:
                    cls_name = f"class_{cls_id}"

                detections.append(
                    {
                        "class": cls_name,
                        "confidence": score,
                        "bbox": [x1, y1, x2, y2],
                    }
                )

        return detections

    # ---------------- Главная логика ----------------

    def process_frame(self, image):
        """
        Комбинированный шаг обработки кадра:

        1) Классификация (BottleClassifier).
        2) Если класс == 'good' и доверие > 0.9 — YOLO не запускается.
        3) Иначе — YOLO локализует дефекты.

        Returns:
            dict с полями:
              - classification: {'class': str, 'confidence': float}
              - detections: список детекций (см. detect)
              - skipped_detection: bool (запускался ли YOLO)
        """
        cls, conf = self.classify(image)

        if cls == "good" and conf > 0.9:
            # Высокая уверенность, что бутылка good → YOLO не нужен
            return {
                "classification": {"class": cls, "confidence": conf},
                "detections": [],
                "skipped_detection": True,
            }
        else:
            # Возможен дефект → запускаем YOLO
            boxes = self.detect(image)
            return {
                "classification": {"class": cls, "confidence": conf},
                "detections": boxes,
                "skipped_detection": False,
            }

    # ---------------- Визуализация ----------------

    def visualize(self, image, result):
        """
        Рисуем результат pipeline на кадре:
        - класс/уверенность (сверху слева),
        - bounding boxes для детекций.

        Args:
            image: исходный BGR кадр
            result: dict из process_frame

        Returns:
            vis: BGR картинка с разметкой
        """
        vis = image.copy()

        cls = result["classification"]["class"]
        conf = result["classification"]["confidence"]

        # Цвет: зелёный для good, красный для дефектов
        color = (0, 255, 0) if cls == "good" else (0, 0, 255)
        text = f"{cls} ({conf:.2f})"
        cv2.putText(
            vis,
            text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            color,
            2,
            lineType=cv2.LINE_AA,
        )

        # Рисуем bounding boxes
        for det in result["detections"]:
            x1, y1, x2, y2 = det["bbox"]
            box_label = f"{det['class']} {det['confidence']:.2f}"
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(
                vis,
                box_label,
                (x1, max(y1 - 10, 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
                lineType=cv2.LINE_AA,
            )

        return vis



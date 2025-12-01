"""
data_preparation.py
===================
Подготовка данных для классификации:
- читает raw MVTec Bottle из data/raw/mvtec_bottle
- создаёт data/processed/bottle/train/val/test с папками по классам
- пишет мета-информацию в data/dataset_info.json
"""

import os
import shutil
import cv2
import shutil
from pathlib import Path
from typing import Dict, List
import numpy as np
from sklearn.model_selection import train_test_split
import logging

from src.utils import create_dirs, save_json, get_project_root, setup_logger


# Какие классы используем для Bottle
BOTTLE_CLASSES = {
    "good": "good",
    "broken_large": "defect_broken_large",
    "broken_small": "defect_broken_small",
    "contamination": "defect_contamination",
}



def collect_bottle_images(raw_root: Path) -> Dict[str, List[Path]]:
    """
    Собирает все изображения Bottle по классам.
    
    Args:
      raw_root: data/raw/mvtec_bottle
    
    Returns::
        dict: { target_class_name: [список путей к изображениям] }
    """
    img_dict = {v: [] for v in BOTTLE_CLASSES.values()}
    
    train_good_dir = raw_root / "train" / "good"
    if not train_good_dir.exists():
        raise FileNotFoundError(f"Не найдена папка {train_good_dir}")

    for img in train_good_dir.glob("*.*"):
        img_dict["good"].append(img)

    test_good_dir = raw_root / "test" / "good"
    if not test_good_dir.exists():
        raise FileNotFoundError(f"Не найдена папка {test_good_dir}")

    for img in test_good_dir.glob("*.*"):
        img_dict["good"].append(img)

    for raw_name, target_name in BOTTLE_CLASSES.items():
        if raw_name == "good":
            continue
        defect_dir = raw_root / "test" / raw_name
        if not defect_dir.exists():
            raise FileNotFoundError(f"Не найдена папка {defect_dir}")
        for img in defect_dir.glob("*.*"):
            img_dict[target_name].append(img)

    return img_dict
        
        
def split_and_copy_bottle(raw_root: Path, processed_root: Path, logger):
    """
    Делит данные Bottle на train/val/test и копирует в processed структуру.
    
    processed_root: data/processed/bottle
    """
    img_dict = collect_bottle_images(raw_root)

    train_root = processed_root / "train"
    val_root = processed_root / "val"
    test_root = processed_root / "test"

    for cls in BOTTLE_CLASSES.values():
        create_dirs(train_root / cls, val_root / cls, test_root / cls)

    dataset_info = {
        "bottle": {
            "classes": list(BOTTLE_CLASSES.values()),
            "splits": {
                "train": {},
                "val": {},
                "test": {},
            }
        }
    }

    for cls_name, img_paths in img_dict.items():
        img_paths = list(img_paths)
        if len(img_paths) == 0:
            logger.warning(f"Класс {cls_name} пуст!")
            continue

        train_imgs, temp_imgs = train_test_split(
            img_paths, test_size=0.3, random_state=42
        )
        val_imgs, test_imgs = train_test_split(
            temp_imgs, test_size=0.5, random_state=42
        )

        dataset_info["bottle"]["splits"]["train"][cls_name] = len(train_imgs)
        dataset_info["bottle"]["splits"]["val"][cls_name] = len(val_imgs)
        dataset_info["bottle"]["splits"]["test"][cls_name] = len(test_imgs)

        for src in train_imgs:
            dst = train_root / cls_name / src.name
            shutil.copy2(src, dst)

        for src in val_imgs:
            dst = val_root / cls_name / src.name
            shutil.copy2(src, dst)

        for src in test_imgs:
            dst = test_root / cls_name / src.name
            shutil.copy2(src, dst)

        logger.info(
            f"Bottle / {cls_name}: train={len(train_imgs)}, "
            f"val={len(val_imgs)}, test={len(test_imgs)}"
        )

    return dataset_info


#================= YOLO ===================
YOLO_BOTTLE_CLASSES = [
    "defect_broken_large",
    "defect_broken_small",
    "defect_contamination",
]

YOLO_CLASS_TO_ID = {name: idx for idx, name in enumerate(YOLO_BOTTLE_CLASSES)}


def create_yolo_dirs(root: Path):
    """Создаёт папки data/yolo/{images,labels}/{train,val,test}"""
    images_root = root / "images"
    labels_root = root / "labels"
    for split in ["train", "val", "test"]:
        (images_root / split).mkdir(parents=True, exist_ok=True)
        (labels_root / split).mkdir(parents=True, exist_ok=True)
        
        
def mask_to_bbox(mask: np.ndarray):
    """
    Находит bounding box по бинарной маске
    mask: HxW, белый (255) = дефект
    Возвращает (x_min, y_min, x_max, y_max) или None, если дефекта нет
    """
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    return x_min, y_min, x_max, y_max


def bbox_to_yolo(x_min, y_min, x_max, y_max, img_w, img_h):
    """Перевод bbox в YOLO формат (нормированные координаты)."""
    x_c = (x_min + x_max) / 2.0 / img_w
    y_c = (y_min + y_max) / 2.0 / img_h
    w = (x_max - x_min) / img_w
    h = (y_max - y_min) / img_h
    return x_c, y_c, w, h


def generate_yolo_for_bottle(root: Path, logger):
    """Генерация ПОЛНЫХ YOLO данных: images + labels"""
    
    processed_root = root / "data" / "processed" / "bottle"
    yolo_root = root / "data" / "yolo"
    raw_root = root / "data" / "raw" / "mvtec_bottle"
    create_yolo_dirs(yolo_root)

    class_to_raw = {
        "defect_broken_large": "broken_large",
        "defect_broken_small": "broken_small", 
        "defect_contamination": "contamination",
    }

    total_images = 0
    total_labels = 0
    
    for split in ["train", "val"]:
        logger.info(f"--- {split} ---")
        split_dir = processed_root / split
        images_out_dir = yolo_root / "images" / split
        labels_out_dir = yolo_root / "labels" / split

        # 1. КОПИРУЕМ ВСЕ good
        good_dir = split_dir / "good"
        good_count = 0
        if good_dir.exists():
            for img_name in os.listdir(good_dir):
                if img_name.lower().endswith((".png", ".jpg")):
                    src_img = good_dir / img_name
                    dst_img = images_out_dir / img_name
                    shutil.copy2(src_img, dst_img)
                    open(labels_out_dir / (Path(img_name).stem + ".txt"), "w").close()
                    good_count += 1
        logger.info(f"good: {good_count} images")

        # 2. ДЕФЕКТЫ с bbox
        defect_count = 0
        for cls_name in ["defect_broken_large", "defect_broken_small", "defect_contamination"]:
            class_dir = split_dir / cls_name
            if not class_dir.exists(): continue
            
            cls_defects = 0
            for img_name in os.listdir(class_dir):
                if not img_name.lower().endswith((".png", ".jpg")): continue

                # КОПИРУЕМ ИЗОБРАЖЕНИЕ
                src_img = class_dir / img_name
                dst_img = images_out_dir / img_name
                shutil.copy2(src_img, dst_img)
                total_images += 1

                # bbox из маски
                raw_defect_name = class_to_raw[cls_name]
                mask_name = Path(img_name).stem + "_mask.png"
                mask_path = raw_root / "ground_truth" / raw_defect_name / mask_name
                
                label_path = labels_out_dir / (Path(img_name).stem + ".txt")
                if not mask_path.exists():
                    open(label_path, "w").close()
                    continue

                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    open(label_path, "w").close()
                    continue

                bbox = mask_to_bbox(mask)
                if bbox is None:
                    open(label_path, "w").close()
                    continue

                # Записываем bbox
                x_min, y_min, x_max, y_max = bbox
                img = cv2.imread(str(class_dir / img_name))
                h, w = img.shape[:2]
                x_c, y_c, bw, bh = bbox_to_yolo(x_min, y_min, x_max, y_max, w, h)
                class_id = YOLO_CLASS_TO_ID[cls_name]
                
                with open(label_path, "w") as f:
                    f.write(f"{class_id} {x_c:.6f} {y_c:.6f} {bw:.6f} {bh:.6f}\n")
                cls_defects += 1
                total_labels += 1
            
            defect_count += cls_defects
            logger.info(f"{cls_name}: {cls_defects} bbox")
        
        logger.info(f"{split}: {good_count} good + {defect_count} defects = {good_count+defect_count} total")
    
    logger.info(f"ИТОГО: {total_images} images, {total_labels} labels")
    
    # data.yaml
    data_yaml_path = yolo_root / "data.yaml"
    with open(data_yaml_path, "w") as f:
        f.write(f"""path: {yolo_root}
train: images/train
val: images/val
test: images/test
nc: 3
names: ['defect_broken_large', 'defect_broken_small', 'defect_contamination']""")





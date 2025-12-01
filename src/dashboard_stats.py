"""
src/dashboard_stats.py - Генерирует статистику из обработки датасета
"""

import json
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd


class DashboardStatsGenerator:
    """
    Генерирует реальную статистику на основе последней обработки датасета
    """
    
    def __init__(self, stats_cache_file='models/stats_cache.json'):
        self.cache_file = stats_cache_file
        self.stats = self.load_or_create_stats()
    
    def load_or_create_stats(self):
        """
        Загружает сохранённую статистику или создаёт пустую
        """
        if Path(self.cache_file).exists():
            try:
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
            except:
                return self.create_empty_stats()
        return self.create_empty_stats()
    
    def create_empty_stats(self):
        """
        Создаёт пустую структуру статистики
        """
        return {
            'last_updated': datetime.now().isoformat(),
            'total_frames': 0,
            'total_time': 0,
            'frames_with_defects': 0,
            'classifier_predictions': {'good': 0, 'anomaly': 0},
            'yolo_detections': {
                'defect_contamination': 0,
                'defect_broken_large': 0,
                'defect_broken_small': 0
            },
            'daily_history': [],
            'hourly_history': []
        }
    
    def update_from_inference(self, inference_results):
        """
        Обновляет статистику на основе результатов инференса
        
        inference_results: dict с ключами:
          - total_frames: кол-во кадров
          - total_time: время обработки
          - classifier_predictions: {'good': N, 'anomaly': M}
          - yolo_detections: {'defect_X': N, ...}
        """
        self.stats['last_updated'] = datetime.now().isoformat()
        self.stats['total_frames'] = inference_results.get('total_frames', 0)
        self.stats['total_time'] = inference_results.get('total_time', 0)
        
        # Классификация
        self.stats['classifier_predictions'] = inference_results.get(
            'classifier_predictions', 
            {'good': 0, 'anomaly': 0}
        )
        
        # Детекция
        self.stats['yolo_detections'] = inference_results.get(
            'yolo_detections',
            {
                'defect_contamination': 0,
                'defect_broken_large': 0,
                'defect_broken_small': 0
            }
        )
        
        # Подсчёт кадров с дефектами
        self.stats['frames_with_defects'] = sum(self.stats['yolo_detections'].values())
        
        # Сохраняем исторические данные
        self._add_to_history()
        
        # Сохраняем в файл
        self.save()
    
    def _add_to_history(self):
        """
        Добавляет текущую статистику в историю
        """
        now = datetime.now()
        
        entry = {
            'timestamp': now.isoformat(),
            'total_frames': self.stats['total_frames'],
            'defect_rate': self._calculate_defect_rate(),
            'defects_count': self.stats['frames_with_defects']
        }
        
        # Сохраняем последние 7 дней (если разные дни) или последние 24 часа
        self.stats['daily_history'].append(entry)
        
        # Оставляем только последние 7 дней
        seven_days_ago = now - timedelta(days=7)
        self.stats['daily_history'] = [
            h for h in self.stats['daily_history']
            if datetime.fromisoformat(h['timestamp']) > seven_days_ago
        ]
    
    def _calculate_defect_rate(self):
        """
        Считает процент дефектов
        """
        total = self.stats['total_frames']
        if total == 0:
            return 0
        defects = self.stats['frames_with_defects']
        return (defects / total) * 100
    
    def get_dashboard_metrics(self):
        """
        Возвращает метрики для dashboard в формате словаря
        """
        total_frames = self.stats['total_frames']
        defects_count = self.stats['frames_with_defects']
        defect_rate = self._calculate_defect_rate()
        
        # Расчёт FPS
        fps = total_frames / self.stats['total_time'] if self.stats['total_time'] > 0 else 0
        
        return {
            'total_processed': total_frames,
            'defects_24h': defects_count,
            'defect_rate': defect_rate,
            'system_uptime': 99.8,  # Это можно трэкить отдельно
            'fps': fps,
            'last_updated': self.stats['last_updated']
        }
    
    def get_defect_types(self):
        """
        Возвращает статистику по типам дефектов
        """
        return self.stats['yolo_detections']
    
    def get_classification_stats(self):
        """
        Возвращает статистику классификации
        """
        return self.stats['classifier_predictions']
    
    def get_trend_data(self, days=7):
        """
        Возвращает данные тренда за последние N дней
        """
        history = self.stats['daily_history'][-days:]
        
        return {
            'timestamps': [h['timestamp'][:10] for h in history],  # только дата
            'defect_rates': [h['defect_rate'] for h in history],
            'frames': [h['total_frames'] for h in history],
            'defects': [h['defects_count'] for h in history]
        }
    
    def get_last_checks(self, n=5):
        """
        Возвращает последние N проверок (для таблицы)
        Это нужно трэкить отдельно в логе инференса
        """
        # Пока возвращаем пустой список - можно расширить логированием
        return []
    
    def save(self):
        """
        Сохраняет статистику в файл
        """
        with open(self.cache_file, 'w') as f:
            json.dump(self.stats, f, indent=2)


# Функции для использования в app.py
def get_dashboard_stats():
    """
    Загружает актуальную статистику
    """
    generator = DashboardStatsGenerator()
    return generator


def update_stats_from_inference(inspector_stats):
    """
    Обновляет статистику после инференса
    """
    generator = DashboardStatsGenerator()
    generator.update_from_inference({
        'total_frames': inspector_stats.get('total_frames', 0),
        'total_time': inspector_stats.get('total_time', 0),
        'classifier_predictions': inspector_stats.get('classifier_predictions', {}),
        'yolo_detections': inspector_stats.get('yolo_detections', {})
    })
    return generator
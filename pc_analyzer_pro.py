#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PC Analyzer Pro - Профессиональная система мониторинга и защиты ПК
Версия 1.0.0
Разработано с использованием Python, PyQt5, scikit-learn
"""

import sys
import os
import json
import time
import datetime
import sqlite3
import threading
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
from collections import deque
import warnings

# Настройка логгирования
warnings.filterwarnings('ignore')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pc_analyzer.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('PCAnalyzerPro')

# Проверка доступности библиотек
try:
    import psutil
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    from sklearn.neural_network import MLPRegressor
    import requests
except ImportError as e:
    logger.error(f"Ошибка импорта библиотеки: {e}")
    sys.exit(1)

try:
    from PyQt5.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QLabel, QPushButton, QTabWidget, QProgressBar, QGroupBox,
        QScrollArea, QFrame, QSizePolicy, QMenu, QAction, QMessageBox,
        QFileDialog, QDialog, QTextEdit, QComboBox, QSpinBox, QCheckBox,
        QSystemTrayIcon, QGridLayout, QSplitter, QTreeWidget, QTreeWidgetItem,
        QStatusBar, QToolBar, QDialogButtonBox, QListWidget, QListWidgetItem,
        QStackedWidget, QSlider, QLCDNumber, QCalendarWidget, QDateTimeEdit,
        QFormLayout, QRadioButton, QButtonGroup, QTableWidget, QTableWidgetItem,
        QHeaderView, QAbstractItemView, QColorDialog, QFontDialog, QSplashScreen
    )
    from PyQt5.QtCore import (
        Qt, QTimer, QThread, pyqtSignal, QObject, QDate, QDateTime,
        QSettings, QPropertyAnimation, QEasingCurve, QSize, QPoint, QRect,
        QParallelAnimationGroup, QSequentialAnimationGroup, QFileInfo,
        QDir, QUrl, QMimeData, QByteArray, QBuffer, QIODevice, QLineF,
        QPointF, QRectF, QMarginsF, QMargins
    )
    from PyQt5.QtGui import (
        QIcon, QPixmap, QImage, QPainter, QColor, QPen, QBrush,
        QFont, QPalette, QCursor, QMovie, QBitmap, QMask, QIconEngine,
        QLinearGradient, QRadialGradient, QConicalGradient, QTransform,
        QMatrix4x4, QVector3D, QQuaternion, QPolygon, QPolygonF, QRegion,
        QTextDocument, QTextCursor, QTextFormat, QTextCharFormat, QFontMetrics,
        QFontInfo, QFontDatabase, QSyntaxHighlighter, QTextOption
    )
    PYQT_AVAILABLE = True
except ImportError:
    PYQT_AVAILABLE = False
    logger.warning("PyQt5 недоступен, запуск в консольном режиме")


# ============================================================================
# КОНФИГУРАЦИЯ И КОНСТАНТЫ
# ============================================================================

class Config:
    """Конфигурация приложения"""
    APP_NAME = "PC Analyzer Pro"
    VERSION = "1.0.0"
    AUTHOR = "AI Assistant"
    
    # Пороги предупреждений
    CPU_WARNING_THRESHOLD = 75
    CPU_CRITICAL_THRESHOLD = 95
    RAM_WARNING_THRESHOLD = 80
    RAM_CRITICAL_THRESHOLD = 95
    DISK_WARNING_THRESHOLD = 85
    DISK_CRITICAL_THRESHOLD = 95
    TEMP_WARNING_THRESHOLD = 70
    TEMP_CRITICAL_THRESHOLD = 90
    
    # Интервалы обновления (мс)
    UPDATE_INTERVAL_FAST = 1000
    UPDATE_INTERVAL_NORMAL = 5000
    UPDATE_INTERVAL_SLOW = 30000
    
    # База данных
    DB_NAME = "pc_analyzer.db"
    
    # Настройки нейросети
    NN_HIDDEN_LAYERS = (100, 50, 25)
    NN_MAX_ITER = 500
    NN_TOLERANCE = 1e-4
    
    # Автозащита
    AUTO_PROTECTION_ENABLED = True
    PROTECTION_COOLDOWN = 60  # секунд между действиями защиты


# ============================================================================
# МОДЕЛИ ДАННЫХ
# ============================================================================

@dataclass
class SystemInfo:
    """Информация о системе"""
    hostname: str = ""
    platform: str = ""
    architecture: str = ""
    processor: str = ""
    cpu_count_physical: int = 0
    cpu_count_logical: int = 0
    cpu_freq_max: float = 0.0
    ram_total: int = 0
    ram_available: int = 0
    boot_time: float = 0.0
    gpu_info: List[str] = None
    
    def __post_init__(self):
        if self.gpu_info is None:
            self.gpu_info = []


@dataclass
class ResourceMetrics:
    """Метрики ресурсов"""
    timestamp: float = 0.0
    cpu_percent: float = 0.0
    cpu_per_core: List[float] = None
    ram_percent: float = 0.0
    ram_used: int = 0
    ram_available: int = 0
    disk_percent: float = 0.0
    disk_used: int = 0
    disk_free: int = 0
    network_sent: int = 0
    network_recv: int = 0
    upload_speed: float = 0.0
    download_speed: float = 0.0
    temp_cpu: float = 0.0
    fan_speed: int = 0
    battery_percent: float = 0.0
    power_plugged: bool = False
    
    def __post_init__(self):
        if self.cpu_per_core is None:
            self.cpu_per_core = []


@dataclass
class ProcessInfo:
    """Информация о процессе"""
    pid: int = 0
    name: str = ""
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    memory_rss: int = 0
    status: str = ""
    username: str = ""
    create_time: float = 0.0
    threads: int = 0
    nice: int = 0


@dataclass
class Alert:
    """Предупреждение/оповещение"""
    id: int = 0
    timestamp: float = 0.0
    severity: str = "info"  # info, warning, critical
    category: str = ""
    message: str = ""
    resolved: bool = False
    resolution_time: float = 0.0


# ============================================================================
# БАЗА ДАННЫХ
# ============================================================================

class DatabaseManager:
    """Менеджер базы данных"""
    
    def __init__(self, db_path: str = Config.DB_NAME):
        self.db_path = db_path
        self.conn = None
        self.init_database()
    
    def connect(self):
        """Подключение к базе данных"""
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        return self.conn.cursor()
    
    def init_database(self):
        """Инициализация структуры базы данных"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Таблица метрик ресурсов
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS resource_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                cpu_percent REAL,
                ram_percent REAL,
                ram_used INTEGER,
                disk_percent REAL,
                network_sent INTEGER,
                network_recv INTEGER,
                temp_cpu REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Таблица процессов
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS processes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                pid INTEGER,
                name TEXT,
                cpu_percent REAL,
                memory_percent REAL,
                memory_rss INTEGER,
                status TEXT
            )
        ''')
        
        # Таблица предупреждений
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                severity TEXT,
                category TEXT,
                message TEXT,
                resolved INTEGER DEFAULT 0,
                resolution_time REAL
            )
        ''')
        
        # Таблица отчетов
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                report_type TEXT,
                generated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                data_json TEXT,
                file_path TEXT
            )
        ''')
        
        # Таблица настроек системы
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS system_profiles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                system_hash TEXT UNIQUE,
                profile_name TEXT,
                cpu_model TEXT,
                ram_total INTEGER,
                disk_total INTEGER,
                gpu_model TEXT,
                performance_score REAL,
                recommended_settings TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Таблица истории сети
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS network_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                interface_name TEXT,
                bytes_sent INTEGER,
                bytes_recv INTEGER,
                packets_sent INTEGER,
                packets_recv INTEGER,
                wifi_signal_strength REAL,
                wifi_ssid TEXT
            )
        ''')
        
        # Таблица температуры
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS temperature_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                sensor_name TEXT,
                temperature REAL,
                fan_speed INTEGER
            )
        ''')
        
        # Индексы для ускорения запросов
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON resource_metrics(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON alerts(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_processes_timestamp ON processes(timestamp)')
        
        conn.commit()
        conn.close()
        logger.info("База данных инициализирована")
    
    def save_metric(self, metrics: ResourceMetrics):
        """Сохранение метрик"""
        cursor = self.connect()
        cursor.execute('''
            INSERT INTO resource_metrics 
            (timestamp, cpu_percent, ram_percent, ram_used, disk_percent, 
             network_sent, network_recv, temp_cpu)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            metrics.timestamp, metrics.cpu_percent, metrics.ram_percent,
            metrics.ram_used, metrics.disk_percent, metrics.network_sent,
            metrics.network_recv, metrics.temp_cpu
        ))
        self.conn.commit()
    
    def save_alert(self, alert: Alert):
        """Сохранение предупреждения"""
        cursor = self.connect()
        cursor.execute('''
            INSERT INTO alerts (timestamp, severity, category, message, resolved)
            VALUES (?, ?, ?, ?, ?)
        ''', (alert.timestamp, alert.severity, alert.category, alert.message, 
              1 if alert.resolved else 0))
        self.conn.commit()
        return cursor.lastrowid
    
    def get_recent_metrics(self, limit: int = 100) -> List[Dict]:
        """Получение последних метрик"""
        cursor = self.connect()
        cursor.execute('''
            SELECT * FROM resource_metrics ORDER BY timestamp DESC LIMIT ?
        ''', (limit,))
        return [dict(row) for row in cursor.fetchall()]
    
    def get_alerts(self, days: int = 7) -> List[Dict]:
        """Получение предупреждений за период"""
        cursor = self.connect()
        cutoff_time = time.time() - (days * 24 * 3600)
        cursor.execute('''
            SELECT * FROM alerts WHERE timestamp >= ? ORDER BY timestamp DESC
        ''', (cutoff_time,))
        return [dict(row) for row in cursor.fetchall()]
    
    def cleanup_old_data(self, days: int = 30):
        """Очистка старых данных"""
        cursor = self.connect()
        cutoff_time = time.time() - (days * 24 * 3600)
        cursor.execute('DELETE FROM resource_metrics WHERE timestamp < ?', (cutoff_time,))
        cursor.execute('DELETE FROM processes WHERE timestamp < ?', (cutoff_time,))
        self.conn.commit()
        logger.info(f"Очищены данные старше {days} дней")


# ============================================================================
# НЕЙРОСЕТЬ ДЛЯ ПРОГНОЗИРОВАНИЯ
# ============================================================================

class PCNeuralNetwork:
    """Нейросеть для анализа и прогнозирования состояния ПК"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.anomaly_detector = IsolationForest(
            contamination=0.1,
            random_state=42,
            n_estimators=100
        )
        self.predictor = MLPRegressor(
            hidden_layer_sizes=Config.NN_HIDDEN_LAYERS,
            max_iter=Config.NN_MAX_ITER,
            tol=Config.NN_TOLERANCE,
            random_state=42
        )
        self.training_data = deque(maxlen=1000)
        self.is_trained = False
        self.anomalies_detected = 0
        
    def add_sample(self, metrics: ResourceMetrics, label: float = 0.0):
        """Добавление образца для обучения"""
        features = self._extract_features(metrics)
        self.training_data.append((features, label))
        
        # Переобучение при накоплении данных
        if len(self.training_data) >= 100 and len(self.training_data) % 50 == 0:
            self.retrain()
    
    def _extract_features(self, metrics: ResourceMetrics) -> np.ndarray:
        """Извлечение признаков из метрик"""
        features = [
            metrics.cpu_percent,
            metrics.ram_percent,
            metrics.disk_percent,
            metrics.temp_cpu if metrics.temp_cpu > 0 else 40,
            metrics.upload_speed,
            metrics.download_speed,
            len(metrics.cpu_per_core),
            np.mean(metrics.cpu_per_core) if metrics.cpu_per_core else 0,
            np.std(metrics.cpu_per_core) if metrics.cpu_per_core else 0,
            metrics.ram_used / (metrics.ram_used + metrics.ram_available + 1)
        ]
        return np.array(features).reshape(1, -1)
    
    def retrain(self):
        """Переобучение модели"""
        if len(self.training_data) < 50:
            return
        
        X = np.vstack([f for f, _ in self.training_data])
        y = np.array([l for _, l in self.training_data])
        
        try:
            X_scaled = self.scaler.fit_transform(X)
            
            # Обучение детектора аномалий
            self.anomaly_detector.fit(X_scaled)
            
            # Обучение предсказателя
            if len(y.shape) > 1 or np.unique(y).size > 1:
                self.predictor.fit(X_scaled, y)
            
            self.is_trained = True
            logger.info(f"Нейросеть переобучена на {len(self.training_data)} образцах")
        except Exception as e:
            logger.error(f"Ошибка обучения нейросети: {e}")
    
    def detect_anomaly(self, metrics: ResourceMetrics) -> Tuple[bool, float]:
        """Обнаружение аномалий"""
        features = self._extract_features(metrics)
        
        if not self.is_trained:
            return False, 0.0
        
        try:
            features_scaled = self.scaler.transform(features)
            prediction = self.anomaly_detector.predict(features_scaled)[0]
            score = self.anomaly_detector.score_samples(features_scaled)[0]
            
            is_anomaly = prediction == -1
            if is_anomaly:
                self.anomalies_detected += 1
            
            return is_anomaly, score
        except Exception as e:
            logger.error(f"Ошибка обнаружения аномалии: {e}")
            return False, 0.0
    
    def predict_overload(self, metrics: ResourceMetrics, horizon: int = 5) -> float:
        """Прогноз перегрузки"""
        if not self.is_trained or len(self.training_data) < 100:
            # Эвристический прогноз
            risk = (
                metrics.cpu_percent * 0.4 +
                metrics.ram_percent * 0.3 +
                (metrics.temp_cpu / 100) * 0.3
            )
            return risk
        
        try:
            features = self._extract_features(metrics)
            features_scaled = self.scaler.transform(features)
            prediction = self.predictor.predict(features_scaled)[0]
            return min(max(prediction, 0), 100)
        except Exception as e:
            logger.error(f"Ошибка прогноза: {e}")
            return 50.0
    
    def get_recommendations(self, metrics: ResourceMetrics) -> List[str]:
        """Получение рекомендаций на основе анализа"""
        recommendations = []
        
        if metrics.cpu_percent > 80:
            recommendations.append("Высокая нагрузка на CPU. Рассмотрите закрытие фоновых приложений.")
        
        if metrics.ram_percent > 85:
            recommendations.append("Недостаточно оперативной памяти. Освободите память или добавьте RAM.")
        
        if metrics.temp_cpu > 75:
            recommendations.append("Высокая температура CPU. Проверьте систему охлаждения.")
        
        if metrics.disk_percent > 90:
            recommendations.append("Критически мало места на диске. Очистите ненужные файлы.")
        
        is_anomaly, _ = self.detect_anomaly(metrics)
        if is_anomaly:
            recommendations.append("Обнаружена аномальная активность системы. Проверьте на вирусы.")
        
        return recommendations
    
    def get_stats(self) -> Dict:
        """Статистика работы нейросети"""
        return {
            "training_samples": len(self.training_data),
            "is_trained": self.is_trained,
            "anomalies_detected": self.anomalies_detected,
            "model_type": "IsolationForest + MLPRegressor"
        }


# ============================================================================
# СИСТЕМНЫЙ МОНИТОР
# ============================================================================

class SystemMonitor(QObject):
    """Мониторинг системы"""
    
    metrics_updated = pyqtSignal(ResourceMetrics)
    alert_triggered = pyqtSignal(Alert)
    process_list_updated = pyqtSignal(list)
    
    def __init__(self):
        super().__init__()
        self.running = False
        self.thread = None
        self.metrics_history = deque(maxlen=1000)
        self.process_cache = {}
        self.last_net_io = psutil.net_io_counters()
        self.last_net_time = time.time()
        self.protection_active = False
        self.last_protection_action = 0
        
        # Получение системной информации
        self.system_info = self._get_system_info()
        self.system_profile = self._identify_system_profile()
        
        logger.info(f"Системный профиль: {self.system_profile}")
    
    def _get_system_info(self) -> SystemInfo:
        """Получение информации о системе"""
        try:
            cpu_freq = psutil.cpu_freq()
            ram = psutil.virtual_memory()
            
            info = SystemInfo(
                hostname=socket.gethostname(),
                platform=f"{os.name} {sys.platform}",
                architecture=os.uname().machine if hasattr(os, 'uname') else "unknown",
                processor=psutil.cpu_info().get('brand_raw', 'Unknown') if hasattr(psutil, 'cpu_info') else "Unknown",
                cpu_count_physical=psutil.cpu_count(logical=False),
                cpu_count_logical=psutil.cpu_count(logical=True),
                cpu_freq_max=cpu_freq.max if cpu_freq else 0,
                ram_total=ram.total,
                ram_available=ram.available,
                boot_time=psutil.boot_time()
            )
            
            # GPU информация (если доступна)
            try:
                import subprocess
                result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                                       capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    info.gpu_info = [line.strip() for line in result.stdout.split('\n') if line.strip()]
            except:
                pass
            
            return info
        except Exception as e:
            logger.error(f"Ошибка получения информации о системе: {e}")
            return SystemInfo()
    
    def _identify_system_profile(self) -> str:
        """Идентификация профиля системы"""
        ram_gb = psutil.virtual_memory().total / (1024**3)
        cpu_cores = psutil.cpu_count(logical=True) or 1
        
        if ram_gb >= 32 and cpu_cores >= 8:
            return "high_performance"
        elif ram_gb >= 16 and cpu_cores >= 4:
            return "standard"
        elif ram_gb >= 8 and cpu_cores >= 2:
            return "basic"
        else:
            return "low_end"
    
    def start_monitoring(self, interval: int = Config.UPDATE_INTERVAL_NORMAL):
        """Запуск мониторинга"""
        if self.running:
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, args=(interval,), daemon=True)
        self.thread.start()
        logger.info("Мониторинг запущен")
    
    def stop_monitoring(self):
        """Остановка мониторинга"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
        logger.info("Мониторинг остановлен")
    
    def _monitor_loop(self, interval: int):
        """Основной цикл мониторинга"""
        while self.running:
            try:
                metrics = self._collect_metrics()
                self.metrics_history.append(metrics)
                self.metrics_updated.emit(metrics)
                
                # Проверка порогов и защита
                self._check_thresholds(metrics)
                
                # Обновление списка процессов
                if len(self.metrics_history) % 5 == 0:
                    processes = self._get_process_list()
                    self.process_list_updated.emit(processes)
                
                time.sleep(interval / 1000)
            except Exception as e:
                logger.error(f"Ошибка в цикле мониторинга: {e}")
                time.sleep(1)
    
    def _collect_metrics(self) -> ResourceMetrics:
        """Сбор метрик"""
        try:
            # CPU
            cpu_percent = psutil.cpu_percent(interval=0.1)
            cpu_per_core = psutil.cpu_percent(percpu=True)
            
            # RAM
            ram = psutil.virtual_memory()
            
            # Disk
            disk = psutil.disk_usage('/')
            
            # Network
            current_net = psutil.net_io_counters()
            current_time = time.time()
            time_delta = current_time - self.last_net_time
            
            if time_delta > 0:
                upload_speed = (current_net.bytes_sent - self.last_net_io.bytes_sent) / time_delta
                download_speed = (current_net.bytes_recv - self.last_net_io.bytes_recv) / time_delta
            else:
                upload_speed = 0
                download_speed = 0
            
            self.last_net_io = current_net
            self.last_net_time = current_time
            
            # Temperature (если доступно)
            temp_cpu = 0
            try:
                temps = psutil.sensors_temperatures()
                if temps:
                    for name, entries in temps.items():
                        for entry in entries:
                            if 'cpu' in entry.label.lower() or 'core' in entry.label.lower():
                                temp_cpu = entry.current
                                break
                        if temp_cpu > 0:
                            break
            except:
                pass
            
            # Battery
            battery_percent = 0
            power_plugged = False
            try:
                battery = psutil.sensors_battery()
                if battery:
                    battery_percent = battery.percent
                    power_plugged = battery.power_plugged
            except:
                pass
            
            metrics = ResourceMetrics(
                timestamp=time.time(),
                cpu_percent=cpu_percent,
                cpu_per_core=cpu_per_core,
                ram_percent=ram.percent,
                ram_used=ram.used,
                ram_available=ram.available,
                disk_percent=disk.percent,
                disk_used=disk.used,
                disk_free=disk.free,
                network_sent=current_net.bytes_sent,
                network_recv=current_net.bytes_recv,
                upload_speed=upload_speed,
                download_speed=download_speed,
                temp_cpu=temp_cpu,
                battery_percent=battery_percent,
                power_plugged=power_plugged
            )
            
            return metrics
        except Exception as e:
            logger.error(f"Ошибка сбора метрик: {e}")
            return ResourceMetrics(timestamp=time.time())
    
    def _get_process_list(self) -> List[ProcessInfo]:
        """Получение списка процессов"""
        processes = []
        try:
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 
                                             'memory_percent', 'memory_rss', 
                                             'status', 'username']):
                try:
                    pinfo = proc.info
                    processes.append(ProcessInfo(
                        pid=pinfo['pid'],
                        name=pinfo['name'] or 'Unknown',
                        cpu_percent=pinfo['cpu_percent'] or 0,
                        memory_percent=pinfo['memory_percent'] or 0,
                        memory_rss=pinfo['memory_rss'] or 0,
                        status=pinfo['status'] or 'unknown',
                        username=pinfo.get('username', 'Unknown')
                    ))
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            # Сортировка по использованию CPU
            processes.sort(key=lambda x: x.cpu_percent, reverse=True)
            self.process_cache = {p.pid: p for p in processes[:50]}
        except Exception as e:
            logger.error(f"Ошибка получения списка процессов: {e}")
        
        return processes[:50]
    
    def _check_thresholds(self, metrics: ResourceMetrics):
        """Проверка порогов и активация защиты"""
        alerts = []
        
        # CPU
        if metrics.cpu_percent >= Config.CPU_CRITICAL_THRESHOLD:
            alerts.append(Alert(
                timestamp=time.time(),
                severity="critical",
                category="CPU",
                message=f"Критическая нагрузка CPU: {metrics.cpu_percent:.1f}%"
            ))
            self._trigger_protection("cpu")
        elif metrics.cpu_percent >= Config.CPU_WARNING_THRESHOLD:
            alerts.append(Alert(
                timestamp=time.time(),
                severity="warning",
                category="CPU",
                message=f"Высокая нагрузка CPU: {metrics.cpu_percent:.1f}%"
            ))
        
        # RAM
        if metrics.ram_percent >= Config.RAM_CRITICAL_THRESHOLD:
            alerts.append(Alert(
                timestamp=time.time(),
                severity="critical",
                category="RAM",
                message=f"Критическое использование RAM: {metrics.ram_percent:.1f}%"
            ))
            self._trigger_protection("ram")
        elif metrics.ram_percent >= Config.RAM_WARNING_THRESHOLD:
            alerts.append(Alert(
                timestamp=time.time(),
                severity="warning",
                category="RAM",
                message=f"Высокое использование RAM: {metrics.ram_percent:.1f}%"
            ))
        
        # Disk
        if metrics.disk_percent >= Config.DISK_CRITICAL_THRESHOLD:
            alerts.append(Alert(
                timestamp=time.time(),
                severity="critical",
                category="DISK",
                message=f"Критически мало места на диске: {metrics.disk_percent:.1f}%"
            ))
        elif metrics.disk_percent >= Config.DISK_WARNING_THRESHOLD:
            alerts.append(Alert(
                timestamp=time.time(),
                severity="warning",
                category="DISK",
                message=f"Мало места на диске: {metrics.disk_percent:.1f}%"
            ))
        
        # Temperature
        if metrics.temp_cpu > 0:
            if metrics.temp_cpu >= Config.TEMP_CRITICAL_THRESHOLD:
                alerts.append(Alert(
                    timestamp=time.time(),
                    severity="critical",
                    category="TEMPERATURE",
                    message=f"Критическая температура CPU: {metrics.temp_cpu:.1f}°C"
                ))
                self._trigger_protection("temperature")
            elif metrics.temp_cpu >= Config.TEMP_WARNING_THRESHOLD:
                alerts.append(Alert(
                    timestamp=time.time(),
                    severity="warning",
                    category="TEMPERATURE",
                    message=f"Высокая температура CPU: {metrics.temp_cpu:.1f}°C"
                ))
        
        # Отправка оповещений
        for alert in alerts:
            self.alert_triggered.emit(alert)
    
    def _trigger_protection(self, reason: str):
        """Активация защиты от зависания"""
        if not Config.AUTO_PROTECTION_ENABLED:
            return
        
        current_time = time.time()
        if current_time - self.last_protection_action < Config.PROTECTION_COOLDOWN:
            return
        
        self.protection_active = True
        self.last_protection_action = current_time
        
        logger.warning(f"Активирована защита от {reason}")
        
        # Действия защиты
        try:
            if reason == "cpu":
                # Снижение приоритета фоновых процессов
                self._reduce_background_priority()
            elif reason == "ram":
                # Очистка кэша
                self._clear_memory_cache()
            elif reason == "temperature":
                # Уведомление о необходимости охлаждения
                logger.critical("Требуется немедленное охлаждение системы!")
        except Exception as e:
            logger.error(f"Ошибка при выполнении защиты: {e}")
        
        self.protection_active = False
    
    def _reduce_background_priority(self):
        """Снижение приоритета фоновых процессов"""
        try:
            for proc in psutil.process_iter(['name', 'nice']):
                try:
                    if proc.info['nice'] == 0:  # Нормальный приоритет
                        # Не трогаем системные процессы
                        if proc.info['name'] not in ['system', 'idle', 'init']:
                            proc.nice(10)  # Снижаем приоритет
                except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
                    continue
        except Exception as e:
            logger.error(f"Ошибка снижения приоритета: {e}")
    
    def _clear_memory_cache(self):
        """Очистка кэша памяти (Linux)"""
        try:
            if sys.platform.startswith('linux'):
                os.system('sync; echo 3 > /proc/sys/vm/drop_caches 2>/dev/null')
                logger.info("Кэш памяти очищен")
        except Exception as e:
            logger.error(f"Ошибка очистки кэша: {e}")
    
    def get_uptime(self) -> str:
        """Получение времени работы системы"""
        uptime_seconds = time.time() - self.system_info.boot_time
        days = int(uptime_seconds // 86400)
        hours = int((uptime_seconds % 86400) // 3600)
        minutes = int((uptime_seconds % 3600) // 60)
        return f"{days}д {hours}ч {minutes}м"
    
    def get_wifi_info(self) -> Dict:
        """Получение информации о WiFi"""
        wifi_info = {
            "connected": False,
            "ssid": "",
            "signal_strength": 0,
            "ip_address": "",
            "gateway": "",
            "dns_servers": []
        }
        
        try:
            # Получение сетевых интерфейсов
            interfaces = psutil.net_if_addrs()
            stats = psutil.net_if_stats()
            
            for iface_name, addrs in interfaces.items():
                if 'wifi' in iface_name.lower() or 'wlan' in iface_name.lower() or 'wireless' in iface_name.lower():
                    if stats.get(iface_name) and stats[iface_name].isup:
                        wifi_info["connected"] = True
                        wifi_info["interface"] = iface_name
                        
                        for addr in addrs:
                            if addr.family == psutil.AF_LINK:
                                wifi_info["mac_address"] = addr.address
                            elif addr.family == socket.AF_INET if 'socket' in dir() else 2:
                                wifi_info["ip_address"] = addr.address
                        
                        # Попытка получить SSID (требует специальных прав)
                        try:
                            import subprocess
                            result = subprocess.run(
                                ['iwgetid', '-r'],
                                capture_output=True, text=True, timeout=5
                            )
                            if result.returncode == 0:
                                wifi_info["ssid"] = result.stdout.strip()
                        except:
                            pass
                        break
        except Exception as e:
            logger.error(f"Ошибка получения WiFi информации: {e}")
        
        return wifi_info
    
    def get_internet_status(self) -> Dict:
        """Проверка статуса интернета"""
        status = {
            "online": False,
            "latency_ms": 0,
            "dns_resolved": False,
            "http_accessible": False
        }
        
        try:
            # Проверка DNS
            try:
                import socket
                socket.create_connection(("8.8.8.8", 53), timeout=2)
                status["dns_resolved"] = True
            except:
                pass
            
            # Проверка HTTP
            try:
                response = requests.get("https://www.google.com", timeout=3)
                if response.status_code == 200:
                    status["http_accessible"] = True
                    status["online"] = True
            except:
                pass
            
            # Измерение пинга
            try:
                import subprocess
                result = subprocess.run(
                    ['ping', '-c', '1', '8.8.8.8'],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    # Парсинг времени пинга
                    import re
                    match = re.search(r'time=(\d+\.?\d*)', result.stdout)
                    if match:
                        status["latency_ms"] = float(match.group(1))
            except:
                pass
        except Exception as e:
            logger.error(f"Ошибка проверки интернета: {e}")
        
        return status


# Импорт socket для WiFi
import socket


# ============================================================================
# ГЕНЕРАТОР ОТЧЕТОВ
# ============================================================================

class ReportGenerator:
    """Генератор отчетов"""
    
    def __init__(self, db_manager: DatabaseManager, system_monitor: SystemMonitor):
        self.db = db_manager
        self.monitor = system_monitor
        self.report_dir = "reports"
        os.makedirs(self.report_dir, exist_ok=True)
    
    def generate_full_report(self, format: str = "json") -> str:
        """Генерация полного отчета"""
        report = {
            "generated_at": datetime.datetime.now().isoformat(),
            "system_info": asdict(self.monitor.system_info),
            "current_metrics": asdict(self.monitor._collect_metrics()),
            "uptime": self.monitor.get_uptime(),
            "wifi_info": self.monitor.get_wifi_info(),
            "internet_status": self.monitor.get_internet_status(),
            "recent_alerts": self.db.get_alerts(7),
            "performance_score": self._calculate_performance_score(),
            "recommendations": self._generate_recommendations()
        }
        
        filename = f"report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.{format}"
        filepath = os.path.join(self.report_dir, filename)
        
        if format == "json":
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
        elif format == "html":
            self._save_html_report(report, filepath + ".html")
        
        logger.info(f"Отчет сохранен: {filepath}")
        return filepath
    
    def _calculate_performance_score(self) -> float:
        """Расчет общего балла производительности"""
        metrics = self.monitor._collect_metrics()
        
        score = 100
        
        # Штрафы за высокую нагрузку
        score -= max(0, metrics.cpu_percent - 50) * 0.5
        score -= max(0, metrics.ram_percent - 60) * 0.5
        score -= max(0, metrics.disk_percent - 70) * 0.3
        
        # Бонусы за хорошую температуру
        if metrics.temp_cpu > 0 and metrics.temp_cpu < 60:
            score += 5
        
        return max(0, min(100, score))
    
    def _generate_recommendations(self) -> List[str]:
        """Генерация рекомендаций"""
        recommendations = []
        metrics = self.monitor._collect_metrics()
        
        if metrics.cpu_percent > 70:
            recommendations.append("Рассмотрите обновление процессора или оптимизацию рабочих процессов")
        
        if metrics.ram_percent > 75:
            recommendations.append("Рекомендуется добавить оперативную память")
        
        if metrics.disk_percent > 80:
            recommendations.append("Очистите диск или установите дополнительный накопитель")
        
        if metrics.temp_cpu > 65:
            recommendations.append("Проверьте систему охлаждения, замените термопасту")
        
        uptime_seconds = time.time() - self.monitor.system_info.boot_time
        if uptime_seconds > 7 * 24 * 3600:  # Более 7 дней
            recommendations.append("Рекомендуется перезагрузить систему для сброса кэшей")
        
        return recommendations
    
    def _save_html_report(self, report: Dict, filepath: str):
        """Сохранение HTML отчета"""
        html = f"""
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <title>Отчет PC Analyzer Pro</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 30px; }}
        .metric {{ display: inline-block; margin: 10px; padding: 15px; background: #ecf0f1; border-radius: 5px; min-width: 150px; }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #2980b9; }}
        .metric-label {{ font-size: 12px; color: #7f8c8d; }}
        .alert {{ padding: 10px; margin: 5px 0; border-radius: 4px; }}
        .alert-critical {{ background: #ffebee; border-left: 4px solid #f44336; }}
        .alert-warning {{ background: #fff3e0; border-left: 4px solid #ff9800; }}
        .alert-info {{ background: #e3f2fd; border-left: 4px solid #2196f3; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 15px; }}
        th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #3498db; color: white; }}
        .score {{ font-size: 48px; font-weight: bold; color: #27ae60; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🖥️ PC Analyzer Pro - Отчет</h1>
        <p>Дата генерации: {report['generated_at']}</p>
        
        <h2>📊 Общая оценка производительности</h2>
        <div class="score">{report['performance_score']:.1f}/100</div>
        
        <h2>💻 Информация о системе</h2>
        <table>
            <tr><th>Параметр</th><th>Значение</th></tr>
            <tr><td>Хост</td><td>{report['system_info'].get('hostname', 'N/A')}</td></tr>
            <tr><td>Платформа</td><td>{report['system_info'].get('platform', 'N/A')}</td></tr>
            <tr><td>Процессор</td><td>{report['system_info'].get('processor', 'N/A')}</td></tr>
            <tr><td>Ядра CPU</td><td>{report['system_info'].get('cpu_count_physical', 0)} физ / {report['system_info'].get('cpu_count_logical', 0)} лог</td></tr>
            <tr><td>RAM</td><td>{report['system_info'].get('ram_total', 0) / (1024**3):.1f} GB</td></tr>
            <tr><td>Время работы</td><td>{report['uptime']}</td></tr>
        </table>
        
        <h2>📈 Текущие метрики</h2>
        <div class="metric">
            <div class="metric-value">{report['current_metrics'].get('cpu_percent', 0):.1f}%</div>
            <div class="metric-label">CPU Load</div>
        </div>
        <div class="metric">
            <div class="metric-value">{report['current_metrics'].get('ram_percent', 0):.1f}%</div>
            <div class="metric-label">RAM Usage</div>
        </div>
        <div class="metric">
            <div class="metric-value">{report['current_metrics'].get('disk_percent', 0):.1f}%</div>
            <div class="metric-label">Disk Usage</div>
        </div>
        <div class="metric">
            <div class="metric-value">{report['current_metrics'].get('temp_cpu', 0):.1f}°C</div>
            <div class="metric-label">CPU Temp</div>
        </div>
        
        <h2>🌐 Сеть</h2>
        <table>
            <tr><th>Параметр</th><th>Значение</th></tr>
            <tr><td>Интернет</td><td>{'✅ Подключен' if report['internet_status'].get('online') else '❌ Отключен'}</td></tr>
            <tr><td>WiFi</td><td>{'✅ Подключен' if report['wifi_info'].get('connected') else '❌ Не подключен'}</td></tr>
            <tr><td>SSID</td><td>{report['wifi_info'].get('ssid', 'N/A')}</td></tr>
            <tr><td>Ping</td><td>{report['internet_status'].get('latency_ms', 0):.1f} ms</td></tr>
        </table>
        
        <h2>⚠️ Последние предупреждения</h2>
        {self._format_alerts_html(report.get('recent_alerts', []))}
        
        <h2>💡 Рекомендации</h2>
        <ul>
            {''.join(f'<li>{rec}</li>' for rec in report.get('recommendations', []))}
        </ul>
    </div>
</body>
</html>
"""
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html)
    
    def _format_alerts_html(self, alerts: List[Dict]) -> str:
        """Форматирование предупреждений для HTML"""
        if not alerts:
            return "<p>Нет предупреждений за последний период</p>"
        
        html = ""
        for alert in alerts[:10]:  # Последние 10
            severity = alert.get('severity', 'info')
            html += f"""
            <div class="alert alert-{severity}">
                <strong>[{alert.get('category', 'N/A')}]</strong> 
                {alert.get('message', 'N/A')}
                <br><small>{datetime.datetime.fromtimestamp(alert.get('timestamp', 0)).strftime('%Y-%m-%d %H:%M')}</small>
            </div>
            """
        return html
    
    def export_to_csv(self, days: int = 7) -> str:
        """Экспорт данных в CSV"""
        metrics = self.db.get_recent_metrics(1000)
        if not metrics:
            return ""
        
        df = pd.DataFrame(metrics)
        filename = f"metrics_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        filepath = os.path.join(self.report_dir, filename)
        
        df.to_csv(filepath, index=False, encoding='utf-8')
        logger.info(f"CSV экспортирован: {filepath}")
        return filepath
    
    def create_chart(self, metric_type: str = "cpu", hours: int = 24) -> str:
        """Создание графика"""
        metrics = self.db.get_recent_metrics(int(hours * 60))  # Примерно точек
        if not metrics:
            return ""
        
        df = pd.DataFrame(metrics)
        df['time'] = pd.to_datetime(df['timestamp'], unit='s')
        
        plt.figure(figsize=(12, 6))
        
        if metric_type == "cpu":
            plt.plot(df['time'], df['cpu_percent'], label='CPU %', color='#e74c3c', linewidth=2)
            plt.title('Нагрузка CPU', fontsize=16)
            plt.ylabel('Проценты')
        elif metric_type == "ram":
            plt.plot(df['time'], df['ram_percent'], label='RAM %', color='#3498db', linewidth=2)
            plt.title('Использование RAM', fontsize=16)
            plt.ylabel('Проценты')
        elif metric_type == "all":
            plt.plot(df['time'], df['cpu_percent'], label='CPU %', color='#e74c3c')
            plt.plot(df['time'], df['ram_percent'], label='RAM %', color='#3498db')
            plt.plot(df['time'], df['disk_percent'], label='Disk %', color='#2ecc71')
            plt.title('Все метрики', fontsize=16)
            plt.legend()
            plt.ylabel('Проценты')
        
        plt.xlabel('Время')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        filename = f"chart_{metric_type}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        filepath = os.path.join(self.report_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"График сохранен: {filepath}")
        return filepath


# ============================================================================
# GUI ПРИЛОЖЕНИЕ
# ============================================================================

if PYQT_AVAILABLE:
    
    class DashboardWidget(QWidget):
        """Виджет главной панели"""
        
        def __init__(self, monitor: SystemMonitor):
            super().__init__()
            self.monitor = monitor
            self.init_ui()
        
        def init_ui(self):
            layout = QVBoxLayout()
            layout.setSpacing(15)
            
            # Заголовок
            title = QLabel("🖥️ PC Analyzer Pro - Главная панель")
            title.setFont(QFont("Arial", 18, QFont.Bold))
            title.setAlignment(Qt.AlignCenter)
            layout.addWidget(title)
            
            # Основные метрики
            metrics_layout = QGridLayout()
            metrics_layout.setSpacing(15)
            
            # CPU
            self.cpu_label = QLabel("CPU")
            self.cpu_label.setAlignment(Qt.AlignCenter)
            self.cpu_label.setFont(QFont("Arial", 12, QFont.Bold))
            self.cpu_progress = QProgressBar()
            self.cpu_progress.setRange(0, 100)
            self.cpu_progress.setValue(0)
            self.cpu_progress.setFormat("%p%")
            self.cpu_value = QLabel("0.0%")
            self.cpu_value.setAlignment(Qt.AlignCenter)
            
            cpu_widget = QWidget()
            cpu_layout = QVBoxLayout(cpu_widget)
            cpu_layout.addWidget(self.cpu_label)
            cpu_layout.addWidget(self.cpu_progress)
            cpu_layout.addWidget(self.cpu_value)
            metrics_layout.addWidget(cpu_widget, 0, 0)
            
            # RAM
            self.ram_label = QLabel("RAM")
            self.ram_label.setAlignment(Qt.AlignCenter)
            self.ram_label.setFont(QFont("Arial", 12, QFont.Bold))
            self.ram_progress = QProgressBar()
            self.ram_progress.setRange(0, 100)
            self.ram_progress.setValue(0)
            self.ram_progress.setFormat("%p%")
            self.ram_value = QLabel("0.0%")
            self.ram_value.setAlignment(Qt.AlignCenter)
            
            ram_widget = QWidget()
            ram_layout = QVBoxLayout(ram_widget)
            ram_layout.addWidget(self.ram_label)
            ram_layout.addWidget(self.ram_progress)
            ram_layout.addWidget(self.ram_value)
            metrics_layout.addWidget(ram_widget, 0, 1)
            
            # Disk
            self.disk_label = QLabel("DISK")
            self.disk_label.setAlignment(Qt.AlignCenter)
            self.disk_label.setFont(QFont("Arial", 12, QFont.Bold))
            self.disk_progress = QProgressBar()
            self.disk_progress.setRange(0, 100)
            self.disk_progress.setValue(0)
            self.disk_progress.setFormat("%p%")
            self.disk_value = QLabel("0.0%")
            self.disk_value.setAlignment(Qt.AlignCenter)
            
            disk_widget = QWidget()
            disk_layout = QVBoxLayout(disk_widget)
            disk_layout.addWidget(self.disk_label)
            disk_layout.addWidget(self.disk_progress)
            disk_layout.addWidget(self.disk_value)
            metrics_layout.addWidget(disk_widget, 0, 2)
            
            # Temperature
            self.temp_label = QLabel("TEMP")
            self.temp_label.setAlignment(Qt.AlignCenter)
            self.temp_label.setFont(QFont("Arial", 12, QFont.Bold))
            self.temp_lcd = QLCDNumber()
            self.temp_lcd.setDigitCount(5)
            self.temp_lcd.display(0.0)
            self.temp_unit = QLabel("°C")
            self.temp_unit.setAlignment(Qt.AlignCenter)
            
            temp_widget = QWidget()
            temp_layout = QVBoxLayout(temp_widget)
            temp_layout.addWidget(self.temp_label)
            temp_layout.addWidget(self.temp_lcd)
            temp_layout.addWidget(self.temp_unit)
            metrics_layout.addWidget(temp_widget, 0, 3)
            
            layout.addLayout(metrics_layout)
            
            # Дополнительная информация
            info_layout = QHBoxLayout()
            
            self.uptime_label = QLabel("Время работы: 0д 0ч 0м")
            self.uptime_label.setFont(QFont("Arial", 10))
            info_layout.addWidget(self.uptime_label)
            
            self.internet_label = QLabel("🌐 Интернет: --")
            self.internet_label.setFont(QFont("Arial", 10))
            info_layout.addWidget(self.internet_label)
            
            self.wifi_label = QLabel("📶 WiFi: --")
            self.wifi_label.setFont(QFont("Arial", 10))
            info_layout.addWidget(self.wifi_label)
            
            layout.addLayout(info_layout)
            
            # Графики (заглушка)
            self.chart_placeholder = QLabel("📊 Графики будут отображаться здесь")
            self.chart_placeholder.setAlignment(Qt.AlignCenter)
            self.chart_placeholder.setMinimumHeight(200)
            self.chart_placeholder.setStyleSheet("background: #ecf0f1; border-radius: 5px;")
            layout.addWidget(self.chart_placeholder)
            
            self.setLayout(layout)
        
        def update_metrics(self, metrics: ResourceMetrics):
            """Обновление метрик"""
            self.cpu_progress.setValue(int(metrics.cpu_percent))
            self.cpu_value.setText(f"{metrics.cpu_percent:.1f}%")
            
            self.ram_progress.setValue(int(metrics.ram_percent))
            self.ram_value.setText(f"{metrics.ram_percent:.1f}%")
            
            self.disk_progress.setValue(int(metrics.disk_percent))
            self.disk_value.setText(f"{metrics.disk_percent:.1f}%")
            
            if metrics.temp_cpu > 0:
                self.temp_lcd.display(metrics.temp_cpu)
            else:
                self.temp_lcd.display(0.0)
            
            # Цвета для прогресс баров
            self._update_progress_color(self.cpu_progress, metrics.cpu_percent)
            self._update_progress_color(self.ram_progress, metrics.ram_percent)
            self._update_progress_color(self.disk_progress, metrics.disk_percent)
            
            # Время работы
            self.uptime_label.setText(f"⏱️ Время работы: {self.monitor.get_uptime()}")
            
            # Интернет
            internet_status = self.monitor.get_internet_status()
            if internet_status["online"]:
                self.internet_label.setText(f"🌐 Интернет: ✅ ({internet_status['latency_ms']:.0f}ms)")
            else:
                self.internet_label.setText("🌐 Интернет: ❌")
            
            # WiFi
            wifi_info = self.monitor.get_wifi_info()
            if wifi_info["connected"]:
                self.wifi_label.setText(f"📶 WiFi: ✅ {wifi_info.get('ssid', 'Unknown')}")
            else:
                self.wifi_label.setText("📶 WiFi: ❌")
        
        def _update_progress_color(self, progress_bar: QProgressBar, value: float):
            """Обновление цвета прогресс бара"""
            if value < 60:
                color = "#27ae60"  # Зеленый
            elif value < 80:
                color = "#f39c12"  # Оранжевый
            else:
                color = "#e74c3c"  # Красный
            
            progress_bar.setStyleSheet(f"""
                QProgressBar {{
                    border: 2px solid #bdc3c7;
                    border-radius: 5px;
                    text-align: center;
                    font-weight: bold;
                }}
                QProgressBar::chunk {{
                    background-color: {color};
                    border-radius: 3px;
                }}
            """)
    
    
    class ProcessesWidget(QWidget):
        """Виджет процессов"""
        
        def __init__(self):
            super().__init__()
            self.init_ui()
        
        def init_ui(self):
            layout = QVBoxLayout()
            
            title = QLabel("📋 Процессы")
            title.setFont(QFont("Arial", 16, QFont.Bold))
            layout.addWidget(title)
            
            # Таблица процессов
            self.table = QTableWidget()
            self.table.setColumnCount(5)
            self.table.setHorizontalHeaderLabels(["PID", "Имя", "CPU %", "RAM %", "Статус"])
            self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
            self.table.setAlternatingRowColors(True)
            self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
            layout.addWidget(self.table)
            
            self.setLayout(layout)
        
        def update_processes(self, processes: List[ProcessInfo]):
            """Обновление списка процессов"""
            self.table.setRowCount(0)
            
            for proc in processes[:30]:  # Топ 30
                row = self.table.rowCount()
                self.table.insertRow(row)
                self.table.setItem(row, 0, QTableWidgetItem(str(proc.pid)))
                self.table.setItem(row, 1, QTableWidgetItem(proc.name[:30]))
                self.table.setItem(row, 2, QTableWidgetItem(f"{proc.cpu_percent:.1f}"))
                self.table.setItem(row, 3, QTableWidgetItem(f"{proc.memory_percent:.1f}"))
                self.table.setItem(row, 4, QTableWidgetItem(proc.status))
    
    
    class AlertsWidget(QWidget):
        """Виджет предупреждений"""
        
        def __init__(self, db_manager: DatabaseManager):
            super().__init__()
            self.db = db_manager
            self.init_ui()
        
        def init_ui(self):
            layout = QVBoxLayout()
            
            title = QLabel("⚠️ Предупреждения")
            title.setFont(QFont("Arial", 16, QFont.Bold))
            layout.addWidget(title)
            
            # Список предупреждений
            self.alert_list = QListWidget()
            layout.addWidget(self.alert_list)
            
            # Кнопки
            btn_layout = QHBoxLayout()
            
            refresh_btn = QPushButton("🔄 Обновить")
            refresh_btn.clicked.connect(self.load_alerts)
            btn_layout.addWidget(refresh_btn)
            
            clear_btn = QPushButton("🗑️ Очистить")
            clear_btn.clicked.connect(self.clear_alerts)
            btn_layout.addWidget(clear_btn)
            
            layout.addLayout(btn_layout)
            
            self.setLayout(layout)
            self.load_alerts()
        
        def load_alerts(self):
            """Загрузка предупреждений"""
            self.alert_list.clear()
            alerts = self.db.get_alerts(7)
            
            for alert in alerts:
                item_text = f"[{alert['severity'].upper()}] {alert['category']}: {alert['message']}"
                item = QListWidgetItem(item_text)
                
                if alert['severity'] == 'critical':
                    item.setBackground(QColor("#ffebee"))
                elif alert['severity'] == 'warning':
                    item.setBackground(QColor("#fff3e0"))
                
                self.alert_list.addItem(item)
        
        def clear_alerts(self):
            """Очистка старых предупреждений"""
            reply = QMessageBox.question(
                self, "Подтверждение",
                "Удалить все старые предупреждения?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                # В реальной реализации здесь был бы SQL DELETE
                self.load_alerts()
                QMessageBox.information(self, "Готово", "Предупреждения очищены")
    
    
    class SettingsWidget(QWidget):
        """Виджет настроек"""
        
        def __init__(self, neural_network: PCNeuralNetwork):
            super().__init__()
            self.nn = neural_network
            self.init_ui()
        
        def init_ui(self):
            layout = QVBoxLayout()
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            
            container = QWidget()
            container_layout = QVBoxLayout(container)
            
            # Защита
            protection_group = QGroupBox("🛡️ Автозащита")
            protection_layout = QVBoxLayout()
            
            self.protection_checkbox = QCheckBox("Включить автозащиту от перегрузки")
            self.protection_checkbox.setChecked(Config.AUTO_PROTECTION_ENABLED)
            protection_layout.addWidget(self.protection_checkbox)
            
            self.cooldown_spin = QSpinBox()
            self.cooldown_spin.setRange(10, 600)
            self.cooldown_spin.setValue(Config.PROTECTION_COOLDOWN)
            self.cooldown_spin.setSuffix(" сек")
            protection_layout.addWidget(QLabel("Кулдаун между действиями защиты:"))
            protection_layout.addWidget(self.cooldown_spin)
            
            protection_group.setLayout(protection_layout)
            container_layout.addWidget(protection_group)
            
            # Пороги
            thresholds_group = QGroupBox("📊 Пороги предупреждений")
            thresholds_layout = QFormLayout()
            
            self.cpu_warning_spin = QSpinBox()
            self.cpu_warning_spin.setRange(50, 94)
            self.cpu_warning_spin.setValue(Config.CPU_WARNING_THRESHOLD)
            self.cpu_warning_spin.setSuffix("%")
            thresholds_layout.addRow("CPU Warning:", self.cpu_warning_spin)
            
            self.cpu_critical_spin = QSpinBox()
            self.cpu_critical_spin.setRange(95, 100)
            self.cpu_critical_spin.setValue(Config.CPU_CRITICAL_THRESHOLD)
            self.cpu_critical_spin.setSuffix("%")
            thresholds_layout.addRow("CPU Critical:", self.cpu_critical_spin)
            
            self.ram_warning_spin = QSpinBox()
            self.ram_warning_spin.setRange(60, 94)
            self.ram_warning_spin.setValue(Config.RAM_WARNING_THRESHOLD)
            self.ram_warning_spin.setSuffix("%")
            thresholds_layout.addRow("RAM Warning:", self.ram_warning_spin)
            
            self.ram_critical_spin = QSpinBox()
            self.ram_critical_spin.setRange(95, 100)
            self.ram_critical_spin.setValue(Config.RAM_CRITICAL_THRESHOLD)
            self.ram_critical_spin.setSuffix("%")
            thresholds_layout.addRow("RAM Critical:", self.ram_critical_spin)
            
            thresholds_group.setLayout(thresholds_layout)
            container_layout.addWidget(thresholds_group)
            
            # Нейросеть
            nn_group = QGroupBox("🧠 Нейросеть")
            nn_layout = QVBoxLayout()
            
            self.nn_stats_label = QLabel("Статистика нейросети:")
            nn_layout.addWidget(self.nn_stats_label)
            
            refresh_nn_btn = QPushButton("🔄 Обновить статистику")
            refresh_nn_btn.clicked.connect(self.update_nn_stats)
            nn_layout.addWidget(refresh_nn_btn)
            
            retrain_btn = QPushButton("🎯 Переобучить нейросеть")
            retrain_btn.clicked.connect(self.retrain_nn)
            nn_layout.addWidget(retrain_btn)
            
            nn_group.setLayout(nn_layout)
            container_layout.addWidget(nn_group)
            
            # Сохранение
            save_btn = QPushButton("💾 Сохранить настройки")
            save_btn.clicked.connect(self.save_settings)
            container_layout.addWidget(save_btn)
            
            scroll.setWidget(container)
            layout.addWidget(scroll)
            self.setLayout(layout)
            
            self.update_nn_stats()
        
        def update_nn_stats(self):
            """Обновление статистики нейросети"""
            stats = self.nn.get_stats()
            self.nn_stats_label.setText(
                f"Образцов: {stats['training_samples']}\n"
                f"Обучена: {'✅' if stats['is_trained'] else '❌'}\n"
                f"Аномалий обнаружено: {stats['anomalies_detected']}\n"
                f"Модель: {stats['model_type']}"
            )
        
        def retrain_nn(self):
            """Переобучение нейросети"""
            self.nn.retrain()
            self.update_nn_stats()
            QMessageBox.information(self, "Готово", "Нейросеть переобучена")
        
        def save_settings(self):
            """Сохранение настроек"""
            Config.AUTO_PROTECTION_ENABLED = self.protection_checkbox.isChecked()
            Config.PROTECTION_COOLDOWN = self.cooldown_spin.value()
            Config.CPU_WARNING_THRESHOLD = self.cpu_warning_spin.value()
            Config.CPU_CRITICAL_THRESHOLD = self.cpu_critical_spin.value()
            Config.RAM_WARNING_THRESHOLD = self.ram_warning_spin.value()
            Config.RAM_CRITICAL_THRESHOLD = self.ram_critical_spin.value()
            
            QMessageBox.information(self, "Готово", "Настройки сохранены")
    
    
    class ReportsWidget(QWidget):
        """Виджет отчетов"""
        
        def __init__(self, report_generator: ReportGenerator):
            super().__init__()
            self.generator = report_generator
            self.init_ui()
        
        def init_ui(self):
            layout = QVBoxLayout()
            
            title = QLabel("📄 Отчеты")
            title.setFont(QFont("Arial", 16, QFont.Bold))
            layout.addWidget(title)
            
            # Кнопки генерации
            btn_layout = QHBoxLayout()
            
            json_btn = QPushButton("📊 JSON отчет")
            json_btn.clicked.connect(lambda: self.generate_report("json"))
            btn_layout.addWidget(json_btn)
            
            html_btn = QPushButton("🌐 HTML отчет")
            html_btn.clicked.connect(lambda: self.generate_report("html"))
            btn_layout.addWidget(html_btn)
            
            csv_btn = QPushButton("📈 CSV экспорт")
            csv_btn.clicked.connect(self.export_csv)
            btn_layout.addWidget(csv_btn)
            
            chart_btn = QPushButton("📉 График")
            chart_btn.clicked.connect(self.create_chart)
            btn_layout.addWidget(chart_btn)
            
            layout.addLayout(btn_layout)
            
            # Список отчетов
            self.reports_list = QListWidget()
            layout.addWidget(self.reports_list)
            
            self.refresh_reports()
            self.setLayout(layout)
        
        def generate_report(self, format: str):
            """Генерация отчета"""
            filepath = self.generator.generate_full_report(format)
            QMessageBox.information(self, "Готово", f"Отчет сохранен:\n{filepath}")
            self.refresh_reports()
        
        def export_csv(self):
            """Экспорт CSV"""
            filepath = self.generator.export_to_csv()
            if filepath:
                QMessageBox.information(self, "Готово", f"CSV экспортирован:\n{filepath}")
            else:
                QMessageBox.warning(self, "Ошибка", "Нет данных для экспорта")
        
        def create_chart(self):
            """Создание графика"""
            filepath = self.generator.create_chart("all")
            if filepath:
                QMessageBox.information(self, "Готово", f"График сохранен:\n{filepath}")
            else:
                QMessageBox.warning(self, "Ошибка", "Нет данных для графика")
        
        def refresh_reports(self):
            """Обновление списка отчетов"""
            self.reports_list.clear()
            try:
                for filename in os.listdir(self.generator.report_dir):
                    if filename.endswith(('.json', '.html', '.csv', '.png')):
                        item = QListWidgetItem(filename)
                        self.reports_list.addItem(item)
            except:
                pass
    
    
    class MainWindow(QMainWindow):
        """Главное окно приложения"""
        
        def __init__(self):
            super().__init__()
            
            # Инициализация компонентов
            self.db = DatabaseManager()
            self.monitor = SystemMonitor()
            self.nn = PCNeuralNetwork()
            self.report_generator = ReportGenerator(self.db, self.monitor)
            
            self.init_ui()
            self.setup_connections()
            self.start_monitoring()
            
            logger.info("Приложение запущено")
        
        def init_ui(self):
            """Инициализация UI"""
            self.setWindowTitle(f"{Config.APP_NAME} v{Config.VERSION}")
            self.setMinimumSize(1200, 800)
            
            # Центральное виджет
            central_widget = QWidget()
            self.setCentralWidget(central_widget)
            
            main_layout = QVBoxLayout()
            central_widget.setLayout(main_layout)
            
            # Табы
            self.tabs = QTabWidget()
            
            # Главная панель
            self.dashboard = DashboardWidget(self.monitor)
            self.tabs.addTab(self.dashboard, "🏠 Главная")
            
            # Процессы
            self.processes_widget = ProcessesWidget()
            self.tabs.addTab(self.processes_widget, "📋 Процессы")
            
            # Предупреждения
            self.alerts_widget = AlertsWidget(self.db)
            self.tabs.addTab(self.alerts_widget, "⚠️ Предупреждения")
            
            # Отчеты
            self.reports_widget = ReportsWidget(self.report_generator)
            self.tabs.addTab(self.reports_widget, "📄 Отчеты")
            
            # Настройки
            self.settings_widget = SettingsWidget(self.nn)
            self.tabs.addTab(self.settings_widget, "⚙️ Настройки")
            
            main_layout.addWidget(self.tabs)
            
            # Статус бар
            self.statusBar = QStatusBar()
            self.setStatusBar(self.statusBar)
            self.statusBar.showMessage("Готов к работе")
            
            # Тулбар
            toolbar = QToolBar("Главная панель")
            self.addToolBar(toolbar)
            
            start_action = QAction("▶️ Старт", self)
            start_action.triggered.connect(self.start_monitoring)
            toolbar.addAction(start_action)
            
            stop_action = QAction("⏹️ Стоп", self)
            stop_action.triggered.connect(self.stop_monitoring)
            toolbar.addAction(stop_action)
            
            toolbar.addSeparator()
            
            report_action = QAction("📊 Создать отчет", self)
            report_action.triggered.connect(lambda: self.report_generator.generate_full_report("html"))
            toolbar.addAction(report_action)
            
            # Таймер обновления
            self.update_timer = QTimer()
            self.update_timer.timeout.connect(self.update_ui)
            self.update_timer.start(1000)
        
        def setup_connections(self):
            """Настройка соединений сигналов"""
            self.monitor.metrics_updated.connect(self.on_metrics_updated)
            self.monitor.alert_triggered.connect(self.on_alert_triggered)
            self.monitor.process_list_updated.connect(self.on_processes_updated)
        
        def start_monitoring(self):
            """Запуск мониторинга"""
            self.monitor.start_monitoring()
            self.statusBar.showMessage("Мониторинг запущен")
        
        def stop_monitoring(self):
            """Остановка мониторинга"""
            self.monitor.stop_monitoring()
            self.statusBar.showMessage("Мониторинг остановлен")
        
        def on_metrics_updated(self, metrics: ResourceMetrics):
            """Обработка обновления метрик"""
            self.dashboard.update_metrics(metrics)
            
            # Добавление в базу данных (каждую минуту)
            if int(metrics.timestamp) % 60 == 0:
                self.db.save_metric(metrics)
            
            # Обучение нейросети
            self.nn.add_sample(metrics)
        
        def on_alert_triggered(self, alert: Alert):
            """Обработка предупреждения"""
            self.db.save_alert(alert)
            self.alerts_widget.load_alerts()
            
            # Уведомление
            if alert.severity == "critical":
                QMessageBox.critical(self, "Критическое предупреждение", alert.message)
            elif alert.severity == "warning":
                self.statusBar.showMessage(f"⚠️ {alert.message}", 5000)
        
        def on_processes_updated(self, processes: List[ProcessInfo]):
            """Обработка обновления процессов"""
            self.processes_widget.update_processes(processes)
        
        def update_ui(self):
            """Периодическое обновление UI"""
            uptime = self.monitor.get_uptime()
            self.statusBar.showMessage(f"⏱️ {uptime} | 🧠 Нейросеть: {len(self.nn.training_data)} образцов")
        
        def closeEvent(self, event):
            """Обработка закрытия"""
            reply = QMessageBox.question(
                self, "Выход",
                "Остановить мониторинг и выйти?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                self.monitor.stop_monitoring()
                event.accept()
            else:
                event.ignore()
    
    
    def main_gui():
        """Запуск GUI приложения"""
        app = QApplication(sys.argv)
        app.setStyle('Fusion')
        
        # Настройка палитры
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(245, 245, 245))
        palette.setColor(QPalette.WindowText, QColor(0, 0, 0))
        palette.setColor(QPalette.Base, QColor(255, 255, 255))
        palette.setColor(QPalette.AlternateBase, QColor(240, 240, 240))
        palette.setColor(QPalette.ToolTipBase, QColor(255, 255, 255))
        palette.setColor(QPalette.ToolTipText, QColor(0, 0, 0))
        palette.setColor(QPalette.Text, QColor(0, 0, 0))
        palette.setColor(QPalette.Button, QColor(240, 240, 240))
        palette.setColor(QPalette.ButtonText, QColor(0, 0, 0))
        palette.setColor(QPalette.Highlight, QColor(52, 152, 219))
        palette.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
        app.setPalette(palette)
        
        window = MainWindow()
        window.show()
        
        sys.exit(app.exec_())

else:
    def main_console():
        """Консольный режим"""
        print("=" * 60)
        print("PC Analyzer Pro - Консольный режим")
        print("=" * 60)
        
        db = DatabaseManager()
        monitor = SystemMonitor()
        nn = PCNeuralNetwork()
        report_gen = ReportGenerator(db, monitor)
        
        print(f"\nСистема: {monitor.system_info.hostname}")
        print(f"Процессор: {monitor.system_info.processor}")
        print(f"RAM: {monitor.system_info.ram_total / (1024**3):.1f} GB")
        print(f"Профиль: {monitor.system_profile}")
        print("\nЗапуск мониторинга... (Ctrl+C для остановки)\n")
        
        monitor.start_monitoring()
        
        try:
            while True:
                metrics = next(iter(monitor.metrics_history), None)
                if metrics:
                    print(f"\rCPU: {metrics.cpu_percent:5.1f}% | "
                          f"RAM: {metrics.ram_percent:5.1f}% | "
                          f"DISK: {metrics.disk_percent:5.1f}% | "
                          f"TEMP: {metrics.temp_cpu:5.1f}°C | "
                          f"NN: {len(nn.training_data)} samples", end="")
                    
                    # Добавление в нейросеть
                    nn.add_sample(metrics)
                    
                    # Проверка аномалий
                    is_anomaly, score = nn.detect_anomaly(metrics)
                    if is_anomaly:
                        print(f" ⚠️ ANOMALY DETECTED! Score: {score:.3f}")
                    
                    # Прогноз перегрузки
                    overload_risk = nn.predict_overload(metrics)
                    if overload_risk > 80:
                        print(f" 🔴 HIGH OVERLOAD RISK: {overload_risk:.1f}%")
                    
                    # Сохранение метрик
                    if int(metrics.timestamp) % 60 == 0:
                        db.save_metric(metrics)
                
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n\nОстановка мониторинга...")
            monitor.stop_monitoring()
            
            # Генерация отчета
            print("\nГенерация отчета...")
            report_path = report_gen.generate_full_report("json")
            print(f"Отчет сохранен: {report_path}")
            
            # Статистика нейросети
            nn_stats = nn.get_stats()
            print(f"\nСтатистика нейросети:")
            print(f"  Образцов: {nn_stats['training_samples']}")
            print(f"  Обучена: {nn_stats['is_trained']}")
            print(f"  Аномалий: {nn_stats['anomalies_detected']}")
            
            print("\nДо свидания!")


# ============================================================================
# ТОЧКА ВХОДА
# ============================================================================

def main():
    """Точка входа в приложение"""
    print("=" * 70)
    print("  PC Analyzer Pro v1.0.0")
    print("  Профессиональная система мониторинга и защиты ПК")
    print("  С встроенной нейросетью для анализа и прогнозирования")
    print("=" * 70)
    print()
    
    if PYQT_AVAILABLE:
        print("✅ Графический интерфейс доступен")
        print("Запуск GUI режима...\n")
        main_gui()
    else:
        print("⚠️ Графический интерфейс недоступен")
        print("Запуск консольного режима...\n")
        main_console()


if __name__ == "__main__":
    main()

# src/monitor_app.py

import sys
import random
from pathlib import Path
from io import BytesIO

import cv2
import torch
import time
import datetime
import matplotlib.pyplot as plt

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QFrame, QSizePolicy,
    QStackedWidget, QGridLayout, QFileDialog, QListWidget,
    QTableWidget, QTableWidgetItem, QHeaderView, QSpacerItem
)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap, QPalette, QColor, QFont

from model import get_model
from dataset import MEAN, STD, IMG_SIZE
from collections import deque
from collections import Counter

CKPT   = Path(__file__).parents[1] / "models" / "best_cnn.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def collect_image_paths(root: Path, exts=None) -> list[Path]:
    if exts is None:
        exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    paths = [p for p in root.rglob("*") if p.suffix in exts]
    random.shuffle(paths)
    return paths

# ───────── VideoPanel ─────────────────────────────────────────────────────────
class VideoPanel(QFrame):
    """
    Центральная зона: режимы Folder / Camera / Uploaded,
    кнопка Mode переключает режим, Load… загружает файлы.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.Box)
        self.setStyleSheet("background-color: #222;")
        self.setLineWidth(2)

        # ── Кнопки слева ────────────────────────────────────────
        self.btn_mode = QPushButton("Mode: Folder")
        self.btn_load = QPushButton("Load…")
        self.btn_reset    = QPushButton("Reset")

        for btn in (self.btn_mode, self.btn_load, self.btn_reset):
            btn.setFixedHeight(50)
            btn.setStyleSheet("color:white; font-size:16px; background:#444;")
            btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        
        self.btn_mode.clicked.connect(self.toggle_mode)
        self.btn_load.clicked.connect(self.load_images)
        self.btn_reset.clicked.connect(self.reset_panel)

        self.recent_results = deque(maxlen=10)
        self.recent_list    = QListWidget()
        self.recent_list.setFixedHeight(120)
        self.recent_list.setStyleSheet("background:#333; color:white;")
        self.info_lbl       = QLabel("")
        self.info_lbl.setStyleSheet("color:white;")

        left_layout = QVBoxLayout()
        left_layout.addWidget(self.btn_mode)
        left_layout.addWidget(self.btn_load)
        left_layout.addWidget(self.btn_reset)
        left_layout.addWidget(self.recent_list)
        left_layout.addWidget(self.info_lbl)
        left_layout.addStretch()
        left_widget = QWidget()
        left_widget.setLayout(left_layout)
        left_widget.setFixedWidth(160)

        # ── Видео-окно ───────────────────────────────────────────
        self.img_lbl = QLabel()
        self.img_lbl.setAlignment(Qt.AlignCenter)            # type: ignore[attr-defined]
        self.img_lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        video_cont = QWidget()
        vh = QHBoxLayout(video_cont)
        vh.setContentsMargins(5,5,5,5)
        vh.addWidget(self.img_lbl)

        # подпись
        self.pred_lbl = QLabel("Stopped")
        self.pred_lbl.setAlignment(Qt.AlignCenter)           # type: ignore[attr-defined]
        self.pred_lbl.setStyleSheet(
            "color:white;"
            "background-color:rgba(0,0,0,0.6);"
            "padding:6px; font-size:20px; border-radius:4px;"
        )

        # ── Сетка компоновки ─────────────────────────────────────
        grid = QGridLayout(self)
        grid.setContentsMargins(0,0,0,0)
        grid.addWidget(left_widget, 0, 0, 2, 1)
        grid.addWidget(video_cont,   0, 1)
        grid.addWidget(self.pred_lbl,1, 1)

        # ── Источники ────────────────────────────────────────────
        # 1) Папка
        data_root = Path(__file__).parents[1] / "data" / "archive" / "images"
        # Сначала получаем список папок-имен в алфавитном порядке
        self.class_dirs = sorted(
            [p.name for p in data_root.iterdir() if p.is_dir()]
        )
        # Потом перебираем все файлы в этих папках
        self.samples = []
        for class_name in self.class_dirs:
            folder = data_root / class_name
            for img_path in folder.glob("*"):
                if img_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    self.samples.append(img_path)
        random.shuffle(self.samples)
        self.sample_idx = 0

        # 2) Камера
        self.cap = cv2.VideoCapture(0)

        # 3) Загруженные файлы
        self.uploaded    = []   # ГАРАНТИРОВАННО список, а не None
        self.upload_idx  = 0

        # Режим: 0=Folder, 1=Camera, 2=Uploaded
        self.mode = 0

        # ── Модель ───────────────────────────────────────────────
        self.model = get_model(8, pretrained=False).to(DEVICE)
        self.model.load_state_dict(torch.load(CKPT, map_location=DEVICE))
        self.model.eval()

        self.last_pred = None
        self.last_prob = None

    def toggle_mode(self):
        self.mode = (self.mode + 1) % 3
        names = ["Folder", "Camera", "Uploaded"]
        self.btn_mode.setText(f"Mode: {names[self.mode]}")
        # При переключении режима сбросим историю, чтобы не путать:
        self.recent_results.clear()
        self.recent_list.clear()
        # Если возвращаемся в Folder, обновим info-лейбл
        if self.mode == 0:
            self.info_lbl.clear()

    def load_images(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select gesture images", "", "Images (*.jpg *.jpeg *.png)"
        )
        if not files:
            return
        # Теперь self.uploaded — список путей, а не None
        self.uploaded   = [Path(f) for f in files]
        self.upload_idx = 0
        self.mode = 2
        self.btn_mode.setText("Mode: Uploaded")
        self.info_lbl.setText(f"Loaded: {len(self.uploaded)}")
        self.recent_results.clear()
        self.recent_list.clear()

    def reset_panel(self):
        self.mode = 0
        self.sample_idx = 0
        self.upload_idx = 0
        self.uploaded.clear()
        self.recent_results.clear()
        self.recent_list.clear()
        self.info_lbl.clear()
        self.pred_lbl.setText("Stopped")

    def next_frame(self):
        # MODE 0: Folder
        if self.mode == 0:
            if not self.samples:
                return None, None
            p = self.samples[self.sample_idx]
            self.sample_idx = (self.sample_idx + 1) % len(self.samples)
            frame = cv2.imread(str(p))
            return (frame, self._predict(frame)) if frame is not None else (None, None)

        # MODE 1: Camera
        if self.mode == 1:
            if self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    return frame, self._predict(frame)
            return None, None

        # MODE 2: Uploaded
        if self.mode == 2:
            if self.upload_idx >= len(self.uploaded):
                # все загруженные кадры уже показаны
                return None, None
            p = self.uploaded[self.upload_idx]
            self.upload_idx += 1
            frame = cv2.imread(str(p))
            return (frame, self._predict(frame)) if frame is not None else (None, None)

        return None, None

    def _predict(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE)).astype("float32")/255.0
        img = (img - MEAN) / STD
        t = torch.tensor(img).permute(2,0,1).unsqueeze(0).float().to(DEVICE)
        with torch.no_grad():
            probs = torch.softmax(self.model(t), dim=1)[0]
        pred = int(torch.argmax(probs))
        return pred, float(probs[pred])

    def update_frame(self):
        frame, info = self.next_frame()
        if frame is None or info is None:
            return
        pred, prob = info
        self.last_pred, self.last_prob = pred, prob

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb.shape
        qimg = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg).scaled(
            self.img_lbl.size(),
            Qt.KeepAspectRatio,        # type: ignore[attr-defined]
            Qt.SmoothTransformation     # type: ignore[attr-defined]
        )

        self.img_lbl.setPixmap(pix)
        class_name = self.class_dirs[pred]
        self.pred_lbl.setText(f"{class_name} ({pred})  {prob:.0%}")

        self.recent_results.append(pred)
        self.recent_list.clear()
        for idx, cls_idx in enumerate(self.recent_results, start=1):
            name = self.class_dirs[cls_idx]
            self.recent_list.addItem(f"{idx}. {name} ({cls_idx})")
        # обновляем инфо о загрузке
        if self.mode == 2:
            self.info_lbl.setText(f"Loaded: {len(self.uploaded)}")

# ───────── StatsGraphWidget ───────────────────────────────────────────────────
class StatsGraphWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.counter = Counter()
        self.plot_lbl = QLabel()
        self.plot_lbl.setAlignment(Qt.AlignCenter)             # type: ignore[attr-defined]
        lay = QVBoxLayout(self)
        lay.addWidget(self.plot_lbl)

    def update_stats(self, pred:int):
        self.counter[pred] += 1
        classes = list(self.counter.keys())
        counts  = [self.counter[c] for c in classes]

        fig, ax = plt.subplots(figsize=(4,3), facecolor="#222")
        ax.bar(classes, counts, color="cyan")
        ax.set_facecolor("#333")
        ax.tick_params(colors="white", labelsize=8)
        ax.set_xlabel("Class", color="white")
        ax.set_ylabel("Count", color="white")
        ax.spines["bottom"].set_color("white")
        ax.spines["left"].set_color("white")

        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=100,
                    bbox_inches="tight", facecolor="#222")
        plt.close(fig)
        buf.seek(0)

        pix = QPixmap()
        pix.loadFromData(buf.getvalue())
        self.plot_lbl.setPixmap(pix.scaled(
            self.plot_lbl.size(),
            Qt.KeepAspectRatio,        # type: ignore[attr-defined]
            Qt.SmoothTransformation     # type: ignore[attr-defined]
        ))
    
    def clear_graph(self):
        """Сбросить всю статистику и убрать картинку."""
        self.counter.clear()
        self.plot_lbl.clear()

class ProbLineWidget(QWidget):
    def __init__(self, window_size: int = 50, parent=None):
        super().__init__(parent)
        from collections import deque
        self.window_size = window_size
        self.probs_deque = deque(maxlen=self.window_size)

        self.plot_lbl = QLabel()
        self.plot_lbl.setAlignment(Qt.AlignCenter)  # type: ignore[attr-defined]
        self.plot_lbl.setStyleSheet("background-color: #333;")
        # Разрешаем QLabel расширяться и заполнять всё пространство
        self.plot_lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)  # type: ignore[attr-defined]
        self.plot_lbl.setScaledContents(True)  # type: ignore[attr-defined]

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)  # убрали все отступы
        layout.setSpacing(0)
        layout.addWidget(self.plot_lbl)

    def update_prob(self, prob: float):
        self.probs_deque.append(prob if prob is not None else 0.0)
        xs = list(range(len(self.probs_deque)))
        ys = list(self.probs_deque)

        # ─── Строим график без лишних полей, с адекватным DPI ─────────────────────────
        fig, ax = plt.subplots(figsize=(4, 2), facecolor="#222", dpi=120)
        ax.plot(xs, ys, color="lime", linewidth=2)
        ax.set_facecolor("#333")
        ax.set_xlabel("Frame", color="white", fontsize=8)
        ax.set_ylabel("Prob", color="white", fontsize=8)
        ax.set_ylim(0, 1.2)
        ax.tick_params(colors="white", labelsize=7)
        for spine in ax.spines.values():
            spine.set_color("white")
        # Убираем все поля вокруг графика, чтобы занять максимум пространства
        plt.subplots_adjust(left=0.08, right=0.98, top=0.95, bottom=0.20)

        buf = BytesIO()
        fig.savefig(buf, format="png", facecolor="#222", dpi=120)
        plt.close(fig)
        buf.seek(0)

        pix = QPixmap()
        pix.loadFromData(buf.getvalue())
        # Масштабируем в размер QLabel без сохранения aspect ratio, чтобы fill-эффект
        self.plot_lbl.setPixmap(
            pix.scaled(
                self.plot_lbl.size(),
                Qt.IgnoreAspectRatio,     # type: ignore[attr-defined]
                Qt.SmoothTransformation    # type: ignore[attr-defined]
            )
        )

    def clear_graph(self):
        self.probs_deque.clear()
        self.plot_lbl.clear()


# ─── PLCGraphWidget ──────────────────────────────────────────────────────────────
class PLCGraphWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        # ── 1) «Лампочка» + подпись «State: …» ───────────────────────────────────────
        self.indicator = QLabel()
        self.indicator.setFixedSize(20, 20)  # type: ignore[attr-defined]
        self.indicator.setStyleSheet("background-color: gray; border-radius: 10px;")

        self.indicator_lbl = QLabel("State: IDLE")
        self.indicator_lbl.setStyleSheet("color: white; font-size: 14px;")
        self.indicator_lbl.setAlignment(Qt.AlignLeft)  # type: ignore[attr-defined]

        lamp_layout = QHBoxLayout()
        lamp_layout.setContentsMargins(0, 0, 0, 0)  # убрали отступы
        lamp_layout.setSpacing(5)                   # небольшой промежуток
        lamp_layout.addWidget(self.indicator)       # type: ignore[attr-defined]
        lamp_layout.addWidget(self.indicator_lbl)   # type: ignore[attr-defined]
        lamp_layout.addStretch()                    # type: ignore[attr-defined]

        lamp_widget = QWidget()
        lamp_widget.setLayout(lamp_layout)

        # ── 2) Таблица «OK / ERROR / IDLE» ──────────────────────────────────────────
        self.state_table = QTableWidget(3, 3)
        self.state_table.setHorizontalHeaderLabels(["State", "Count", "Last Time"])  # type: ignore[attr-defined]
        self.state_table.verticalHeader().setVisible(False)  # type: ignore[attr-defined]
        self.state_table.setEditTriggers(QTableWidget.NoEditTriggers)  # type: ignore[attr-defined]
        self.state_table.setSelectionMode(QTableWidget.NoSelection)  # type: ignore[attr-defined]

        # Компактный шрифт, чтобы таблица адекватно отображалась
        state_font = QFont("Consolas", 10)
        self.state_table.setFont(state_font)  # type: ignore[attr-defined]
        self.state_table.setStyleSheet(
            "QTableWidget { background-color: #333; color: white; gridline-color: #555; }"
            "QHeaderView::section { background-color: #444; color: white; font-weight: bold; }"
        )

        # Высота строк: 20px, заголовок 25px, рамка ≈2px
        self.state_table.verticalHeader().setDefaultSectionSize(20)  # type: ignore[attr-defined]
        self.state_table.setFixedHeight(20 * 3 + 25 + 2)  # type: ignore[attr-defined]

        # Отключаем скроллбары — таблица будет фиксированной высоты, но ширина может расширяться
        self.state_table.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)    # type: ignore[attr-defined]
        self.state_table.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)  # type: ignore[attr-defined]

        # Растягиваем колонки на всю ширину таблицы
        header0 = self.state_table.horizontalHeader()  # type: ignore[attr-defined]
        header0.setSectionResizeMode(QHeaderView.Stretch)  # type: ignore[attr-defined]

        for row, state in enumerate(["OK", "ERROR", "IDLE"]):
            item_state = QTableWidgetItem(state)  # type: ignore[attr-defined]
            item_state.setFlags(Qt.ItemIsEnabled)  # type: ignore[attr-defined]
            self.state_table.setItem(row, 0, item_state)  # type: ignore[attr-defined]

            item_cnt = QTableWidgetItem("0")  # type: ignore[attr-defined]
            item_cnt.setFlags(Qt.ItemIsEnabled)  # type: ignore[attr-defined]
            self.state_table.setItem(row, 1, item_cnt)  # type: ignore[attr-defined]

            item_time = QTableWidgetItem("")  # type: ignore[attr-defined]
            item_time.setFlags(Qt.ItemIsEnabled)  # type: ignore[attr-defined]
            self.state_table.setItem(row, 2, item_time)  # type: ignore[attr-defined]

        # ── 3) Таблица «Last / Mean / Min / Max» ────────────────────────────────────
        self.prob_stats_table = QTableWidget(1, 4)
        self.prob_stats_table.setHorizontalHeaderLabels(["Last", "Mean", "Min", "Max"])  # type: ignore[attr-defined]
        self.prob_stats_table.verticalHeader().setVisible(False)  # type: ignore[attr-defined]
        self.prob_stats_table.setEditTriggers(QTableWidget.NoEditTriggers)  # type: ignore[attr-defined]
        self.prob_stats_table.setSelectionMode(QTableWidget.NoSelection)  # type: ignore[attr-defined]

        prob_font = QFont("Consolas", 10)
        self.prob_stats_table.setFont(prob_font)  # type: ignore[attr-defined]
        self.prob_stats_table.setStyleSheet(
            "QTableWidget { background-color: #333; color: white; gridline-color: #555; }"
            "QHeaderView::section { background-color: #444; color: white; font-weight: bold; }"
        )

        # Высота: одна строка 20px + заголовок 25px + рамка ≈2px
        self.prob_stats_table.verticalHeader().setDefaultSectionSize(20)  # type: ignore[attr-defined]
        self.prob_stats_table.setFixedHeight(20 + 25 + 2)  # type: ignore[attr-defined]

        self.prob_stats_table.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)    # type: ignore[attr-defined]
        self.prob_stats_table.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)  # type: ignore[attr-defined]

        header1 = self.prob_stats_table.horizontalHeader()  # type: ignore[attr-defined]
        header1.setSectionResizeMode(QHeaderView.Stretch)  # type: ignore[attr-defined]

        for col in range(4):
            item = QTableWidgetItem("0.00")  # type: ignore[attr-defined]
            item.setFlags(Qt.ItemIsEnabled)  # type: ignore[attr-defined]
            self.prob_stats_table.setItem(0, col, item)  # type: ignore[attr-defined]

        # ── 4) «Малый» график вероятности ─────────────────────────────────────────────
        self.prob_widget = ProbLineWidget(window_size=50)  # type: ignore[attr-defined]
        self.prob_widget.setStyleSheet("background-color: #222;")
        self.prob_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)  # type: ignore[attr-defined]

        # ── 5) Текстовый лог «Сообщения» ─────────────────────────────────────────────
        self.log_list = QListWidget()
        log_font = QFont("Consolas", 11)
        self.log_list.setFont(log_font)  # type: ignore[attr-defined]
        self.log_list.setStyleSheet("QListWidget { background-color: #111; color: white; }")
        self.log_list.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)  # type: ignore[attr-defined]
        self.log_list.setFixedHeight(140)  # type: ignore[attr-defined]
        self.log_list.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)   # type: ignore[attr-defined]
        self.log_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)  # type: ignore[attr-defined]

        # ─────────────────────── Собираем «Top Widget» ─────────────────────────────────
        top_widget = QWidget()
        top_layout = QHBoxLayout(top_widget)
        top_layout.setContentsMargins(0, 0, 0, 0)  # убрали внешние отступы
        top_layout.setSpacing(5)                   # небольшой промежуток между колонками

        # ▪ Левая часть: лампочка + таблицы (вертикально), без пустых областей
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)  # убрали отступы вокруг
        left_layout.setSpacing(50)                   # небольшой промежуток между элементами

        left_layout.addWidget(lamp_widget)           # лампочка + подпись
        left_layout.addWidget(self.state_table)      # таблица OK/ERROR/IDLE
        left_layout.addWidget(self.prob_stats_table) # таблица Last/Mean/Min/Max
        left_layout.addStretch()                     # чтобы таблицы «прижавались» к верху

        # ▪ Правая часть: график ProbLineWidget (заполняет всё пространство)
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)  # убрали отступы вокруг
        right_layout.setSpacing(0)
        right_layout.addWidget(self.prob_widget)    # type: ignore[attr-defined]

        # Распределяем ширину: левая часть 2, правая часть 5 (можно корректировать)
        top_layout.addWidget(left_widget, 2)   # type: ignore[attr-defined]
        top_layout.addWidget(right_widget, 5)  # type: ignore[attr-defined]

        # ─────────────────── Собираем основной «Main Layout» ──────────────────────────
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)  # убрали внешние отступы
        main_layout.setSpacing(0)

        # 1) Верхняя область: таблицы + график
        main_layout.addWidget(top_widget)  # type: ignore[attr-defined]
        # 2) Нижняя область: лог
        main_layout.addWidget(self.log_list)  # type: ignore[attr-defined]

        self.setLayout(main_layout)  # type: ignore[attr-defined]

        # ───────────────── Инициализация вспомогательных данных ─────────────────────
        self.counters = {"OK": 0, "ERROR": 0, "IDLE": 0}
        self.log_buffer = deque(maxlen=100)

    def add_entry(self, pred: int | None, prob: float, fps: float) -> None:
        """
        Каждое обновление кадра:
         1) Определяем состояние и меняем лампочку + подпись
         2) Обновляем state_table (Count + Last Time)
         3) Передаём prob в ProbLineWidget (update_prob): график теперь заполняет всю правую часть
         4) Пересчитываем Last/Mean/Min/Max и пишем в prob_stats_table
         5) Записываем строку в log_buffer и обновляем QListWidget
        """
        now = time.strftime("%H:%M:%S", time.localtime())

        # ─── 1) Определяем состояние лампочки ───────────────────────────────────
        if pred is None or prob < 0.5 or fps < 0.8:
            state = "ERROR"
            self.indicator.setStyleSheet("background-color: red; border-radius: 10px;")
            self.indicator_lbl.setText("State: ERROR")
        else:
            state = "OK"
            self.indicator.setStyleSheet("background-color: green; border-radius: 10px;")
            self.indicator_lbl.setText("State: OK")

        # ─── 2) Обновляем таблицу «OK / ERROR / IDLE» ────────────────────────────
        self.counters[state] += 1
        for row in range(self.state_table.rowCount()):
            if self.state_table.item(row, 0).text() == state:  # type: ignore[attr-defined]
                self.state_table.item(row, 1).setText(str(self.counters[state]))  # type: ignore[attr-defined]
                self.state_table.item(row, 2).setText(now)  # type: ignore[attr-defined]
                break

        # ─── 3) Обновляем график вероятности ─────────────────────────────────────
        self.prob_widget.update_prob(prob)  # type: ignore[attr-defined]

        # ─── 4) Пересчитываем Last/Mean/Min/Max ──────────────────────────────────
        try:
            buffer = self.prob_widget.probs_deque  # type: ignore[attr-defined]
        except AttributeError:
            buffer = deque([prob], maxlen=self.prob_widget.window_size)

        last_val = buffer[-1] if buffer else 0.0
        mean_val = sum(buffer) / len(buffer) if buffer else 0.0
        min_val  = min(buffer) if buffer else 0.0
        max_val  = max(buffer) if buffer else 0.0

        stats = [last_val, mean_val, min_val, max_val]
        for col, value in enumerate(stats):
            item = self.prob_stats_table.item(0, col)  # type: ignore[attr-defined]
            item.setText(f"{value:.2f}")  # type: ignore[attr-defined]

        # ─── 5) Записываем сообщение в лог ───────────────────────────────────────
        log_entry = f"[{now}] Class {pred}  Prob: {prob:.2f}  FPS: {fps:.2f}  → {state}"
        self.log_buffer.append(log_entry)

        self.log_list.clear()  # type: ignore[attr-defined]
        for line in self.log_buffer:
            self.log_list.addItem(line)  # type: ignore[attr-defined]

    def clear_graph(self):
        """
        Сброс PLC-интерфейса:
         1) Лампочка → серый, подпись «State: IDLE»
         2) Обнуляем таблицу OK/ERROR/IDLE
         3) Очищаем ProbLineWidget и таблицу Last/Mean/Min/Max
         4) Очищаем лог
        """
        # 1) Сброс лампочки
        self.indicator.setStyleSheet("background-color: gray; border-radius: 10px;")
        self.indicator_lbl.setText("State: IDLE")

        # 2) Сброс таблицы OK/ERROR/IDLE
        self.counters = {"OK": 0, "ERROR": 0, "IDLE": 0}
        for row in range(self.state_table.rowCount()):
            self.state_table.item(row, 1).setText("0")  # type: ignore[attr-defined]
            self.state_table.item(row, 2).setText("")   # type: ignore[attr-defined]

        # 3) Сброс графика и таблицы Last/Mean/Min/Max
        self.prob_widget.clear_graph()  # type: ignore[attr-defined]
        for col in range(4):
            self.prob_stats_table.item(0, col).setText("0.00")  # type: ignore[attr-defined]

        # 4) Сброс логов
        self.log_buffer.clear()
        self.log_list.clear()  # type: ignore[attr-defined]


class FPSGraphWidget(QWidget):
    """
    Виджет для графика FPS (кадров в секунду) over time + текстовая сводка.
    Хранит последние window_size значений FPS в deque.
    """
    def __init__(self, window_size: int = 50, parent=None):
        super().__init__(parent)
        from collections import deque
        self.window_size = window_size
        self.fps_deque = deque(maxlen=self.window_size)

        # QLabel для рисунка графика
        self.plot_lbl = QLabel()
        self.plot_lbl.setAlignment(Qt.AlignCenter)  # type: ignore[attr-defined]
        self.plot_lbl.setStyleSheet("background-color: #333;")

        # QLabel для текстовой сводки под графиком
        self.summary_lbl = QLabel()
        self.summary_lbl.setStyleSheet("color:white; font-size:12px;")
        self.summary_lbl.setAlignment(Qt.AlignLeft)  # type: ignore[attr-defined]

        lay = QVBoxLayout(self)
        lay.setContentsMargins(5,5,5,5)
        lay.addWidget(self.plot_lbl, stretch=1)
        lay.addWidget(self.summary_lbl, stretch=0)

    def update_fps(self, fps: float):
        """
        Добавляет новое значение FPS, строит bar-график и обновляет текстовую сводку.
        """
        self.fps_deque.append(fps)
        xs = list(range(len(self.fps_deque)))
        ys = list(self.fps_deque)

        # 1) Рисуем столбчатый график вместо линии, чтобы маленькие значения
        #    было легче различить на тёмном фоне
        fig, ax = plt.subplots(figsize=(4,2.0), facecolor="#222")
        ax.bar(xs, ys, color="orange", width=0.8)
        ax.set_facecolor("#333")
        ax.set_xlabel("Frame", color="white", fontsize=8)
        ax.set_ylabel("FPS", color="white", fontsize=8)

        # Динамические границы Y: если все FPS < 5, то верхняя граница маленькая,
        # иначе — чуть выше максимального. Так столбцы будут заметнее.
        max_fps = max(ys) if ys else 1
        top = max(5, max_fps + 2)
        ax.set_ylim(0, top)

        ax.tick_params(colors="white", labelsize=7)
        for spine in ax.spines.values():
            spine.set_color("white")

        # 2) Рисуем в буфер
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight", facecolor="#222")
        plt.close(fig)
        buf.seek(0)

        # 3) Показываем в QLabel
        pix = QPixmap()
        pix.loadFromData(buf.getvalue())
        self.plot_lbl.setPixmap(
            pix.scaled(
                self.plot_lbl.size(),
                Qt.KeepAspectRatio,         # type: ignore[attr-defined]
                Qt.SmoothTransformation      # type: ignore[attr-defined]
            )
        )

        # 4) Обновляем текстовую сводку: текущий, средний, min, max
        if ys:
            curr = ys[-1]
            mean = sum(ys) / len(ys)
            mn   = min(ys)
            mx   = max(ys)
            summary = (
                f"Curr FPS: {curr:.1f}    "
                f"Mean: {mean:.1f}    "
                f"Min: {mn:.1f}    "
                f"Max: {mx:.1f}"
            )
        else:
            summary = "No data"
        self.summary_lbl.setText(summary)

    def clear_graph(self):
        """
        Полностью сбрасывает накопленные FPS-данные,
        очищает график и текстовую сводку.
        """
        self.fps_deque.clear()
        self.plot_lbl.clear()
        self.summary_lbl.clear()


# ───────── MainApp ────────────────────────────────────────────────────────────
class MainApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hand-Gesture SCADA Monitor")
        self.resize(900, 650)

        # страницы
        self.video_w = VideoPanel()
        self.stats_w = StatsGraphWidget()
        self.plc_w   = PLCGraphWidget()
        self.fps_w   = FPSGraphWidget()

        self.video_w.btn_reset.clicked.connect(self.reset_all)

        # стек
        self.stack = QStackedWidget()
        self.stack.addWidget(self.video_w)    # idx=0 → Video
        self.stack.addWidget(self.stats_w)    # idx=1 → Stats
        self.stack.addWidget(self.plc_w)      # idx=2 → PLC
        self.stack.addWidget(self.fps_w)      # idx=4 → FPS

        # табы
        self.btn_video = QPushButton("Video")
        self.btn_stats = QPushButton("Stats")
        self.btn_plc   = QPushButton("PLC")
        self.btn_fps   = QPushButton("FPS")
        
        for i, btn in enumerate((self.btn_video, self.btn_stats, self.btn_plc, self.btn_fps)):
            btn.setStyleSheet("color:white; background:#444; font-size:18px; padding:10px;")
            btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            btn.clicked.connect(lambda _, idx=i: self.stack.setCurrentIndex(idx))

        tabs = QHBoxLayout()
        tabs.addWidget(self.btn_video)
        tabs.addWidget(self.btn_stats)
        tabs.addWidget(self.btn_plc)
        tabs.addWidget(self.btn_fps)

        # Start/Stop
        self.timer = QTimer(self)
        self.timer.setInterval(400)
        self.timer.timeout.connect(self._on_tick)

        self.btn_start = QPushButton("Start")
        self.btn_stop  = QPushButton("Stop")
        for b in (self.btn_start, self.btn_stop):
            b.setStyleSheet("color:white; background:#555; font-size:18px; padding:10px;")
            b.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.btn_start.clicked.connect(self.timer.start)
        self.btn_stop.clicked.connect(self.stop)

        ctrl = QHBoxLayout()
        ctrl.addWidget(self.btn_start)
        ctrl.addWidget(self.btn_stop)

        # центральный layout
        central = QWidget()
        lay = QVBoxLayout(central)
        lay.setContentsMargins(5,5,5,5)
        lay.addLayout(ctrl)
        lay.addWidget(self.stack, stretch=1)
        lay.addLayout(tabs)
        self.setCentralWidget(central)

        self.prev_time = None

    def _on_tick(self):
        current_time = time.perf_counter()
        self.video_w.update_frame()

        if self.prev_time is not None:
            dt = current_time - self.prev_time
            fps = 1.0 / dt if dt > 0 else 0.0
            self.fps_w.update_fps(fps)
        else:
            fps = 0.0

        # 4) Сохраняем время для следующего тика
        self.prev_time = current_time

        # 2) сразу сохраняем в статистику и PLC, если есть предсказание
        pred = self.video_w.last_pred
        prob = self.video_w.last_prob
        self.plc_w.add_entry(pred, prob if prob is not None else 0.0, fps)
        if pred is not None:
            self.stats_w.update_stats(pred)

        # 3) только после этого проверяем, не исчерпаны ли загруженные файлы
        if self.video_w.mode == 2 and self.video_w.upload_idx >= len(self.video_w.uploaded):
            self.stop()

        # если только что завершили показ загруженных картинок — останавливаемся
        if self.video_w.mode == 2 and self.video_w.upload_idx >= len(self.video_w.uploaded):
            self.stop()
            return

        # иначе обновляем статистику и PLC как обычно
        pred = self.video_w.last_pred
        if pred is not None:
            self.stats_w.update_stats(pred)

    def reset_all(self):
        # 1) Останавливаем таймер
        self.stop()

        # 2) Очищаем Stats и PLC
        self.stats_w.clear_graph()
        self.plc_w.clear_graph()

        self.fps_w.clear_graph()

        # 5) Возвращаем таб на Video
        self.stack.setCurrentIndex(0)

    def stop(self):
        self.timer.stop()
        self.video_w.pred_lbl.setText("Stopped")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    pal = app.palette()
    pal.setColor(QPalette.Window, QColor("#111"))
    app.setPalette(pal)

    win = MainApp()
    win.show()
    sys.exit(app.exec())

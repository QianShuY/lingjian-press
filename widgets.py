"""Shared desktop controls and the image teaching canvas."""

from PySide6.QtCore import QPoint, QRect, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QFrame, QLabel, QPushButton, QVBoxLayout, QWidget


STYLE = """
QMainWindow, QWidget#page { background: #edf2f5; color: #172b38; }
QWidget { color: #172b38; }
QFrame#header { background: #102933; }
QLabel#brand { color: #ffffff; font-size: 21px; font-weight: 700; }
QLabel#brandSub { color: #b0cacd; font-size: 12px; }
QLabel#mode { color: #ffe5a1; background: #453b22; border: 1px solid #695a31;
              border-radius: 6px; padding: 6px 11px; font-weight: 700; }
QFrame#card { background: #ffffff; border: 1px solid #d9e3e8; border-radius: 10px; }
QFrame#line { background: #e7edf1; }
QLabel#title { font-size: 26px; font-weight: 700; }
QLabel#section { font-size: 17px; font-weight: 700; }
QLabel#muted { color: #465f6d; }
QLabel#hint { color: #426575; background: #eaf3f6; border-radius: 6px; padding: 9px; }
QLabel#value { font-size: 25px; font-weight: 700; }
QLabel#metric { font-size: 13px; color: #465f6d; }
QLabel#status { border-radius: 7px; padding: 10px; font-size: 21px; font-weight: 700; }
QLabel#status[tone="idle"] { color: #254d60; background: #eaf2f6; }
QLabel#status[tone="ok"] { color: #106241; background: #e3f4e9; }
QLabel#status[tone="ng"] { color: #a32022; background: #fce9e9; }
QLabel#status[tone="review"] { color: #825310; background: #fff3d8; }
QLabel#status[tone="fault"] { color: #873123; background: #fcece4; }
QPushButton { background: #ffffff; border: 1px solid #bfced7; border-radius: 7px; padding: 8px 14px;
              min-height: 28px; font-weight: 600; }
QPushButton:hover { background: #edf5f6; border-color: #70979e; }
QPushButton:focus, QLineEdit:focus, QComboBox:focus, QDoubleSpinBox:focus { border: 2px solid #087e86; }
QPushButton:disabled { color: #60727c; background: #e7ecef; border-color: #d6e0e5; }
QPushButton#primary { background: #086c73; border-color: #086c73; color: #ffffff; }
QPushButton#primary:hover { background: #075b62; }
QPushButton#primary:disabled { background: #d6e4e6; border-color: #d6e4e6; color: #597279; }
QPushButton#tab { border: 1px solid transparent; color: #bcd2d5; background: transparent;
                  min-width: 80px; font-size: 14px; }
QPushButton#tab:checked { color: #ffffff; background: #245358; border-color: #376668; }
QPushButton#tab:hover, QPushButton#tab:focus { border-color: #82b9bd; color: #ffffff; }
QComboBox, QLineEdit, QDoubleSpinBox { background: white; border: 1px solid #bdcdd5;
                                    padding: 7px 9px; border-radius: 6px; min-height: 25px; }
QListWidget { background: #ffffff; border: 1px solid #d9e3e8; border-radius: 6px; }
QListWidget::item { border-bottom: 1px solid #eef2f5; padding: 7px; }
QListWidget::item:selected { background: #e0f0f1; color: #153b41; }
QTabWidget::pane { border: 0; border-top: 1px solid #d9e3e8; }
QTabBar::tab { background: #edf3f5; color: #465f6d; border-radius: 6px;
              min-height: 27px; padding: 7px 12px; margin-right: 5px; font-weight: 600; }
QTabBar::tab:selected { background: #e0f0f1; color: #07565d; }
QTabBar::tab:hover { background: #e7f2f3; }
QDialog { background: #f3f7f9; }
"""


def label(text="", style="", word_wrap=False):
    result = QLabel(text)
    result.setTextFormat(Qt.TextFormat.PlainText)
    result.setObjectName(style)
    result.setWordWrap(word_wrap)
    return result


def button(text, action=None, primary=False):
    result = QPushButton(text)
    result.setCursor(Qt.CursorShape.PointingHandCursor)
    if primary:
        result.setObjectName("primary")
    if action is not None:
        result.clicked.connect(action)
    return result


def card():
    frame = QFrame()
    frame.setObjectName("card")
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(16, 14, 16, 14)
    layout.setSpacing(10)
    return frame, layout


def set_tone(widget, tone):
    widget.setProperty("tone", tone)
    widget.style().unpolish(widget)
    widget.style().polish(widget)


class FrameView(QWidget):
    point_picked = Signal(int, int)
    zone_picked = Signal(tuple)
    line_picked = Signal(tuple)

    def __init__(self, placeholder="暂无图像"):
        super().__init__()
        self.setMinimumSize(280, 195)
        self.setAccessibleName("检测图像")
        self.placeholder = placeholder
        self.pixmap = None
        self.mode = None
        self.drag_start = None
        self.drag_end = None

    def set_frame(self, image, placeholder=None):
        if placeholder is not None:
            self.placeholder = placeholder
        self.pixmap = None
        if image is not None:
            h, w = image.shape[:2]
            qimage = QImage(image.data, w, h, image.strides[0], QImage.Format.Format_BGR888)
            self.pixmap = QPixmap.fromImage(qimage)
        self.update()

    def set_mode(self, mode):
        self.mode = mode
        self.drag_start = self.drag_end = None
        shape = Qt.CursorShape.CrossCursor if mode else Qt.CursorShape.ArrowCursor
        self.setCursor(shape)
        self.update()

    def image_rect(self):
        if self.pixmap is None:
            return QRect()
        size = self.pixmap.size().scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio)
        return QRect((self.width() - size.width()) // 2, (self.height() - size.height()) // 2,
                     size.width(), size.height())

    def paintEvent(self, _):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#11222b"))
        if self.pixmap is None:
            painter.setPen(QColor("#c6d9df"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self.placeholder)
            return
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        painter.drawPixmap(self.image_rect(), self.pixmap)
        if self.drag_start is not None and self.drag_end is not None:
            painter.setPen(QPen(QColor("#66e5df"), 2, Qt.PenStyle.DashLine))
            if self.mode == "line":
                painter.drawLine(self.drag_start, self.drag_end)
            else:
                painter.drawRect(QRect(self.drag_start, self.drag_end).normalized())

    def to_image(self, point):
        rect = self.image_rect()
        x = min(max(point.x() - rect.x(), 0), rect.width() - 1)
        y = min(max(point.y() - rect.y(), 0), rect.height() - 1)
        return (x * self.pixmap.width() // rect.width(), y * self.pixmap.height() // rect.height())

    def mousePressEvent(self, event):
        pos = event.position().toPoint()
        if event.button() != Qt.MouseButton.LeftButton or not self.image_rect().contains(pos):
            return
        if self.mode == "hole":
            self.point_picked.emit(*self.to_image(pos))
        elif self.mode in ("line", "zone"):
            self.drag_start = self.drag_end = pos

    def mouseMoveEvent(self, event):
        if self.drag_start is not None:
            point = event.position().toPoint()
            bounds = self.image_rect()
            self.drag_end = QPoint(min(max(point.x(), bounds.left()), bounds.right()),
                                   min(max(point.y(), bounds.top()), bounds.bottom()))
            self.update()

    def mouseReleaseEvent(self, event):
        if self.drag_start is None or event.button() != Qt.MouseButton.LeftButton:
            return
        left, top = self.to_image(self.drag_start)
        right, bottom = self.to_image(event.position().toPoint())
        self.drag_start = self.drag_end = None
        self.update()
        if self.mode == "line":
            self.line_picked.emit(((left, top), (right, bottom)))
        else:
            self.zone_picked.emit((min(left, right), min(top, bottom), max(left, right), max(top, bottom)))

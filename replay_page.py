"""Operator view for automatic replay and per-stroke review."""

from dataclasses import replace
from threading import Event
from time import monotonic

import cv2
from PySide6.QtCore import QObject, QRunnable, QThreadPool, Qt, Signal
from PySide6.QtWidgets import QDialog, QHBoxLayout, QListWidget, QListWidgetItem, QMessageBox, QVBoxLayout, QWidget

from front_edge import DEFAULT_LINE, FrontEdge, TOLERANCE
from inspection import teach_hole
from line_replay import COLORS, DETAIL, VIDEO, LineInspector, LineReplay, ReplaySetup, SAMPLE_EVERY, crop
from widgets import FrameView, button, card, label, set_tone


STATUS = {"ok": "通过", "ng": "异常", "review": "待复核", "idle": "未设置"}
REPORT_STATUS = {**STATUS, "idle": "孔位未设"}
LEGEND = "青线为送料基准 · 绿/红线为实测前沿 · 圈为已设孔位 · 蓝框为杂物检查区"


class ReplaySignals(QObject):
    frame = Signal(object, float, float, str, object)
    last_report = Signal(object)
    done = Signal(str, bool)


class ReplayJob(QRunnable):
    def __init__(self, setup):
        super().__init__()
        self.setup = setup
        self.signals = ReplaySignals()
        self.play = Event()
        self.play.set()
        self.cancel = Event()

    def run(self):
        replay = None
        error = ""
        try:
            replay = LineReplay(setup=self.setup)
            frames = iter(replay.frames())
            duration = replay.total / replay.fps
            interval = SAMPLE_EVERY / replay.fps
            while not self.cancel.is_set():
                if not self.play.wait(0.05):
                    continue
                if self.cancel.is_set():
                    break
                started = monotonic()
                try:
                    image, seconds, phase, report = next(frames)
                except StopIteration:
                    last_report = replay.gate.missed(duration)
                    if last_report is not None:
                        self.signals.last_report.emit(last_report)
                    break
                self.signals.frame.emit(image, seconds, duration, phase, report)
                self.cancel.wait(max(0, interval - (monotonic() - started)))
        except ValueError as problem:
            error = str(problem)
        except Exception:
            error = "回放检测遇到错误，请重新打开程序。"
        finally:
            if replay is not None:
                replay.close()
            self.signals.done.emit(error, self.cancel.is_set())


class ReplaySetupDialog(QDialog):
    def __init__(self, parent, setup, reference):
        super().__init__(parent)
        self.setWindowTitle("设置检测位置")
        self.setMinimumSize(620, 540)
        self.resize(760, 650)
        self.setup = setup
        self.reference = reference
        self.gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
        root = QVBoxLayout(self)
        tools = QHBoxLayout()
        tools.addWidget(button("画送料线", self.teach_line))
        tools.addWidget(button("点选孔位", self.teach_holes))
        tools.addStretch()
        tools.addWidget(button("恢复默认送料线", lambda: self.use_line(DEFAULT_LINE)))
        self.remove = button("撤销最后一个孔", self.remove_hole)
        tools.addWidget(self.remove)
        root.addLayout(tools)
        self.help = label("", "hint", True)
        root.addWidget(self.help)
        self.view = FrameView()
        self.view.setAccessibleName("开模基准图")
        self.view.line_picked.connect(self.draw_line)
        self.view.point_picked.connect(self.add_hole)
        root.addWidget(self.view, 1)
        self.description = label("", "muted")
        root.addWidget(self.description)
        footer = QHBoxLayout()
        footer.addStretch()
        footer.addWidget(button("取消", self.reject))
        footer.addWidget(button("保存并返回", self.save, primary=True))
        root.addLayout(footer)
        self.teach_line()
        self.refresh()

    def teach_line(self):
        self.view.set_mode("line")
        self.help.setText("在图上沿凸出来的料头边拖一条线。送进方向朝画面下方。")

    def teach_holes(self):
        self.view.set_mode("hole")
        self.help.setText("逐个点击要检的实际孔中心。只选轮廓看得清的孔。")

    def draw_line(self, endpoints):
        self.use_line(tuple((x+DETAIL[0], y+DETAIL[1]) for x, y in endpoints))

    def use_line(self, line):
        try:
            FrontEdge(self.gray, line)
        except ValueError as error:
            self.help.setText(str(error))
            return
        self.setup = replace(self.setup, line=line)
        self.refresh()
        self.help.setText("送料线已更新。可点选孔位，或直接保存。")

    def add_hole(self, x, y):
        x += DETAIL[0]
        y += DETAIL[1]
        index = len(self.setup.holes) + 1
        hole = teach_hole(self.reference, x, y, f"孔 {index:02d}")
        if hole is None:
            self.help.setText("这里没找到清楚的孔轮廓，请放准孔中心再点。")
            return
        if any((hole.x-item.x)**2 + (hole.y-item.y)**2 <= 8**2 for item in self.setup.holes):
            self.help.setText("这个孔已设置，继续点下一个。")
            return
        self.setup = replace(self.setup, holes=self.setup.holes + (hole,))
        self.refresh()
        self.help.setText(f"已标记 {hole.name}。确认圆圈在实际孔上，再保存。")

    def remove_hole(self):
        self.setup = replace(self.setup, holes=self.setup.holes[:-1])
        self.refresh()
        self.help.setText("已撤销最后一个孔。")

    def refresh(self):
        marked = self.reference.copy()
        FrontEdge(self.gray, self.setup.line).draw(marked, None, COLORS["ok"])
        for hole in self.setup.holes:
            cv2.circle(marked, (hole.x, hole.y), 9, COLORS["ok"], 2, cv2.LINE_AA)
        self.view.set_frame(crop(marked, DETAIL).copy())
        self.remove.setEnabled(bool(self.setup.holes))
        self.description.setText(f"送料演示公差 ±{TOLERANCE} px   ·   已设置 {len(self.setup.holes)} 个孔位")

    def save(self):
        try:
            LineInspector(self.reference, self.setup)
        except ValueError as error:
            self.help.setText(str(error))
            return
        self.accept()


class ReplayPage(QWidget):
    def __init__(self, repository):
        super().__init__()
        self.setObjectName("page")
        self.pool = QThreadPool(self)
        self.pool.setMaxThreadCount(1)
        self.job = None
        self.reports = []
        self.repository = repository
        self.setup = ReplaySetup.read(repository.load_replay_setup())
        self.available = VIDEO.is_file()

        root = QVBoxLayout(self)
        root.setContentsMargins(23, 18, 23, 18)
        root.setSpacing(12)
        heading = QHBoxLayout()
        heading.addWidget(label("现场回放", "title"))
        heading.addWidget(label("级进模 · 现场实拍", "muted"))
        heading.addStretch()
        self.configure = button("设置检测位置", self.open_setup)
        heading.addWidget(self.configure)
        self.phase = label("等待回放", "hint")
        heading.addWidget(self.phase)
        root.addLayout(heading)

        panels = QHBoxLayout()
        panels.setSpacing(12)
        picture, pictures = card()
        picture.setMinimumWidth(500)
        views = QHBoxLayout()
        self.live = FrameView("点击开始回放")
        self.detail = FrameView("开模取图后显示")
        for header, view, stretch in (("现场画面", self.live, 3), ("检测取图", self.detail, 4)):
            column = QVBoxLayout()
            column.addWidget(label(header, "section"))
            view.setMinimumWidth(150)
            view.setAccessibleName(header)
            column.addWidget(view, 1)
            views.addLayout(column, stretch)
        pictures.addLayout(views, 1)
        self.time = label("等待播放", "muted")
        pictures.addWidget(self.time)
        self.detail_note = label(LEGEND, "muted", True)
        pictures.addWidget(self.detail_note)
        panels.addWidget(picture, 1)

        side, status = card()
        side.setMinimumWidth(306)
        side.setMaximumWidth(346)
        status.setSpacing(7)
        self.summary = label("等待检测", "status")
        set_tone(self.summary, "idle")
        status.addWidget(self.summary)
        self.checks = []
        for name in ("送料到位", "孔位", "有无杂物"):
            title = label(name, "section")
            detail = label("等待取图", "muted", True)
            status.addWidget(title)
            status.addWidget(detail)
            self.checks.append((name, title, detail))
        status.addWidget(label("叠料、跳料 · 待异常样本", "hint", True))
        status.addWidget(label("逐冲记录", "section"))
        self.list = QListWidget()
        self.list.setAccessibleName("逐冲记录")
        self.list.setMinimumHeight(82)
        self.list.currentRowChanged.connect(self.display_report)
        self.list.itemClicked.connect(self.inspect_record)
        self.list.itemActivated.connect(self.inspect_record)
        status.addWidget(self.list, 1)
        self.action = button("开始回放", self.toggle, primary=True)
        status.addWidget(self.action)
        panels.addWidget(side)
        root.addLayout(panels, 1)
        if not self.available:
            self.phase.setText("缺少现场回放视频")
            self.live.set_frame(None, "请在「运行」页使用模拟产线")
            self.configure.setEnabled(False)
            self.action.setEnabled(False)

    def toggle(self):
        if self.job is None:
            self.reports.clear()
            self.list.clear()
            self.display_report(-1)
            self.live.set_frame(None, "准备回放")
            self.time.setText("正在读取视频")
            self.phase.setText("正在打开现场视频")
            job = ReplayJob(self.setup)
            self.job = job
            job.signals.frame.connect(self.frame_ready, Qt.ConnectionType.QueuedConnection)
            job.signals.last_report.connect(self.add_report, Qt.ConnectionType.QueuedConnection)
            job.signals.done.connect(self.finished, Qt.ConnectionType.QueuedConnection)
            self.pool.start(job)
        elif self.job.play.is_set():
            self.pause()
            return
        else:
            self.job.play.set()
            self.phase.setText("继续回放")
        self.action.setText("暂停回放")

    def open_setup(self):
        self.pause()
        try:
            replay = LineReplay(setup=self.setup)
            reference = replay.gate.inspector.reference
            replay.close()
            dialog = ReplaySetupDialog(self, self.setup, reference)
            if dialog.exec() != QDialog.DialogCode.Accepted or dialog.setup == self.setup:
                return
            self.repository.save_replay_setup(dialog.setup.encode())
        except Exception as error:
            QMessageBox.warning(self, "无法设置检测位置", str(error))
            return
        self.setup = dialog.setup
        if self.job:
            self.action.setEnabled(False)
            self.job.cancel.set()
            self.job.play.set()
        self.list.clear()
        self.reports.clear()
        self.display_report(-1)
        self.live.set_frame(None, "点击开始回放")
        self.time.setText("等待播放")
        self.phase.setText("设置已保存")
        self.action.setText("开始回放")

    def pause(self):
        if self.job and self.job.play.is_set():
            self.job.play.clear()
            self.action.setText("继续回放")
            self.phase.setText("回放已暂停")

    def shutdown(self):
        if self.job:
            self.job.cancel.set()
            self.job.play.set()
            self.pool.waitForDone()

    def frame_ready(self, image, seconds, duration, phase, report):
        if self.job is None or self.job.cancel.is_set():
            return
        if self.job.play.is_set():
            self.live.set_frame(image)
            self.time.setText(f"{seconds:04.1f} / {duration:.1f} 秒")
            self.phase.setText(phase)
        if report is not None:
            self.add_report(report)

    def add_report(self, report):
        if self.job is None or self.job.cancel.is_set():
            return
        self.reports.append(report)
        self.list.addItem(QListWidgetItem(
            f"第 {len(self.reports)} 冲   {report.seconds:.1f} 秒   {REPORT_STATUS[report.status]}"
        ))
        if self.job.play.is_set():
            self.list.setCurrentRow(self.list.count()-1)
        self.list.scrollToBottom()

    def inspect_record(self, _):
        self.pause()
        self.display_report(self.list.currentRow())

    def display_report(self, index):
        if index < 0:
            self.summary.setText("等待检测")
            set_tone(self.summary, "idle")
            self.detail.set_frame(None, "开模取图后显示")
            self.detail_note.setText(LEGEND)
            for name, title, detail in self.checks:
                title.setText(name)
                detail.setText("等待取图")
            return
        report = self.reports[index]
        if self.job is None or not self.job.play.is_set():
            self.live.set_frame(report.original, "本冲未取得稳定图像")
            self.time.setText(f"第 {index+1} 冲记录 · {report.seconds:.1f} 秒")
        self.summary.setText(f"第 {index+1} 冲 · {REPORT_STATUS[report.status]}")
        set_tone(self.summary, report.status)
        self.detail.set_frame(report.image, "本冲未取得稳定图像")
        timing = f"取图后对齐与判定 {report.elapsed_ms:.1f} ms" if report.image is not None else "本冲没有检测图"
        self.detail_note.setText(f"{timing}\n{LEGEND}")
        for (name, title, detail), result in zip(self.checks, report.checks):
            title.setText(f"{name} · {STATUS[result.status]}")
            detail.setText(result.message)

    def finished(self, error, cancelled):
        self.job = None
        self.action.setEnabled(True)
        if cancelled:
            return
        self.action.setText("重新回放")
        if error:
            self.phase.setText(error)
        else:
            self.phase.setText(f"回放结束 · {len(self.reports)} 冲")
            self.display_report(self.list.currentRow())

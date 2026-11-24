"""Lingjian Press desktop entry point."""

import argparse
from pathlib import Path
import os
import sys
import uuid

import cv2
from PySide6.QtCore import QThreadPool, QTimer, Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication, QButtonGroup, QFrame, QHBoxLayout, QListWidget, QListWidgetItem,
    QMainWindow, QMessageBox, QStackedWidget, QVBoxLayout, QWidget,
)

from cycles import CycleJob, CycleResult
from inspection import Inspector, Verdict
from recipe_page import RecipePage
from replay_page import ReplayPage
from storage import EVENT_LIMIT, Repository, decode
from widgets import FrameView, STYLE, button, card, label, set_tone


RESULT_LABELS = {"ok": "合格", "ng": "工件异常", "review": "待复核", "fault": "检测故障"}
OPERATOR_HINTS = {
    "ok": "自动等待下一冲。",
    "ng": "请查看标记位置，并按现场流程处理工件。",
    "review": "请对照图片检查表面。",
    "fault": "请查看故障原因。确认后可继续查看最新结果。",
}


class RunPage(QWidget):
    def __init__(self, toggle, acknowledge):
        super().__init__()
        self.setObjectName("page")
        root = QVBoxLayout(self)
        root.setContentsMargins(23, 18, 23, 18)
        root.setSpacing(12)
        heading = QHBoxLayout()
        heading.addWidget(label("运行监控", "title"))
        self.recipe_name = label("", "muted")
        heading.addWidget(self.recipe_name)
        heading.addStretch()
        self.live_status = label("等待冲床", "hint")
        heading.addWidget(self.live_status)
        root.addLayout(heading)

        panels = QHBoxLayout()
        panels.setSpacing(14)
        frame, visual = card()
        self.frame_title = label("配方良品图", "section")
        visual.addWidget(self.frame_title)
        self.view = FrameView()
        visual.addWidget(self.view, 1)
        self.frame_footer = label("", "muted", True)
        visual.addWidget(self.frame_footer)
        panels.addWidget(frame, 1)

        status_card, status = card()
        status_card.setMinimumWidth(300)
        status_card.setMaximumWidth(340)
        self.result_title = label("当前判定", "section")
        status.addWidget(self.result_title)
        self.result = label("等待触发", "status")
        set_tone(self.result, "idle")
        status.addWidget(self.result)
        self.reason = label("已加载模具配方", word_wrap=True)
        self.reason.setMinimumHeight(42)
        status.addWidget(self.reason)
        self.cycle_note = label("", "muted", True)
        status.addWidget(self.cycle_note)
        self.hint = label("启动模拟产线后，软件会逐冲自动检测。", "hint", True)
        self.hint.setMinimumHeight(64)
        status.addWidget(self.hint)
        self.ack = button("确认已查看", acknowledge)
        self.ack.hide()
        status.addWidget(self.ack)
        status.addStretch()

        line = QFrame()
        line.setObjectName("line")
        line.setFixedHeight(1)
        status.addWidget(line)
        counts = QHBoxLayout()
        self.metrics = {}
        for key, name in (("ok", "合格"), ("issue", "需处理"), ("fault", "故障")):
            box = QVBoxLayout()
            box.setSpacing(2)
            value = label("0", "value")
            box.addWidget(value)
            box.addWidget(label(name, "metric"))
            counts.addLayout(box)
            self.metrics[key] = value
        status.addLayout(counts)
        self.toggle = button("启动模拟产线", toggle, primary=True)
        status.addWidget(self.toggle)
        panels.addWidget(status_card)
        root.addLayout(panels, 1)

    def show_result(self, sample, result, pinned):
        self.ack.setVisible(pinned)
        self.result_title.setText("异常待查看" if pinned else "当前判定")
        if result is None:
            set_tone(self.result, "idle")
            self.result.setText("等待触发")
            self.reason.setText("已加载模具配方")
            self.cycle_note.setText(self.position_mode)
            self.hint.setText("启动模拟产线后，软件会逐冲自动检测。")
            self.frame_title.setText("配方良品图")
            self.view.set_frame(sample)
            return
        verdict = result.verdict
        self.result.setText(RESULT_LABELS[verdict.status])
        set_tone(self.result, verdict.status)
        self.reason.setText(verdict.reason)
        elapsed = f"模拟冲次 {result.number:06d}  ·  判定 {result.elapsed:.1f} ms"
        self.cycle_note.setText("\n".join(filter(None, (elapsed, verdict.position))))
        self.frame_title.setText(f"异常留图 · 冲次 {result.number:06d}" if pinned else f"检测图 · 冲次 {result.number:06d}")
        hint = OPERATOR_HINTS[verdict.status]
        if result.evidence_error:
            hint += "\n异常记录写入失败，请检查本地存储。"
        self.hint.setText(hint)
        self.view.set_frame(verdict.image, "本次无图像")


class HistoryPage(QWidget):
    def __init__(self, repository):
        super().__init__()
        self.repository = repository
        self.current = None
        self.setObjectName("page")
        root = QVBoxLayout(self)
        root.setContentsMargins(23, 18, 23, 18)
        root.setSpacing(12)
        toolbar = QHBoxLayout()
        toolbar.addWidget(label("异常记录", "title"))
        toolbar.addWidget(label(f"最近 {EVENT_LIMIT} 条 · 模拟产线", "muted"))
        toolbar.addStretch()
        toolbar.addWidget(button("刷新记录", self.refresh))
        root.addLayout(toolbar)

        content = QHBoxLayout()
        content.setSpacing(14)
        items, item_layout = card()
        items.setMinimumWidth(295)
        items.setMaximumWidth(330)
        self.entries = QListWidget()
        self.entries.setAccessibleName("异常列表")
        self.entries.setWordWrap(True)
        self.entries.currentItemChanged.connect(self.selected)
        self.empty = label("还没有异常记录。", "hint", True)
        item_layout.addWidget(self.empty)
        item_layout.addWidget(self.entries, 1)
        content.addWidget(items)

        visual, right = card()
        header = QHBoxLayout()
        self.title = label("异常照片", "section")
        header.addWidget(self.title)
        header.addStretch()
        self.swap = button("查看原图", self.switch_image)
        self.swap.setCheckable(True)
        self.swap.setEnabled(False)
        header.addWidget(self.swap)
        right.addLayout(header)
        self.view = FrameView("选择一条记录查看现场图片")
        right.addWidget(self.view, 1)
        self.detail = label("", "muted", True)
        right.addWidget(self.detail)
        content.addWidget(visual, 1)
        root.addLayout(content, 1)

    def refresh(self):
        current_id = self.entries.currentItem().data(Qt.ItemDataRole.UserRole) if self.entries.currentItem() else None
        rows = self.repository.events()
        self.entries.blockSignals(True)
        self.entries.clear()
        selected_row = 0
        for index, row in enumerate(rows):
            text = f"{row['created']}   {RESULT_LABELS[row['status']]}\n冲次 {row['cycle']:06d}  ·  {row['recipe'][:18]}"
            item = QListWidgetItem(text)
            item.setData(Qt.ItemDataRole.UserRole, row["id"])
            self.entries.addItem(item)
            if row["id"] == current_id:
                selected_row = index
        if rows:
            self.entries.setCurrentRow(selected_row)
        self.entries.blockSignals(False)
        self.empty.setVisible(not rows)
        self.selected(self.entries.currentItem(), None)

    def selected(self, item, _previous):
        self.current = self.repository.event(item.data(Qt.ItemDataRole.UserRole)) if item else None
        self.swap.setChecked(False)
        self.swap.setEnabled(self.current is not None and self.current["original"] is not None)
        if self.current is None:
            self.title.setText("异常照片")
            self.detail.clear()
        else:
            row = self.current
            self.title.setText(f"{RESULT_LABELS[row['status']]} · 冲次 {row['cycle']:06d}")
            self.detail.setText(f"{row['reason']}\n{row['created']}  ·  {row['elapsed_ms']:.1f} ms")
        self.switch_image()

    def switch_image(self):
        show_raw = self.swap.isChecked()
        self.swap.setText("查看标记图" if show_raw else "查看原图")
        data = None if self.current is None else self.current["original" if show_raw else "marked"]
        self.view.set_frame(decode(data), "本次无图像" if self.current else "选择一条记录查看现场图片")


class PressWindow(QMainWindow):
    def __init__(self, repository, interval_ms=1100):
        super().__init__()
        self.repository = repository
        self.run_id = uuid.uuid4().hex
        self.number = self.recipe_cycle = 0
        self.counts = {"ok": 0, "issue": 0, "fault": 0}
        self.latest = self.alert = None
        self.current_job = None
        self.running = False
        self.pool = QThreadPool(self)
        self.pool.setMaxThreadCount(1)
        self.timer = QTimer(self)
        self.timer.setInterval(interval_ms)
        self.timer.timeout.connect(self.trigger)
        self.setWindowTitle("灵鉴 · 冲压检测")
        self.setMinimumSize(935, 660)
        screen = QApplication.primaryScreen().availableGeometry()
        self.resize(min(1190, screen.width()), min(795, screen.height() - 30))
        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(self.make_header())
        self.pages = QStackedWidget()
        self.runtime = RunPage(self.toggle_running, self.acknowledge)
        self.recipes = RecipePage(repository)
        self.recipes.activated.connect(self.activate_recipe)
        self.history = HistoryPage(repository)
        self.replay = ReplayPage(repository)
        for page in (self.runtime, self.recipes, self.history, self.replay):
            self.pages.addWidget(page)
        active_page = 3 if self.replay.available else 0
        self.pages.setCurrentIndex(active_page)
        self.tabs.button(active_page).setChecked(True)
        root.addWidget(self.pages, 1)
        self.setCentralWidget(central)
        self.load_inspector(*self.repository.load_active())

    def make_header(self):
        header = QFrame()
        header.setObjectName("header")
        row = QHBoxLayout(header)
        row.setContentsMargins(23, 10, 23, 10)
        row.setSpacing(9)
        identity = QVBoxLayout()
        identity.setSpacing(0)
        identity.addWidget(label("灵鉴 · 冲压检测", "brand"))
        identity.addWidget(label("级进模视觉检测台", "brandSub"))
        row.addLayout(identity)
        row.addSpacing(16)
        self.tabs = QButtonGroup(self)
        for index, text in ((3, "现场回放"), (0, "运行"), (1, "模具配方"), (2, "异常记录")):
            tab = button(text)
            tab.setObjectName("tab")
            tab.setCheckable(True)
            self.tabs.addButton(tab, index)
            tab.setChecked(index == 3)
            row.addWidget(tab)
        self.tabs.idClicked.connect(self.switch_page)
        row.addStretch()
        row.addWidget(label("离线模式", "mode"))
        return header

    def switch_page(self, index):
        if index == 3 and self.running:
            self.toggle_running()
        if index != 3:
            self.replay.pause()
        if index == 2:
            self.history.refresh()
        self.pages.setCurrentIndex(index)

    def activate_recipe(self, recipe_id):
        if self.running or self.current_job:
            return
        try:
            recipe, reference = self.repository.get_recipe(recipe_id)
            inspector = Inspector(recipe, reference)
            self.repository.activate(recipe_id)
        except Exception as error:
            QMessageBox.warning(self, "配方无法应用", str(error))
            return
        self.load_inspector(recipe, reference, inspector)
        self.recipes.refresh(recipe_id)
        self.recipes.show_message("已应用到运行。")

    def load_inspector(self, recipe, reference, inspector=None):
        self.inspector = inspector or Inspector(recipe, reference)
        self.runtime.recipe_name.setText(f"当前模具 · {recipe.name}")
        self.runtime.position_mode = "位置跟随已开启" if recipe.anchor else "固定位置检测"
        legend = []
        if recipe.feed:
            legend.append("青圈 · 送料定位点")
        if recipe.holes:
            legend.append("绿圈 · 已定位")
        if recipe.feed or recipe.holes:
            legend.append("红圈 · 位置异常")
        if recipe.review:
            legend.append("黄圈 · 翻边待复核")
        if recipe.zones:
            legend.append("橙圈 · 疑似异物")
        self.runtime.frame_footer.setText("     ".join(legend))
        self.latest = self.alert = None
        self.recipe_cycle = 0
        self.counts = {"ok": 0, "issue": 0, "fault": 0}
        self.refresh_runtime()

    def toggle_running(self):
        self.running = not self.running
        self.runtime.live_status.setText("自动检测中" if self.running else "等待冲床")
        self.runtime.toggle.setText("暂停模拟产线" if self.running else "启动模拟产线")
        self.recipes.set_blocked(self.running or self.current_job is not None)
        if self.running:
            self.timer.start()
            self.trigger()
        else:
            self.timer.stop()

    def trigger(self):
        if not self.running:
            return
        self.number += 1
        self.recipe_cycle += 1
        if self.current_job is not None:
            verdict = Verdict("fault", "上一冲尚未完成，本冲未取图")
            evidence_error = False
            try:
                self.repository.add_event(self.run_id, self.number, self.inspector.recipe.name, verdict, 0, None)
            except Exception:
                evidence_error = True
            self.receive_result(CycleResult(self.number, verdict, 0, evidence_error))
            return
        job = CycleJob(self.number, self.recipe_cycle, self.inspector, self.repository, self.run_id)
        job.signals.done.connect(self.job_done, Qt.ConnectionType.QueuedConnection)
        self.current_job = job
        self.pool.start(job)

    def job_done(self, result):
        self.current_job = None
        self.recipes.set_blocked(self.running)
        self.receive_result(result)

    def receive_result(self, result):
        status = result.verdict.status
        self.counts[status if status in ("ok", "fault") else "issue"] += 1
        if self.latest is None or self.latest.number < result.number:
            self.latest = result
        if status != "ok" and (self.alert is None or self.alert.number < result.number):
            self.alert = result
        self.refresh_runtime()
        if self.pages.currentIndex() == 2 and status != "ok":
            self.history.refresh()

    def refresh_runtime(self):
        self.runtime.show_result(self.inspector.reference, self.alert or self.latest, self.alert is not None)
        for key, value in self.counts.items():
            self.runtime.metrics[key].setText(str(value))

    def acknowledge(self):
        self.alert = None
        self.refresh_runtime()

    def closeEvent(self, event):
        self.running = False
        self.timer.stop()
        self.replay.shutdown()
        self.pool.waitForDone()
        super().closeEvent(event)


def create_app():
    app = QApplication(sys.argv[:1])
    app.setApplicationName("灵鉴 · 冲压检测")
    app.setStyle("Fusion")
    app.setFont(QFont("Microsoft YaHei UI", 10))
    app.setStyleSheet(STYLE)
    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path(os.environ.get("LOCALAPPDATA", Path.home())) / "LingjianPress")
    arguments = parser.parse_args()
    cv2.setNumThreads(1)
    app = create_app()
    try:
        window = PressWindow(Repository(arguments.data_dir))
    except Exception as error:
        QMessageBox.critical(None, "无法启动灵鉴 · 冲压检测", str(error))
        return 1
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())

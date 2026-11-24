"""Four-step recipe teaching and replay for a fixed camera view."""

from dataclasses import replace
import sqlite3
import uuid

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox, QDoubleSpinBox, QFileDialog, QHBoxLayout, QInputDialog, QLineEdit,
    QListWidget, QStackedWidget, QTabWidget, QVBoxLayout, QWidget,
)

from alignment import AnchorTracker
from inspection import FEED_DIRECTIONS, FeedCheck, FeedInspector, Inspector, draw_reference, read_image, teach_hole
from storage import BUILTIN_IDS, Recipe
from widgets import FrameView, button, card, label


IMAGE_FILTER = "图片 (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"
STEPS = ("良品图", "检测点", "表面区域", "试跑并保存")


class RecipePage(QWidget):
    activated = Signal(str)

    def __init__(self, repository):
        super().__init__()
        self.repository = repository
        self.active_id = None
        self.blocked = False
        self.draft_id = None
        self.reference = None
        self.holes = []
        self.zones = []
        self.review = ()
        self.anchor = None
        self.feed_hole = None
        self.picking_anchor = False
        self.step = 0
        self.setObjectName("page")
        root = QVBoxLayout(self)
        root.setContentsMargins(23, 18, 23, 18)
        root.setSpacing(12)

        top = QHBoxLayout()
        top.addWidget(label("模具配方", "title"))
        top.addStretch()
        self.selector = QComboBox()
        self.selector.setAccessibleName("已保存的模具配方")
        self.selector.setMinimumWidth(185)
        self.selector.currentIndexChanged.connect(self.show_saved)
        top.addWidget(self.selector)
        self.apply = button("应用到运行", self.apply_selected)
        self.edit = button("编辑所选", self.edit_selected)
        self.new = button("新建配方", self.start_new, primary=True)
        for control in (self.apply, self.edit, self.new):
            top.addWidget(control)
        root.addLayout(top)

        display = QHBoxLayout()
        display.setSpacing(14)
        picture, picture_layout = card()
        self.picture_title = label("良品图", "section")
        self.view = FrameView("选择一张良品图")
        self.view.point_picked.connect(self.select_point)
        self.view.zone_picked.connect(self.pick_region)
        picture_layout.addWidget(self.picture_title)
        picture_layout.addWidget(self.view, 1)
        display.addWidget(picture, 1)

        panel, side = card()
        panel.setMinimumWidth(300)
        panel.setMaximumWidth(345)
        self.step_title = label("所选配方", "section")
        self.tip = label("", "muted", True)
        side.addWidget(self.step_title)
        side.addWidget(self.tip)
        self.steps = QStackedWidget()
        self.steps.addWidget(self.make_reference_step())
        self.steps.addWidget(self.make_detector_step())
        self.steps.addWidget(self.make_zone_step())
        self.steps.addWidget(self.make_trial_step())
        saved = QWidget()
        saved_layout = QVBoxLayout(saved)
        saved_layout.setContentsMargins(0, 0, 0, 0)
        self.saved_summary = label("", "hint", True)
        saved_layout.addWidget(self.saved_summary)
        saved_layout.addStretch()
        self.steps.addWidget(saved)
        side.addWidget(self.steps, 1)
        self.message = label("", "hint", True)
        self.message.hide()
        side.addWidget(self.message)
        self.wizard_buttons = QWidget()
        nav = QHBoxLayout(self.wizard_buttons)
        nav.setContentsMargins(0, 0, 0, 0)
        self.cancel = button("取消", self.show_saved)
        self.back = button("上一步", self.previous_step)
        self.forward = button("下一步", self.next_step, primary=True)
        for control in (self.cancel, self.back, self.forward):
            nav.addWidget(control)
        side.addWidget(self.wizard_buttons)
        display.addWidget(panel)
        root.addLayout(display, 1)
        self.refresh()

    @staticmethod
    def step_container():
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 4, 0, 0)
        return widget, layout

    def make_reference_step(self):
        widget, layout = self.step_container()
        self.name = QLineEdit()
        self.name.setMaxLength(40)
        self.name.setPlaceholderText("例如 A01 左支架")
        self.name.setAccessibleName("配方名称")
        layout.addWidget(label("配方名称"))
        layout.addWidget(self.name)
        layout.addWidget(button("选择良品图", self.select_reference))
        self.anchor_state = label("", "muted", True)
        layout.addWidget(self.anchor_state)
        self.anchor_button = button("框选定位基准", self.start_anchor)
        layout.addWidget(self.anchor_button)
        self.clear_anchor_button = button("清除定位基准", self.clear_anchor)
        layout.addWidget(self.clear_anchor_button)
        layout.addStretch()
        return widget

    def make_detector_step(self):
        self.detectors = QTabWidget()
        self.detectors.addTab(self.make_hole_controls(), "孔位检测")
        self.detectors.addTab(self.make_feed_controls(), "送料到位")
        self.detectors.currentChanged.connect(self.update_detector_tip)
        return self.detectors

    def make_hole_controls(self):
        widget, layout = self.step_container()
        self.hole_list = QListWidget()
        self.hole_list.setAccessibleName("已选必检孔")
        layout.addWidget(self.hole_list, 1)
        actions = QHBoxLayout()
        actions.addWidget(button("改名称", self.rename_hole))
        actions.addWidget(button("移除所选", self.remove_hole))
        layout.addLayout(actions)
        self.tolerance = QDoubleSpinBox()
        self.tolerance.setRange(0.5, 16)
        self.tolerance.setValue(5)
        self.tolerance.setDecimals(1)
        self.tolerance.setSuffix(" px")
        self.tolerance.setAccessibleName("孔位允许偏差")
        layout.addWidget(label("孔位允许偏差"))
        layout.addWidget(self.tolerance)
        return widget

    def make_feed_controls(self):
        widget, layout = self.step_container()
        layout.addWidget(label("送料方向"))
        self.feed_direction = QComboBox()
        self.feed_direction.setAccessibleName("画面中的送料方向")
        for code, (direction, _) in FEED_DIRECTIONS.items():
            self.feed_direction.addItem(direction, code)
        layout.addWidget(self.feed_direction)
        layout.addWidget(label("送料允许偏差"))
        self.feed_tolerance = QDoubleSpinBox()
        self.feed_tolerance.setAccessibleName("送料允许偏差")
        self.feed_tolerance.setRange(0.5, 16)
        self.feed_tolerance.setValue(5)
        self.feed_tolerance.setSuffix(" px")
        self.feed_tolerance.setDecimals(1)
        layout.addWidget(self.feed_tolerance)
        self.feed_hint = label("在良品图上点选一个料带孔", "hint", True)
        layout.addWidget(self.feed_hint)
        self.remove_feed_button = button("移除送料检测", self.remove_feed)
        layout.addWidget(self.remove_feed_button)
        layout.addStretch()
        self.feed_direction.currentIndexChanged.connect(self.feed_settings_changed)
        return widget

    def make_zone_step(self):
        widget, layout = self.step_container()
        self.zone_list = QListWidget()
        self.zone_list.setAccessibleName("表面检测区域")
        layout.addWidget(self.zone_list, 1)
        layout.addWidget(button("移除所选", self.remove_zone))
        return widget

    def make_trial_step(self):
        widget, layout = self.step_container()
        layout.addWidget(button("试跑良品", self.trial_good))
        layout.addWidget(button("导入试检图", self.trial_other))
        layout.addStretch()
        return widget

    def refresh(self, active_id=None):
        self.active_id = active_id or self.repository.load_active()[0].id
        self.selector.blockSignals(True)
        self.selector.clear()
        for row in self.repository.recipes():
            self.selector.addItem(row["name"], row["id"])
        self.selector.setCurrentIndex(self.selector.findData(self.active_id))
        self.selector.blockSignals(False)
        self.show_saved()

    def set_blocked(self, blocked):
        self.blocked = blocked
        self.update_controls()

    def update_controls(self):
        draft = self.draft_id is not None
        self.selector.setEnabled(not draft)
        self.apply.setEnabled(not self.blocked and not draft and self.selector.currentData() != self.active_id)
        self.edit.setEnabled(not self.blocked and not draft)
        self.new.setEnabled(not self.blocked and not draft)
        self.steps.setEnabled(not self.blocked or not draft)
        self.wizard_buttons.setVisible(draft)
        self.wizard_buttons.setEnabled(not self.blocked)
        self.anchor_button.setEnabled(self.reference is not None)
        self.clear_anchor_button.setVisible(self.anchor is not None)
        self.anchor_state.setText("已开启位置跟随" if self.anchor else "位置跟随未开启")
        self.feed_hint.setText("已选送料定位孔" if self.feed_hole else "待选送料定位孔")
        self.remove_feed_button.setEnabled(self.feed_hole is not None)
        mode = None
        if draft and not self.blocked:
            if self.step == 1:
                mode = "hole"
            elif self.step == 2 or (self.step == 0 and self.picking_anchor):
                mode = "zone"
        self.view.set_mode(mode)

    def show_saved(self):
        recipe_id = self.selector.currentData()
        if not recipe_id:
            return
        self.draft_id = None
        self.reference = None
        self.show_message()
        self.step_title.setText("所选配方")
        self.steps.setCurrentIndex(4)
        self.edit.setText("复制并修改" if recipe_id in BUILTIN_IDS else "编辑所选")
        try:
            recipe, image = self.repository.get_recipe(recipe_id)
            self.view.set_frame(draw_reference(image, recipe.holes, recipe.zones, recipe.review,
                                               anchor=recipe.anchor, feed=recipe.feed))
            current = "当前运行配方" if recipe_id == self.active_id else "可应用到运行"
            self.tip.setText(f"{recipe.name}\n{current}")
            self.picture_title.setText("良品图")
            summary = f"{len(recipe.holes)} 个必检孔 · {len(recipe.zones)} 个表面区域"
            if recipe.holes:
                summary += f"\n孔位偏差 {recipe.tolerance:g} px"
            if recipe.feed:
                direction = FEED_DIRECTIONS[recipe.feed.direction][0]
                summary += f"\n送料到位 · {direction} · ±{recipe.feed.tolerance:g} px"
            summary += "\n位置跟随已开启" if recipe.anchor else "\n固定位置检测"
            if recipe.review:
                summary += f"\n{len(recipe.review)} 处翻边孔待复核"
            self.saved_summary.setText(summary)
        except Exception as error:
            self.view.set_frame(None, "无法打开良品图")
            self.show_message(str(error))
        self.update_controls()

    def apply_selected(self):
        self.activated.emit(self.selector.currentData())

    def start_new(self):
        self.start_draft(None, None)

    def edit_selected(self):
        try:
            recipe, image = self.repository.get_recipe(self.selector.currentData())
        except Exception as error:
            self.show_message(str(error))
            return
        self.start_draft(recipe, image)

    def start_draft(self, recipe, image):
        copying_demo = recipe is not None and recipe.id in BUILTIN_IDS
        self.reference = image
        self.anchor = recipe.anchor if recipe else None
        self.draft_id = recipe.id if recipe and not copying_demo else uuid.uuid4().hex
        name = recipe.name if recipe else ""
        self.name.setText(f"{name} 副本" if copying_demo else name)
        self.holes = list(recipe.holes) if recipe else []
        self.hole_list.clear()
        self.hole_list.addItems([hole.name for hole in self.holes])
        self.zones = list(recipe.zones) if recipe else []
        self.update_zones()
        self.review = recipe.review if recipe else ()
        feed = recipe.feed if recipe else None
        self.feed_hole = feed.hole if feed else None
        self.feed_direction.setCurrentIndex(self.feed_direction.findData(feed.direction if feed else "right"))
        self.feed_tolerance.setValue(feed.tolerance if feed else 5.0)
        self.detectors.setCurrentIndex(1 if feed else 0)
        self.tolerance.setValue(recipe.tolerance if recipe else 5.0)
        self.show_step(0)

    def show_message(self, message=""):
        self.message.setText(message)
        self.message.setVisible(bool(message))

    def show_step(self, index):
        self.step = index
        self.picking_anchor = False
        self.show_message()
        self.step_title.setText(f"{index + 1} / 4  {STEPS[index]}")
        self.tip.setText((
            "选择良品图。需要位置跟随时，框一个清晰、稳定的模具或夹具边角。",
            "",
            "拖动鼠标框出需要检测的平面，避开翻边和强反光。只检孔时可跳过。",
            "先试跑一张良品，也可以导入同机位的不良图。保存后应用到运行。",
        )[index])
        if index == 1:
            self.update_detector_tip()
        self.steps.setCurrentIndex(index)
        self.back.setEnabled(index > 0)
        self.forward.setText("保存并应用" if index == 3 else "下一步")
        self.preview()
        self.update_controls()

    def select_reference(self):
        file, _ = QFileDialog.getOpenFileName(self, "选择良品图", "", IMAGE_FILTER)
        if not file:
            return
        try:
            image = read_image(file)
        except (OSError, ValueError) as error:
            self.show_message(str(error))
            return
        self.reference = image
        self.holes.clear()
        self.hole_list.clear()
        self.zones.clear()
        self.update_zones()
        self.review = ()
        self.anchor = None
        self.feed_hole = None
        self.picking_anchor = False
        self.preview()
        self.update_controls()
        self.show_message("良品图已加载。请在接下来的步骤重新选择孔和表面区域。")

    def preview(self):
        self.picture_title.setText("良品图")
        image = None
        if self.reference is not None:
            image = draw_reference(self.reference, self.holes, self.zones, self.review,
                                   anchor=self.anchor, feed=self.feed_check())
        self.view.set_frame(image)

    def start_anchor(self):
        self.picking_anchor = True
        self.update_controls()
        self.show_message("在左侧按住鼠标拖框，选中一处清晰的边角。")

    def clear_anchor(self):
        self.anchor = None
        self.picking_anchor = False
        self.preview()
        self.update_controls()
        self.show_message("已清除定位基准。")

    def update_detector_tip(self):
        if self.step != 1:
            return
        self.show_message()
        self.tip.setText(
            "点一个料带定位孔，选择画面中的送料方向。基准框应放在固定的模具或夹具上。"
            if self.detectors.currentIndex() == 1 else
            "用鼠标点一下孔中心。绿圈出现后，孔就加到必检列表里了。"
        )

    def feed_check(self):
        if self.feed_hole is None:
            return None
        return FeedCheck(self.feed_hole, self.feed_direction.currentData(), self.feed_tolerance.value())

    def feed_settings_changed(self):
        if self.step == 1 and self.reference is not None:
            self.preview()

    def remove_feed(self):
        self.feed_hole = None
        self.preview()
        self.update_controls()
        self.show_message("已移除送料检测。")

    def select_point(self, x, y):
        if self.detectors.currentIndex() == 0:
            self.add_hole(x, y)
        else:
            self.select_feed_hole(x, y)

    def select_feed_hole(self, x, y):
        if self.anchor is None:
            self.show_message("请先回到上一步，框选固定模具或夹具上的定位基准。")
            return
        hole = teach_hole(self.reference, x, y, "送料定位孔")
        if hole is None:
            self.show_message("未找到清晰孔口，请点在能看清的料带孔内。")
            return
        try:
            feed = FeedCheck(hole, self.feed_direction.currentData(), self.feed_tolerance.value())
            FeedInspector(self.reference, feed, self.anchor)
        except ValueError as error:
            self.show_message(str(error))
            return
        self.feed_hole = hole
        self.preview()
        self.update_controls()
        self.show_message("已添加送料到位。青色箭头表示送料方向。")

    def pick_region(self, region):
        if self.step == 2:
            self.add_zone(region)
            return
        if not self.picking_anchor:
            return
        try:
            tracker = AnchorTracker(self.reference, region)
            if tracker.find(self.reference) is None:
                raise ValueError("这处基准不易区分，请换另一个有明显拐点的区域。")
        except ValueError as error:
            self.show_message(str(error))
            return
        self.anchor = region
        self.picking_anchor = False
        self.preview()
        self.update_controls()
        self.show_message("基准已设置，检测区将跟随它移动。")

    def add_hole(self, x, y):
        if self.reference is None:
            return
        names = {hole.name for hole in self.holes}
        index = 1
        while f"孔 {index:02d}" in names:
            index += 1
        hole = teach_hole(self.reference, x, y, f"孔 {index:02d}")
        if hole is None:
            self.show_message("这里没有找到清晰孔口，请点在孔内或换一张取景更清楚的图。")
            return
        if any((hole.x - item.x) ** 2 + (hole.y - item.y) ** 2 < 8 ** 2 for item in self.holes):
            self.show_message("这个孔已经加入必检列表。")
            return
        self.holes.append(hole)
        self.hole_list.addItem(hole.name)
        self.hole_list.setCurrentRow(len(self.holes) - 1)
        self.preview()
        self.show_message(f"已添加 {hole.name}")

    def rename_hole(self):
        index = self.hole_list.currentRow()
        if index < 0:
            return
        name, accepted = QInputDialog.getText(self, "修改孔名称", "输入现场名称", text=self.holes[index].name)
        name = name.strip()[:28]
        if not accepted or not name:
            return
        if any(hole.name == name and i != index for i, hole in enumerate(self.holes)):
            self.show_message("已有同名必检孔，请换个名称。")
            return
        self.holes[index] = replace(self.holes[index], name=name)
        self.hole_list.item(index).setText(self.holes[index].name)

    def remove_hole(self):
        index = self.hole_list.currentRow()
        if index >= 0:
            self.holes.pop(index)
            self.hole_list.takeItem(index)
            self.preview()

    def add_zone(self, zone):
        left, top, right, bottom = zone
        if right - left < 18 or bottom - top < 18:
            self.show_message("选区太小，请重新拖一个稍大的矩形。")
            return
        self.zones.append(zone)
        self.update_zones()
        self.zone_list.setCurrentRow(len(self.zones) - 1)
        self.preview()
        self.show_message(f"已添加表面区域 {len(self.zones)}")

    def update_zones(self):
        self.zone_list.clear()
        self.zone_list.addItems([f"表面区域 {index}" for index in range(1, len(self.zones) + 1)])

    def remove_zone(self):
        index = self.zone_list.currentRow()
        if index >= 0:
            self.zones.pop(index)
            self.update_zones()
            self.preview()

    def previous_step(self):
        self.show_step(self.step - 1)

    def next_step(self):
        if self.step == 0 and (not self.name.text().strip() or self.reference is None):
            self.show_message("请填写配方名称，并选择一张良品图。")
        elif self.step == 1 and not (self.holes or self.feed_hole):
            self.show_message("请点选一个必检孔或送料定位孔。")
        elif self.step == 1 and self.feed_hole and self.anchor is None:
            self.show_message("请先回到上一步，重新框选固定定位基准。")
        elif self.step == 3:
            self.save()
        else:
            self.show_step(self.step + 1)

    def recipe(self):
        return Recipe(self.draft_id, self.name.text().strip(), tuple(self.holes), tuple(self.zones),
                      self.tolerance.value(), self.review, self.anchor, self.feed_check())

    def trial_good(self):
        self.trial(self.reference)

    def trial_other(self):
        file, _ = QFileDialog.getOpenFileName(self, "导入同机位试检图", "", IMAGE_FILTER)
        if file:
            try:
                self.trial(read_image(file))
            except (ValueError, OSError) as error:
                self.show_message(str(error))

    def trial(self, frame):
        try:
            result = Inspector(self.recipe(), self.reference).inspect(frame)
        except ValueError as error:
            self.show_message(str(error))
            return
        states = {"ok": "合格", "ng": "工件异常", "review": "待复核", "fault": "无法判定"}
        self.picture_title.setText(f"试跑结果 · {states[result.status]}")
        self.view.set_frame(frame if result.image is None else result.image)
        self.show_message("\n".join(filter(None, (result.reason, result.position))))

    def save(self):
        try:
            recipe = self.recipe()
            Inspector(recipe, self.reference)
            self.repository.save_recipe(recipe, self.reference)
        except (ValueError, OSError, sqlite3.Error) as error:
            self.show_message(str(error))
            return
        self.refresh(recipe.id)
        self.show_message("配方已保存。")
        self.activated.emit(recipe.id)

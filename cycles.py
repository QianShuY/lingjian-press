"""Simulation source and one-result-per-trigger background processing."""

from dataclasses import dataclass
from time import perf_counter

import cv2
from PySide6.QtCore import QObject, QRunnable, Signal

from inspection import Verdict
from storage import DEMO_ID, FEED_DEMO_BG, FEED_DEMO_ID, FEED_MARGIN_LEFT, FEED_MARGIN_OTHER


DEMO_SEQUENCE = ("normal", "follow", "rotate", "missing", "shift", "foreign", "lost", "timeout")
FEED_DEMO_SEQUENCE = (
    ("normal", 0), ("follow", 0), ("short", -13), ("over", 15),
    ("normal", 0), ("missing", 0), ("lost", 0), ("timeout", 0),
)


def simulated_frame(inspector, number):
    reference = inspector.reference
    if inspector.recipe.id == FEED_DEMO_ID:
        return simulated_feed_frame(inspector, number)
    if inspector.recipe.id != DEMO_ID:
        return reference
    scenario = DEMO_SEQUENCE[(number - 1) % len(DEMO_SEQUENCE)]
    if scenario == "timeout":
        return None
    if scenario == "normal":
        return reference
    if scenario == "follow":
        return move_view(reference, 26, 18, 0)
    if scenario == "rotate":
        return move_view(reference, 24, 13, 4)
    if scenario == "lost":
        image = reference.copy()
        x0, y0, x1, y1 = inspector.recipe.anchor
        image[y0-12:y1+12, x0-12:x1+12] = 150
        return image
    image = reference.copy()
    if scenario == "missing":
        x, y = map(round, inspector.expected[0])
        cv2.circle(image, (x, y), 13, (220, 220, 220), -1)
    elif scenario == "shift":
        x, y = map(round, inspector.expected[5])
        red = tuple(int(channel) for channel in image[y, x])
        cv2.circle(image, (x, y), 15, (220, 220, 220), -1)
        cv2.ellipse(image, (x + 11, y), (9, 7), 0, 0, 360, red, -1)
        return move_view(image, 26, 18, 0)
    elif scenario == "foreign":
        cv2.ellipse(image, (1180 - 776, 580), (6, 4), 20, 0, 360, (25, 25, 25), -1)
    return image


def simulated_feed_frame(inspector, number):
    scenario, offset = FEED_DEMO_SEQUENCE[(number - 1) % len(FEED_DEMO_SEQUENCE)]
    image = inspector.reference
    if scenario == "timeout":
        return None
    if scenario == "normal":
        return image
    if scenario == "follow":
        return move_view(image, 22, 10, 3)
    trial = image.copy()
    if offset:
        top, left = FEED_MARGIN_OTHER, FEED_MARGIN_LEFT
        h, w = image.shape[:2]
        bottom, right = h - FEED_MARGIN_OTHER, w - FEED_MARGIN_OTHER
        trial[:, left:right] = FEED_DEMO_BG
        trial[top+offset:bottom+offset, left:right] = image[top:bottom, left:right]
        return move_view(trial, -14, 16, 2) if scenario == "over" else trial
    if scenario == "missing":
        hole = inspector.recipe.feed.hole
        cv2.circle(trial, (hole.x, hole.y), 18, (220, 220, 220), -1)
    if scenario == "lost":
        left, top, right, bottom = inspector.recipe.anchor
        trial[top:bottom, left:right] = FEED_DEMO_BG
    return trial


def move_view(image, dx, dy, angle):
    h, w = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    matrix[:, 2] += (dx, dy)
    return cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(28, 25, 91))


@dataclass(frozen=True)
class CycleResult:
    number: int
    verdict: Verdict
    elapsed: float
    evidence_error: bool = False


class JobSignals(QObject):
    done = Signal(object)


class CycleJob(QRunnable):
    def __init__(self, number, recipe_cycle, inspector, repository, run_id):
        super().__init__()
        self.signals = JobSignals()
        self.number = number
        self.recipe_cycle = recipe_cycle
        self.inspector = inspector
        self.repository = repository
        self.run_id = run_id
        self.started = perf_counter()

    def run(self):
        frame = None
        try:
            frame = simulated_frame(self.inspector, self.recipe_cycle)
            verdict = self.inspector.inspect(frame)
        except Exception:
            verdict = Verdict("fault", "图像处理失败，请检查当前配方。")
        elapsed = (perf_counter() - self.started) * 1000
        try:
            self.repository.add_event(self.run_id, self.number, self.inspector.recipe.name, verdict, elapsed, frame)
        except Exception:
            self.signals.done.emit(CycleResult(self.number, verdict, elapsed, True))
        else:
            self.signals.done.emit(CycleResult(self.number, verdict, elapsed))

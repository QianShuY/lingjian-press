"""Phase gating and inspection for the bundled phone video of the progressive die."""

from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from alignment import match, rectangle_points
from front_edge import DEFAULT_LINE, FrontEdge, TOLERANCE
from inspection import Hole, hole_candidates


VIDEO = Path(__file__).resolve().parent / "assets" / "line-replay.mp4"
REFERENCE_FRAME = 125
SAMPLE_EVERY = 3
DETAIL = (260, 450, 611, 886)
ANCHORS = (
    (107, 478, 202, 559), (161, 591, 229, 633), (224, 790, 274, 850),
    (79, 971, 155, 1039), (535, 637, 580, 683), (575, 824, 636, 885),
    (451, 882, 565, 939), (609, 461, 672, 510), (256, 515, 309, 564),
)
SURFACE_GUIDES = ((376, 604, 463, 629), (368, 707, 458, 739))
SURFACE = ((413, 564, 454, 583), (411, 680, 455, 703))
HOLE_TOLERANCE = 4
COLORS = {
    "ok": (121, 233, 98), "surface": (248, 151, 120),
    "review": (10, 185, 255), "ng": (60, 67, 255),
}


@dataclass(frozen=True)
class ReplaySetup:
    line: tuple[tuple[int, int], tuple[int, int]] = DEFAULT_LINE
    holes: tuple[Hole, ...] = ()

    @classmethod
    def read(cls, payload):
        if not payload:
            return cls()
        config = json.loads(payload)
        return cls(tuple(tuple(point) for point in config["line"]),
                   tuple(Hole(**hole) for hole in config["holes"]))

    def encode(self):
        return json.dumps(asdict(self), ensure_ascii=False)


@dataclass(frozen=True)
class Check:
    status: str
    message: str


@dataclass(frozen=True)
class Report:
    seconds: float
    image: np.ndarray | None
    checks: tuple[Check, ...]
    elapsed_ms: float = 0.0
    original: np.ndarray | None = None

    @property
    def status(self):
        return max((check.status for check in self.checks), key=("ok", "idle", "review", "ng").index)


def crop(image, rect, dx=0, dy=0):
    left, top, right, bottom = rect
    return image[top+dy:bottom+dy, left+dx:right+dx]


def outline(image, rect, status, dx=0, dy=0):
    left, top, right, bottom = rect
    cv2.rectangle(image, (left+dx, top+dy), (right+dx-1, bottom+dy-1), COLORS[status], 2, cv2.LINE_AA)


def locate_patch(image, template, rect, width, height, dx=0, dy=0):
    left, top, right, bottom = rect
    search = image[top+dy-height:bottom+dy+height, left+dx-width:right+dx+width]
    score, (x, y), _ = match(search, template)
    return score, x + dx - width, y + dy - height


class LineInspector:
    def __init__(self, reference, setup=ReplaySetup()):
        self.reference = reference
        self.setup = setup
        gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
        self.anchors = tuple(crop(gray, rect).copy() for rect in ANCHORS)
        self.edge = FrontEdge(gray, setup.line)
        self.guides = tuple(crop(gray, rect).copy() for rect in SURFACE_GUIDES)
        self.surface = tuple(cv2.GaussianBlur(crop(gray, rect), (5, 5), 0).astype(np.int16)
                             for rect in SURFACE)
        self.holes = []
        for hole in setup.holes:
            candidates = list(hole_candidates(reference, hole.x, hole.y, hole.method, 27))
            nearby = [item for item in candidates if np.hypot(item[0]-hole.x, item[1]-hole.y) <= 3]
            if len(nearby) != 1:
                raise ValueError(f"{hole.name} 在基准图中无法确认，请重新点选。")
            x, y, area = nearby[0]
            station = int(y >= (SURFACE_GUIDES[0][1] + SURFACE_GUIDES[1][1])/2)
            self.holes.append((hole, x, y, area, station))

    def align(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        centers, hits = [], []
        for rect, template in zip(ANCHORS, self.anchors):
            left, top, right, bottom = rect
            x, y = max(0, left - 140), max(0, top - 110)
            patch = gray[y:bottom+110, x:right+115]
            score, (u, v), _ = match(patch, template)
            if score >= 0.77:
                cx, cy = (right-left)/2, (bottom-top)/2
                centers.append((left+cx, top+cy))
                hits.append((x+u+cx, y+v+cy))
        if len(hits) < 4:
            return None
        centers = np.float32(centers)
        matrix, valid = cv2.estimateAffine2D(centers, np.float32(hits), ransacReprojThreshold=3.5)
        if matrix is None or valid.sum() < 4:
            return None
        inliers = centers[valid[:, 0] == 1]
        if inliers[:, 0].min() > 270 or inliers[:, 0].max() < 550 or np.ptp(inliers[:, 1]) < 210:
            return None
        scales = np.linalg.svd(matrix[:, :2], compute_uv=False)
        if np.linalg.det(matrix[:, :2]) <= 0 or scales[0] > 1.23 or scales[1] < 0.8:
            return None
        corners = cv2.transform(np.float32([rectangle_points(DETAIL)]), matrix)[0]
        if (corners < 1).any() or (corners >= (width-1, height-1)).any():
            return None
        return cv2.warpAffine(image, matrix, (width, height), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)

    def inspect(self, aligned, seconds):
        gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
        marked = aligned.copy()
        positions = []
        for rect, guide in zip(SURFACE_GUIDES, self.guides):
            score, dx, dy = locate_patch(gray, guide, rect, 18, 26)
            positions.append((dx, dy) if score >= 0.84 and abs(dx) < 18 and abs(dy) < 26 else None)
        checks = (self.check_feed(gray, marked), self.check_holes(aligned, marked, positions),
                  self.check_surface(gray, marked, positions))
        return Report(seconds, crop(marked, DETAIL).copy(), checks)

    def check_feed(self, gray, marked):
        offset = self.edge.measure(gray)
        if offset is None:
            self.edge.draw(marked, None, COLORS["review"])
            return Check("review", "未能确认料头前沿")
        status = "ok" if abs(offset) <= TOLERANCE else "ng"
        self.edge.draw(marked, offset, COLORS[status])
        outcome = "到位" if status == "ok" else ("送过了" if offset > 0 else "没送到")
        return Check(status, f"{outcome} · 前沿偏差 {offset:+.1f} px")

    def check_holes(self, aligned, marked, positions):
        if not self.holes:
            return Check("idle", "在设置检测位置中点选实际要检的孔")
        issues = []
        status = "ok"
        for hole, base_x, base_y, area, station in self.holes:
            local = positions[station]
            dx, dy = local or (0, 0)
            x, y = round(base_x + dx), round(base_y + dy)
            nearby = [
                (u, v) for u, v, size in hole_candidates(aligned, x, y, hole.method, 22)
                if 0.60*area <= size <= 1.55*area and np.hypot(u-x, v-y) <= 15
            ] if local is not None else []
            color = "review"
            if len(nearby) == 1:
                u, v = nearby[0]
                distance = np.hypot(u-x, v-y)
                x, y = round(u), round(v)
                color = "ok" if distance <= HOLE_TOLERANCE else "ng"
                if color == "ng":
                    issues.append(f"{hole.name} 偏移 {distance:.1f} px")
                    status = "ng"
            else:
                issues.append(f"{hole.name} 未能确认")
                if status == "ok":
                    status = "review"
            cv2.circle(marked, (x, y), 9, COLORS[color], 2, cv2.LINE_AA)
        if issues:
            tail = f"等 {len(issues)} 项" if len(issues) > 2 else ""
            return Check(status, "、".join(issues[:2]) + tail)
        return Check("ok", f"{len(self.holes)} 个已设置孔位通过")

    def check_surface(self, gray, marked, positions):
        found = 0
        unavailable = False
        for rect, reference, position in zip(SURFACE, self.surface, positions):
            if position is None:
                unavailable = True
                outline(marked, rect, "review")
                continue
            dx, dy = position
            patch = crop(gray, rect, dx, dy)
            change = cv2.GaussianBlur(patch, (5, 5), 0).astype(np.int16) - reference
            mask = np.uint8(np.abs(change - np.median(change)) >= 33)
            _, _, stats, _ = cv2.connectedComponentsWithStats(mask)
            candidates = []
            for x, y, width, height, area in stats[1:]:
                if area >= 18 and min(width, height) >= 3:
                    candidates.append((x, y, width, height))
            outline(marked, rect, "review" if candidates else "surface", dx, dy)
            for x, y, width, height in candidates:
                center = (rect[0]+dx+x+width//2, rect[1]+dy+y+height//2)
                cv2.circle(marked, center, max(width, height)//2+4, COLORS["review"], 2, cv2.LINE_AA)
            found += len(candidates)
        if found:
            return Check("review", f"有 {found} 处疑似异物或反光")
        if unavailable:
            return Check("review", "有区域未能定位")
        return Check("ok", "两处可见区域无明显异常")


def die_visibility(image):
    # This threshold is specific to the yellow trim in the bundled video.
    hsv = cv2.cvtColor(np.ascontiguousarray(image[:850:3, 10:195:3]), cv2.COLOR_BGR2HSV)
    yellow = cv2.inRange(hsv, (16, 100, 132), (41, 255, 255))
    rows = np.flatnonzero(np.count_nonzero(yellow, axis=1) > 12)
    if not len(rows):
        return "unknown"
    edge = 3 * np.quantile(rows, 0.9)
    if edge <= 230:
        return "open"
    return "blocked" if edge >= 320 else "unknown"


class CycleGate:
    def __init__(self, inspector):
        self.inspector = inspector
        self.armed = False
        self.saw_window = False
        self.blocked = self.opened = self.still = 0
        self.previous = None

    def missed(self, seconds):
        if self.armed and self.saw_window:
            self.armed = self.saw_window = False
            check = Check("review", "未取得稳定开模画面")
            return Report(seconds, None, (check,) * 3)
        return None

    def update(self, image, seconds):
        visibility = die_visibility(image)
        if visibility != "open":
            self.opened = self.still = 0
            self.previous = None
            self.blocked = self.blocked + 1 if visibility == "blocked" else 0
            report = None
            if self.blocked == 3:
                report = self.missed(seconds)
                self.armed = True
                self.saw_window = False
            return "上模遮挡 · 等待开模" if visibility == "blocked" else "等待清晰画面", report
        self.blocked = 0
        if not self.armed:
            return "开模 · 等待下一冲" if not self.saw_window else "本冲已取图", None
        self.opened += 1
        self.saw_window = self.opened >= 3 or self.saw_window
        if self.opened < 2:
            return "开模 · 等待料带稳定", None
        started = perf_counter()
        aligned = self.inspector.align(image)
        if aligned is None:
            self.previous = None
            self.still = 0
            return "开模 · 查找下模基准", None
        area = cv2.cvtColor(aligned[502:748, 360:475], cv2.COLOR_BGR2GRAY)
        difference = cv2.absdiff(area, self.previous).mean() if self.previous is not None else float("inf")
        self.previous = area
        self.still = self.still + 1 if difference <= 11.5 else 0
        if self.still < 2:
            return "开模 · 等待料带稳定", None
        self.armed = False
        report = self.inspector.inspect(aligned, seconds)
        return "本冲已取图", replace(report, elapsed_ms=(perf_counter()-started)*1000, original=image)


class LineReplay:
    def __init__(self, video=VIDEO, setup=ReplaySetup()):
        self.capture = cv2.VideoCapture(str(video), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1])
        try:
            self.fps = self.capture.get(cv2.CAP_PROP_FPS)
            self.total = round(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, REFERENCE_FRAME)
            ready, reference = self.capture.read()
            if not ready or reference.shape != (1280, 720, 3) or self.fps <= 0 or self.total <= REFERENCE_FRAME:
                raise ValueError("现场视频无法读取，请检查程序目录中的回放文件。")
            self.gate = CycleGate(LineInspector(reference, setup))
            if not self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0):
                raise ValueError("现场视频无法从头播放。")
        except Exception:
            self.capture.release()
            raise

    def frames(self):
        for index in range(0, self.total, SAMPLE_EVERY):
            ready, image = self.capture.read()
            if not ready:
                raise ValueError("现场视频在回放结束前无法读取。")
            seconds = index / self.fps
            phase, report = self.gate.update(image, seconds)
            yield image, seconds, phase, report
            for _ in range(min(SAMPLE_EVERY - 1, self.total - index - 1)):
                if not self.capture.grab():
                    raise ValueError("现场视频在回放结束前无法读取。")

    def close(self):
        self.capture.release()

"""Fixed-view inspection, using the local hole and surface tests from the pilot."""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from alignment import AnchorTracker, rectangle_points


@dataclass(frozen=True)
class Hole:
    name: str
    x: int
    y: int
    method: str


FEED_DIRECTIONS = {
    "right": ("从左往右", (1, 0)),
    "left": ("从右往左", (-1, 0)),
    "down": ("从上往下", (0, 1)),
    "up": ("从下往上", (0, -1)),
}
FEED_SPAN = 30
FEED_LATERAL = 12
FEED_COLOR = (199, 192, 14)


@dataclass(frozen=True)
class FeedCheck:
    hole: Hole
    direction: str
    tolerance: float


@dataclass(frozen=True)
class Verdict:
    status: str
    reason: str
    image: np.ndarray | None = None
    position: str = ""


def read_image(path):
    image = cv2.imdecode(np.fromfile(Path(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("无法打开图片，请选择有效的 JPG、PNG 或 BMP。")
    h, w = image.shape[:2]
    if min(h, w) < 100:
        raise ValueError("图片尺寸太小，请使用检测相机原图。")
    if max(h, w) > 1834:
        scale = 1834 / max(h, w)
        image = cv2.resize(image, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    return image


def hole_candidates(image, x, y, method, radius):
    left, top = max(0, x - radius), max(0, y - radius)
    patch = image[top:y + radius + 1, left:x + radius + 1]
    if patch.size == 0:
        return
    if method == "red":
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (0, 90, 22), (12, 255, 255))
        mask |= cv2.inRange(hsv, (165, 90, 22), (179, 255, 255))
    else:
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 110, 255, cv2.THRESH_BINARY_INV)
    count, _, stats, centers = cv2.connectedComponentsWithStats(mask)
    for index in range(1, count):
        u, v, width, height, area = stats[index]
        if (
            not 22 <= area <= 1100 or min(width, height) < 4 or u == 0 or v == 0
            or u + width == mask.shape[1] or v + height == mask.shape[0]
        ):
            continue
        u, v = centers[index] + (left, top)
        yield (round(float(u), 2), round(float(v), 2), int(area))


def find_center(image, x, y, method, radius=27):
    best = None
    best_distance = radius - 6
    for u, v, _ in hole_candidates(image, x, y, method, radius):
        distance = float(np.hypot(u - x, v - y))
        if distance < best_distance:
            best = (u, v)
            best_distance = distance
    return best


def teach_hole(image, x, y, name):
    matches = [(find_center(image, x, y, method), method) for method in ("red", "dark")]
    matches = [(center, method) for center, method in matches if center is not None]
    if not matches:
        return None
    center, method = min(matches, key=lambda match: np.hypot(match[0][0] - x, match[0][1] - y))
    if np.hypot(center[0] - x, center[1] - y) > 12:
        return None
    return Hole(name, round(center[0]), round(center[1]), method)


def surface_spots(image, zones, holes):
    if not zones:
        return []
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    spots = []
    for left, top, right, bottom in zones:
        patch = gray[top:bottom, left:right]
        mask = cv2.morphologyEx(patch, cv2.MORPH_BLACKHAT, kernel)
        _, mask = cv2.threshold(mask, 17, 255, cv2.THRESH_BINARY)
        count, _, stats, _ = cv2.connectedComponentsWithStats(mask)
        for index in range(1, count):
            x, y, w, h, area = (int(value) for value in stats[index])
            if not 10 <= area <= 240 or x < 2 or y < 2 or x+w >= mask.shape[1]-1 or y+h >= mask.shape[0]-1:
                continue
            cx, cy = left+x+w//2, top+y+h//2
            if any(np.hypot(cx - hole.x, cy - hole.y) <= 17 for hole in holes):
                continue
            spots.append((cx, cy, max(w, h)//2 + 5))
    return spots


def round_point(x, y):
    return round(x), round(y)


def outline_region(image, region, color, point):
    contour = np.array([point(x, y) for x, y in rectangle_points(region)], dtype=np.int32)
    cv2.polylines(image, [contour], True, color, 2, cv2.LINE_AA)


def draw_reference(image, holes, zones, review=(), anchor=None, point=round_point, feed=None):
    marked = image.copy()
    for region in zones:
        outline_region(marked, region, (205, 157, 46), point)
    if anchor:
        outline_region(marked, anchor, (123, 232, 216), point)
    for hole in holes:
        mark_hole(marked, *point(hole.x, hole.y), (89, 207, 41))
    for x, y in review:
        cv2.circle(marked, point(x, y), 11, (35, 230, 255), 2, cv2.LINE_AA)
    if feed:
        x, y = feed.hole.x, feed.hole.y
        dx, dy = FEED_DIRECTIONS[feed.direction][1]
        mark_hole(marked, *point(x, y), FEED_COLOR)
        cv2.arrowedLine(marked, point(x + dx*16, y + dy*16), point(x + dx*38, y + dy*38),
                        FEED_COLOR, 2, cv2.LINE_AA, tipLength=0.35)
    return marked


def mark_hole(image, x, y, color):
    cv2.circle(image, (x, y), 13, color, 2, cv2.LINE_AA)
    cv2.drawMarker(image, (x, y), color, cv2.MARKER_CROSS, 10, 1, cv2.LINE_AA)


class Inspector:
    def __init__(self, recipe, reference):
        self.recipe = recipe
        self.reference = reference
        self.expected = [find_center(reference, h.x, h.y, h.method) for h in recipe.holes]
        if any(point is None for point in self.expected):
            raise ValueError("良品图的必检孔无法定位，请检查配方。")
        self.tracker = AnchorTracker(reference, recipe.anchor) if recipe.anchor else None
        self.feed = FeedInspector(reference, recipe.feed, recipe.anchor) if recipe.feed else None
        masked_holes = recipe.holes + ((recipe.feed.hole,) if recipe.feed else ())
        self.masked_holes = masked_holes
        self.reference_spots = surface_spots(reference, recipe.zones, masked_holes)
        self.coverage = [point for zone in recipe.zones for point in rectangle_points(zone)]
        for x, y in self.expected:
            self.coverage.extend(((x - 7, y - 7), (x - 7, y + 7), (x + 7, y - 7), (x + 7, y + 7)))
        if self.feed:
            self.coverage.extend(self.feed.coverage)

    def covers_checks(self, point, width, height):
        for x, y in self.coverage:
            u, v = point(x, y)
            if not (0 <= u < width and 0 <= v < height):
                return False
        return True

    def inspect(self, image):
        if image is None:
            return Verdict("fault", "本次拍照超时")
        if image.shape != self.reference.shape:
            return Verdict("fault", "图片尺寸与配方不一致")
        aligned = image
        point = round_point
        position = ""
        if self.tracker:
            placement = self.tracker.find(image)
            if placement is None:
                return Verdict("fault", "找不到定位基准，请检查相机位置或基准是否被挡住。", image, "基准丢失")
            point = placement.point
            height, width = image.shape[:2]
            if not self.covers_checks(point, width, height):
                return Verdict("fault", "有必检位置超出画面，请调整相机取景。", image, "检测区出界")
            aligned = cv2.warpAffine(image, placement.matrix, (width, height),
                                     flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
            position = (
                f"定位已跟随  X {round(placement.dx):+d}  Y {round(placement.dy):+d} px  "
                f"{round(placement.angle, 1) + 0.0:+.1f}°"
            )
        marked = draw_reference(image, (), self.recipe.zones, self.recipe.review, point=point)
        missing = []
        shifted = []
        for hole, expected in zip(self.recipe.holes, self.expected):
            center = find_center(aligned, hole.x, hole.y, hole.method)
            if center is None:
                missing.append(hole.name)
                mark_hole(marked, *point(hole.x, hole.y), (46, 57, 235))
                continue
            x, y = point(*center)
            offset = float(np.hypot(center[0] - expected[0], center[1] - expected[1]))
            color = (89, 207, 41)
            if offset > self.recipe.tolerance:
                color = (46, 57, 235)
                shifted.append(f"{hole.name}偏移 {offset:.1f} px")
            mark_hole(marked, x, y, color)
        feed_status, feed_reason = ("ok", "")
        if self.feed:
            feed_status, feed_reason = self.feed.measure(aligned, marked, point)
        candidates = []
        for x, y, radius in surface_spots(aligned, self.recipe.zones, self.masked_holes):
            if any(np.hypot(x-u, y-v) <= 5 for u, v, _ in self.reference_spots):
                continue
            if any(np.hypot(x-u, y-v) <= 5 for u, v in candidates):
                continue
            candidates.append((x, y))
            cv2.circle(marked, point(x, y), radius, (0, 151, 255), 2, cv2.LINE_AA)
        problems = [*(f"{name}未找到" for name in missing), *shifted]
        if feed_status != "ok":
            problems.insert(0, feed_reason)
        if candidates:
            problems.append(f"表面有 {len(candidates)} 处疑似异物")
        if problems:
            reason = "，".join(problems[:3])
            if len(problems) > 3:
                reason += f"等 {len(problems)} 项"
            status = feed_status
            if status != "fault":
                status = "ng" if status == "ng" or missing or shifted else "review"
            return Verdict(status, reason, marked, position)
        summary = [feed_reason] if feed_reason else []
        if self.recipe.holes:
            summary.append(f"{len(self.recipe.holes)} 个必检孔通过")
        if not summary:
            summary.append("表面未发现疑似异物")
        return Verdict("ok", " · ".join(summary), marked, position)


class FeedInspector:
    def __init__(self, reference, feed, anchor):
        if anchor is None:
            raise ValueError("送料到位需要先框选固定模具或夹具上的定位基准。")
        if feed.direction not in FEED_DIRECTIONS or not 0.5 <= feed.tolerance <= 16:
            raise ValueError("请检查送料方向和允许偏差。")
        hole = feed.hole
        left, top, right, bottom = anchor
        if left <= hole.x < right and top <= hole.y < bottom:
            raise ValueError("请在固定基准框外选择料带定位孔。")
        candidates = list(hole_candidates(reference, hole.x, hole.y, hole.method, 27))
        near = [item for item in candidates if np.hypot(item[0] - hole.x, item[1] - hole.y) <= 8]
        if len(near) != 1:
            raise ValueError("良品图上的送料定位孔看不清，请重新点选。")
        self.x, self.y, self.area = near[0]
        self.hole = hole
        self.tolerance = feed.tolerance
        self.dx, self.dy = FEED_DIRECTIONS[feed.direction][1]
        x_pad, y_pad = (FEED_SPAN + 6, FEED_LATERAL + 6) if self.dx else (FEED_LATERAL + 6, FEED_SPAN + 6)
        self.coverage = ((self.x-x_pad, self.y-y_pad), (self.x-x_pad, self.y+y_pad),
                         (self.x+x_pad, self.y-y_pad), (self.x+x_pad, self.y+y_pad))
        h, w = reference.shape[:2]
        if any(not (0 <= x < w and 0 <= y < h) for x, y in self.coverage):
            raise ValueError("送料定位孔离画面边缘太近，请选择画面内更靠中的孔。")
        if len(self.matching_holes(reference)) != 1:
            raise ValueError("这里有多个相近的孔，请换一个容易区分的送料定位孔。")

    def matching_holes(self, image):
        hits = []
        for u, v, area in hole_candidates(image, self.hole.x, self.hole.y, self.hole.method, FEED_SPAN + 13):
            if not 0.6 * self.area <= area <= 1.55 * self.area:
                continue
            x, y = u - self.x, v - self.y
            along = self.dx*x + self.dy*y
            across = self.dy*x - self.dx*y
            if abs(along) <= FEED_SPAN and abs(across) <= FEED_LATERAL:
                hits.append((u, v, along))
        return hits

    def measure(self, aligned, marked, point):
        hits = self.matching_holes(aligned)
        if len(hits) != 1:
            mark_hole(marked, *point(self.x, self.y), (46, 57, 235))
            reason = "料带定位孔未找到" if not hits else "找到多个相似的送料孔，请检查取景和配方"
            return "fault", reason
        u, v, along = hits[0]
        offset = round(along, 1) + 0.0
        expected = point(self.x, self.y)
        measured = point(u, v)
        cv2.circle(marked, expected, 18, FEED_COLOR, 2, cv2.LINE_AA)
        if abs(along) <= self.tolerance:
            mark_hole(marked, *measured, FEED_COLOR)
            return "ok", f"送料到位 {offset:+.1f} px"
        color = (46, 57, 235)
        cv2.line(marked, expected, measured, color, 2, cv2.LINE_AA)
        mark_hole(marked, *measured, color)
        cause = "送料偏短" if along < 0 else "送料偏过"
        return "ng", f"{cause} {abs(offset):.1f} px"

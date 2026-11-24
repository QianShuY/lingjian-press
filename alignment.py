"""Match a taught image region and map reference coordinates into the current shot."""

from dataclasses import dataclass

import cv2
import numpy as np


ANGLES = range(-8, 9)
COARSE_ANGLES = range(-8, 9, 2)
MIN_SCORE = 0.88
MIN_PEAK_GAP = 0.03


@dataclass(frozen=True)
class Alignment:
    matrix: np.ndarray
    dx: float
    dy: float
    angle: float

    def point(self, x, y):
        u, v = self.matrix @ (x, y, 1)
        return round(float(u)), round(float(v))


def rectangle_points(rect):
    left, top, right, bottom = rect
    return ((left, top), (right - 1, top), (right - 1, bottom - 1), (left, bottom - 1))


def offset_of_peak(left, middle, right):
    curvature = left - 2 * middle + right
    if curvature >= -1e-6:
        return 0.0
    return float(np.clip((left - right) / (2 * curvature), -0.5, 0.5))


def half_size(image):
    return cv2.resize(image, (image.shape[1] // 2, image.shape[0] // 2), interpolation=cv2.INTER_AREA)


def match(image, template):
    response = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    _, score, _, position = cv2.minMaxLoc(response)
    return score, position, response


class AnchorTracker:
    def __init__(self, reference, region):
        left, top, right, bottom = region
        height, width = reference.shape[:2]
        w, h = right - left, bottom - top
        if not (0 <= left < right <= width and 0 <= top < bottom <= height):
            raise ValueError("定位基准框超出良品图。")
        if min(w, h) < 30 or max(w, h) > 300:
            raise ValueError("定位基准框大小建议为 30–300 像素，请框一处清楚的边角。")
        gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
        sample = gray[top:bottom, left:right]
        corners = cv2.goodFeaturesToTrack(sample, maxCorners=6, qualityLevel=0.05, minDistance=6)
        if sample.std() < 12 or corners is None or len(corners) < 3:
            raise ValueError("定位基准纹理太少，请把明显的边角或拐点框进去。")

        self.center = ((left + right - 1) / 2, (top + bottom - 1) / 2)
        self.left = left
        self.top = top
        margin = max(60, min(150, round(min(height, width) * 0.13)))
        self.search = (max(0, left - margin), max(0, top - margin),
                       min(width, right + margin), min(height, bottom + margin))
        self.templates = {}
        for angle in ANGLES:
            transform = cv2.getRotationMatrix2D(self.center, angle, 1.0)
            transform[:, 2] -= (left, top)
            template = cv2.warpAffine(gray, transform, (w, h), borderMode=cv2.BORDER_REFLECT_101)
            self.templates[angle] = template
        self.coarse = {angle: half_size(self.templates[angle]) for angle in COARSE_ANGLES}
        self.peak_radius = max(6, min(w, h) // 10)

    def find(self, image):
        left, top, right, bottom = self.search
        gray = cv2.cvtColor(image[top:bottom, left:right], cv2.COLOR_BGR2GRAY)
        reduced = half_size(gray)
        rough_angle, (rough_score, (rough_x, rough_y), rough_map) = max(
            ((angle, match(reduced, template)) for angle, template in self.coarse.items()),
            key=lambda candidate: candidate[1][0],
        )
        radius = self.peak_radius
        rough_map[max(0, rough_y-radius):rough_y+radius+1, max(0, rough_x-radius):rough_x+radius+1] = -1
        if rough_score - float(rough_map.max()) < MIN_PEAK_GAP:
            return None

        height, width = self.templates[0].shape
        x0, y0 = max(0, 2*rough_x - 6), max(0, 2*rough_y - 6)
        patch = gray[y0:2*rough_y+height+7, x0:2*rough_x+width+7]
        refined = {
            angle: match(patch, self.templates[angle])
            for angle in range(max(ANGLES.start, rough_angle-2), min(ANGLES.stop, rough_angle+3))
        }
        best_angle = max(refined, key=lambda angle: refined[angle][0])
        best_score, (x, y), best_map = refined[best_angle]
        if best_score < MIN_SCORE or abs(best_angle) == 8:
            return None
        rows, cols = best_map.shape
        if x == 0 or y == 0 or x == cols - 1 or y == rows - 1:
            return None
        px = offset_of_peak(best_map[y, x - 1], best_score, best_map[y, x + 1])
        py = offset_of_peak(best_map[y - 1, x], best_score, best_map[y + 1, x])
        angle = float(best_angle)
        if best_angle - 1 in refined and best_angle + 1 in refined:
            angle += offset_of_peak(refined[best_angle - 1][0], best_score, refined[best_angle + 1][0])
        dx = left + x0 + x + px - self.left
        dy = top + y0 + y + py - self.top
        matrix = cv2.getRotationMatrix2D(self.center, angle, 1.0)
        matrix[:, 2] += (dx, dy)
        return Alignment(matrix, dx, dy, angle)

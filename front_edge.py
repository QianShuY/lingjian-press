"""Measure the material's leading edge along the normal of a taught line."""

import cv2
import numpy as np


DEFAULT_LINE = ((392, 826), (432, 820))
TOLERANCE = 8
SEARCH = 32
HALF_TEMPLATE = 6
SEGMENTS = 5
COLUMNS = 40


class FrontEdge:
    def __init__(self, gray, line):
        start, end = np.float32(sorted(line))
        direction = end - start
        if direction[0] < 22 or abs(direction[1]) > direction[0] * 0.6:
            raise ValueError("请沿凸出来的料头边，从一侧拖到另一侧，线段不要太短或太陡。")
        direction /= np.linalg.norm(direction)
        # In this camera view, forward is toward the bottom of the image.
        self.normal = np.float32((-direction[1], direction[0]))
        self.ends = np.float32((start, end))
        columns = np.linspace(start, end, COLUMNS, dtype=np.float32)
        extent = SEARCH + HALF_TEMPLATE
        offsets = np.arange(-extent, extent + 1, dtype=np.float32)
        grid = columns[None, :, :] + offsets[:, None, None] * self.normal
        height, width = gray.shape
        if (grid < 1).any() or (grid >= (width - 1, height - 1)).any():
            raise ValueError("送料线离图像边缘太近，请在画面内侧设置。")
        self.map_x, self.map_y = cv2.convertMaps(grid[:, :, 0], grid[:, :, 1], cv2.CV_16SC2)
        reference = self.gradient(gray)[SEARCH:SEARCH+2*HALF_TEMPLATE+1]
        self.templates = np.split(reference, SEGMENTS, axis=1)
        if sum(np.max(np.abs(segment), axis=0).mean() >= 10 for segment in self.templates) < 4:
            raise ValueError("线附近的前沿不清晰，请沿可见的料头边重画。")

    def gradient(self, gray):
        strip = cv2.remap(gray, self.map_x, self.map_y, cv2.INTER_LINEAR)
        smooth = cv2.GaussianBlur(strip, (3, 3), 0.8)
        return cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3, scale=0.125)

    def measure(self, gray):
        found = []
        for strip, reference in zip(np.split(self.gradient(gray), SEGMENTS, axis=1), self.templates):
            response = cv2.matchTemplate(strip, reference, cv2.TM_CCOEFF_NORMED).ravel()
            peak = int(response.argmax())
            score = response[peak]
            if score < 0.80 or peak in (0, len(response) - 1):
                continue
            response[max(0, peak-4):peak+5] = -1
            if score - response.max() >= 0.08:
                found.append(peak - SEARCH)
        if len(found) < 3:
            return None
        median = np.median(found)
        agreement = [offset for offset in found if abs(offset - median) <= 3]
        return float(np.median(agreement)) if len(agreement) >= 3 else None

    def points(self, offset=0, inset=0):
        a, b = self.ends
        points = (a + (b-a)*inset, b - (b-a)*inset)
        return tuple(tuple(np.rint(point + self.normal*offset).astype(int)) for point in points)

    def draw(self, image, offset, color):
        left, right = self.points()
        limits = [self.points(-TOLERANCE), self.points(TOLERANCE)]
        for start, end in limits:
            cv2.line(image, start, end, (193, 179, 92), 1, cv2.LINE_AA)
        cv2.line(image, left, right, (233, 210, 15), 2, cv2.LINE_AA)
        for end in (left, right):
            cv2.circle(image, end, 3, (233, 210, 15), -1, cv2.LINE_AA)
        if offset is not None:
            start, end = self.points(offset, inset=0.12)
            cv2.line(image, start, end, color, 2, cv2.LINE_AA)

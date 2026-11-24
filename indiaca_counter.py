"""Count Indiaca rally passes in a video with classic OpenCV.

The shuttle is found by colour (red, HSV threshold), separated from static
red objects in the scene, followed from frame to frame and counted every
time it crosses the vertical centre line of the image (where the net is).

Usage:
    python indiaca_counter.py path/to/video.mov
    python indiaca_counter.py path/to/video.mov --tune      # adjust HSV thresholds live
    python indiaca_counter.py path/to/video.mov --no-display  # just print the count
"""

import argparse
import math
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

# All pixel-based thresholds are tuned for 1080p and scaled to the video size.
REFERENCE_WIDTH = 1920


@dataclass
class Settings:
    """Detection and tracking parameters (pixel values are for 1920 px width)."""

    # Red wraps around the hue axis, so two HSV ranges are combined.
    red_low: tuple = ((0, 100, 100), (20, 255, 250))
    red_high: tuple = ((160, 100, 100), (179, 228, 250))

    min_area: float = 100            # contour area, px² at reference size
    max_area: float = 10000
    static_distance: float = 5       # closer than this to an earlier detection -> static object
    static_memory: int = 5           # frames to remember for static-object rejection
    max_jump: float = 200            # max distance between consecutive shuttle positions
    max_turn_angle: float = 45       # max change of flight direction between frames, degrees
    lost_frames_before_reset: int = 10
    frames_to_reacquire: int = 3
    count_window: int = 3            # frames used to judge a centre-line crossing
    min_crossing_move: float = 20    # horizontal movement over the window
    max_crossing_move: float = 400

    def scaled(self, frame_width):
        """Return a copy with pixel thresholds scaled to the given frame width."""
        s = frame_width / REFERENCE_WIDTH
        return Settings(
            red_low=self.red_low,
            red_high=self.red_high,
            min_area=self.min_area * s**2,
            max_area=self.max_area * s**2,
            static_distance=self.static_distance * s,
            static_memory=self.static_memory,
            max_jump=self.max_jump * s,
            max_turn_angle=self.max_turn_angle,
            lost_frames_before_reset=self.lost_frames_before_reset,
            frames_to_reacquire=self.frames_to_reacquire,
            count_window=self.count_window,
            min_crossing_move=self.min_crossing_move * s,
            max_crossing_move=self.max_crossing_move * s,
        )


@dataclass
class Candidate:
    center: tuple
    rect: tuple


def distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def turn_angle(p0, p1, p2):
    """Angle in degrees between the movements p0->p1 and p1->p2, or None if undefined."""
    v1 = np.subtract(p1, p0)
    v2 = np.subtract(p2, p1)
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    if norm == 0:
        return None
    cos_angle = np.clip(np.dot(v1, v2) / norm, -1.0, 1.0)
    return math.degrees(math.acos(cos_angle))


def red_mask(frame, settings):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    low = cv2.inRange(hsv, *settings.red_low)
    high = cv2.inRange(hsv, *settings.red_high)
    return cv2.bitwise_or(low, high)


def detect_candidates(mask, settings):
    """Bounding-box centres of all red blobs within the size limits."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for contour in contours:
        if settings.min_area < cv2.contourArea(contour) < settings.max_area:
            x, y, w, h = cv2.boundingRect(contour)
            candidates.append(Candidate(center=(x + w / 2, y + h / 2), rect=(x, y, w, h)))
    return candidates


class IndiacaTracker:
    """Keeps the tracking state across frames and counts centre-line crossings."""

    def __init__(self, settings, frame_width):
        self.s = settings
        self.center_x = frame_width / 2
        self.previous_detections = deque(maxlen=settings.static_memory)
        self.track = deque(maxlen=2)                 # last accepted positions
        self.recent = deque(maxlen=settings.count_window)
        self.in_turn_recovery = False
        self.lost_frames = 0
        self.reacquire_frames = 0
        self.count = 0

    def update(self, candidates):
        """Process one frame. Returns (shuttle_position_or_None, moving, static)."""
        moving, static = self._split_static(candidates)
        position = self._select(moving)
        position = self._check_direction(position)
        if position is not None:
            self.recent.append(position)
        position = self._handle_reacquire(position)
        if position is not None:
            self._count_crossing()
        return position, moving, static

    def _split_static(self, candidates):
        """Blobs that were already (almost) in the same place recently are static objects."""
        moving, static = [], []
        for c in candidates:
            seen_before = any(
                distance(c.center, old.center) < self.s.static_distance
                for frame in self.previous_detections
                for old in frame
            )
            (static if seen_before else moving).append(c)
        self.previous_detections.append(candidates)
        return moving, static

    def _select(self, moving):
        """Pick the shuttle among the moving blobs."""
        if not moving:
            return None
        if len(moving) == 1:
            position = moving[0].center
            if self.in_turn_recovery:
                self.track.append(position)
            return position
        if not self.track:
            return None
        closest = min(moving, key=lambda c: distance(c.center, self.track[-1]))
        if distance(closest.center, self.track[-1]) < self.s.max_jump:
            return closest.center
        return None

    def _check_direction(self, position):
        """Reject positions that would mean an implausibly sharp change of direction."""
        if position is None:
            return None
        if len(self.track) < 2:
            self.track.append(position)
            return position
        if self.in_turn_recovery:
            self.in_turn_recovery = False
            return position
        angle = turn_angle(self.track[-2], self.track[-1], position)
        if angle is not None and angle < self.s.max_turn_angle:
            self.track.append(position)
            return position
        self.in_turn_recovery = True
        return None

    def _handle_reacquire(self, position):
        """After losing the shuttle for a while, require a few stable frames first."""
        if position is None:
            self.lost_frames += 1
            self.reacquire_frames = 0
            return None
        if self.lost_frames > self.s.lost_frames_before_reset:
            self.reacquire_frames += 1
            if self.reacquire_frames < self.s.frames_to_reacquire:
                return None
            self.lost_frames = 0
        return position

    def _count_crossing(self):
        if len(self.recent) < self.s.count_window:
            return
        previous_x, current_x = self.recent[-2][0], self.recent[-1][0]
        movement = self.recent[-1][0] - self.recent[0][0]
        lo, hi = self.s.min_crossing_move, self.s.max_crossing_move
        if previous_x < self.center_x <= current_x and lo < movement < hi:
            self.count += 1
        elif previous_x > self.center_x >= current_x and -hi < movement < -lo:
            self.count += 1


def draw_overlay(frame, position, moving, static, count, scale):
    for c in static:
        x, y, w, h = c.rect
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
    for c in moving:
        x, y, w, h = c.rect
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    if position is not None:
        r = 15 * scale
        x, y = position
        cv2.rectangle(frame, (int(x - r), int(y - r)), (int(x + r), int(y + r)), (255, 0, 0), -1)
    cv2.putText(frame, f"Count: {count}", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)


# --- HSV tuning window -------------------------------------------------------

def create_tuning_window(settings):
    cv2.namedWindow("Controls")
    (h0, s0, v0), (h1, s1, v1) = settings.red_low
    (h2, _, _), _ = settings.red_high
    for name, value, maximum in [
        ("Hue low max", h1, 179), ("Hue high min", h2, 179),
        ("Sat min", s0, 255), ("Sat max", s1, 255),
        ("Val min", v0, 255), ("Val max", v1, 255),
    ]:
        cv2.createTrackbar(name, "Controls", value, maximum, lambda _: None)


def read_tuning_window(settings):
    get = lambda name: cv2.getTrackbarPos(name, "Controls")
    s_min, s_max, v_min, v_max = get("Sat min"), get("Sat max"), get("Val min"), get("Val max")
    settings.red_low = ((0, s_min, v_min), (get("Hue low max"), s_max, v_max))
    settings.red_high = ((get("Hue high min"), s_min, v_min), (179, s_max, v_max))


# --- main --------------------------------------------------------------------

def run(video, settings, display=True, tune=False, verbose=False):
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    scale = width / REFERENCE_WIDTH
    scaled = settings.scaled(width)
    tracker = IndiacaTracker(scaled, width)

    if tune:
        create_tuning_window(scaled)
    paused = False
    frame = None
    frame_number = 0

    while True:
        if not paused or frame is None:
            ok, frame = cap.read()
            if not ok:
                break
            frame_number += 1
            if tune:
                read_tuning_window(scaled)
            mask = red_mask(frame, scaled)
            position, moving, static = tracker.update(detect_candidates(mask, scaled))
            if verbose:
                print(f"frame {frame_number}: position={position} count={tracker.count}")

        if display:
            shown = frame.copy()
            draw_overlay(shown, position, moving, static, tracker.count, scale)
            cv2.imshow("Indiaca counter", shown)
            if tune:
                cv2.imshow("Mask", mask)
            key = cv2.waitKey(30 if paused else 33) & 0xFF
            if key == 27:           # Esc
                break
            if key == ord(" "):     # Space: pause / resume
                paused = not paused
            if paused and tune:     # re-evaluate the paused frame while tuning
                read_tuning_window(scaled)
                mask = red_mask(frame, scaled)
                cv2.imshow("Mask", mask)

    cap.release()
    if display:
        cv2.destroyAllWindows()
    if tune:
        print(f"HSV ranges: low={scaled.red_low} high={scaled.red_high}")
    return tracker.count


def main():
    parser = argparse.ArgumentParser(description="Count Indiaca rally passes in a video.")
    parser.add_argument("video", help="path to a video file")
    parser.add_argument("--tune", action="store_true", help="show sliders to adjust the HSV thresholds")
    parser.add_argument("--no-display", action="store_true", help="run without windows, only print the result")
    parser.add_argument("--verbose", action="store_true", help="print the tracked position for every frame")
    args = parser.parse_args()

    count = run(args.video, Settings(), display=not args.no_display, tune=args.tune, verbose=args.verbose)
    print(f"Counted passes: {count}")


if __name__ == "__main__":
    main()

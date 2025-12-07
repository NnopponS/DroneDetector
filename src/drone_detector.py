#!/usr/bin/env python3
"""
Drone detection, classification (drone vs bird), geolocation regression, and
video tracking with speed estimation.

Key modes (see CLI at bottom):
- build-shape: create Hu-moment shape codes from ./Drones and ./Birds.
- train-geo: fit pixel->lat/lon/alt regression from ./P2_DATA_TRAIN.
- detect-image: run detection on a still image.
- detect-video: run detection/tracking on a video and render trajectories.
"""
from __future__ import annotations

import argparse
import base64
import json
import math
import sys
import threading
import time
from types import SimpleNamespace
import tkinter as tk
from tkinter import filedialog, messagebox
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np


# --------- Shape code utilities ---------


def _to_log_hu(cnt: np.ndarray) -> Optional[np.ndarray]:
    """Return log-scale Hu moments for a contour or None if degenerate."""
    if cnt is None or len(cnt) < 5:
        return None
    moments = cv2.moments(cnt)
    hu = cv2.HuMoments(moments)
    if hu is None:
        return None
    hu = np.squeeze(hu)
    # Log transform with sign preservation makes distance metrics usable.
    return np.sign(hu) * np.log1p(np.abs(hu))


def _largest_contour_from_image(img_path: Path) -> Optional[np.ndarray]:
    """Extract the largest contour from an image path."""
    image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = cv2.bitwise_not(binary)
    binary = cv2.dilate(binary, np.ones((3, 3), np.uint8), iterations=2)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


class ShapeCodeLibrary:
    """
    Builds and stores shape-code centroids for drones and birds using Hu moments.
    """

    def __init__(
        self,
        drone_dir: Path | str = Path("Drones"),
        bird_dir: Path | str = Path("Birds"),
        cache_path: Path | str = Path("shape_codes.json"),
    ) -> None:
        self.drone_dir = Path(drone_dir)
        self.bird_dir = Path(bird_dir)
        self.cache_path = Path(cache_path)
        self.centroids: Dict[str, np.ndarray] = {}
        self.radii: Dict[str, float] = {}

    @property
    def ready(self) -> bool:
        return bool(self.centroids)

    def build(self, max_samples: Optional[int] = None) -> None:
        datasets = {"drone": self.drone_dir, "bird": self.bird_dir}
        centroids: Dict[str, np.ndarray] = {}
        radii: Dict[str, float] = {}
        for label, folder in datasets.items():
            hu_vectors: List[np.ndarray] = []
            paths = sorted(folder.glob("*"))
            if max_samples:
                paths = paths[:max_samples]
            for img_path in paths:
                if not img_path.is_file():
                    continue
                cnt = _largest_contour_from_image(img_path)
                hu = _to_log_hu(cnt) if cnt is not None else None
                if hu is not None:
                    hu_vectors.append(hu)
            if not hu_vectors:
                raise RuntimeError(f"No usable contours found in {folder}")
            mat = np.vstack(hu_vectors)
            centroid = np.mean(mat, axis=0)
            # Typical radius based on 80th percentile distance to centroid.
            dists = np.linalg.norm(mat - centroid, axis=1)
            radius = float(np.percentile(dists, 80) + 0.05)
            centroids[label] = centroid
            radii[label] = radius
        self.centroids = centroids
        self.radii = radii

    def save(self) -> None:
        payload = {
            "centroids": {k: v.tolist() for k, v in self.centroids.items()},
            "radii": self.radii,
        }
        with open(self.cache_path, "w", encoding="utf-8") as fp:
            json.dump(payload, fp, indent=2)

    def load(self) -> None:
        with open(self.cache_path, "r", encoding="utf-8") as fp:
            data = json.load(fp)
        self.centroids = {k: np.array(v, dtype=np.float32) for k, v in data["centroids"].items()}
        self.radii = {k: float(v) for k, v in data["radii"].items()}

    def ensure_ready(self) -> None:
        if self.ready:
            return
        if self.cache_path.exists():
            self.load()
        else:
            self.build()
            self.save()

    def classify(self, contour: np.ndarray) -> Tuple[str, float, float]:
        """
        Return (label, confidence, distance). label can be 'drone', 'bird', or 'unknown'.
        """
        self.ensure_ready()
        code = _to_log_hu(contour)
        if code is None:
            return "unknown", 0.0, float("inf")
        best_label = "unknown"
        best_dist = float("inf")
        for label, centroid in self.centroids.items():
            dist = float(np.linalg.norm(code - centroid))
            if dist < best_dist:
                best_dist = dist
                best_label = label
        radius = self.radii.get(best_label, 1.0)
        if best_dist > radius * 1.3:
            return "unknown", 0.0, best_dist
        confidence = 1.0 / (1.0 + best_dist)
        return best_label, confidence, best_dist


# --------- Geolocation regression ---------


def _fit_linear_3d(samples: List[Tuple[float, float, float, float, float]]) -> np.ndarray:
    """
    Fit [x,y,1] -> [lat, lon, alt] affine matrix using least squares.
    """
    if len(samples) < 3:
        raise RuntimeError("Need at least 3 samples to fit geolocation regressor")
    A: List[List[float]] = []
    b_lat: List[float] = []
    b_lon: List[float] = []
    b_alt: List[float] = []
    for x, y, lat, lon, alt in samples:
        A.append([x, y, 1.0])
        b_lat.append(lat)
        b_lon.append(lon)
        b_alt.append(alt)
    A_mat = np.array(A, dtype=np.float64)
    mat_lat, _, _, _ = np.linalg.lstsq(A_mat, np.array(b_lat), rcond=None)
    mat_lon, _, _, _ = np.linalg.lstsq(A_mat, np.array(b_lon), rcond=None)
    mat_alt, _, _, _ = np.linalg.lstsq(A_mat, np.array(b_alt), rcond=None)
    return np.vstack([mat_lat, mat_lon, mat_alt])


class GeolocationRegressor:
    """
    Maps pixel coordinates to lat/lon/alt using an affine model.
    """

    def __init__(self, model_path: Path | str = Path("geo_model.npz")) -> None:
        self.model_path = Path(model_path)
        self.matrix: Optional[np.ndarray] = None

    def predict(self, x: float, y: float) -> Optional[Tuple[float, float, float]]:
        if self.matrix is None:
            if self.model_path.exists():
                self.load()
            else:
                return None
        vec = np.array([x, y, 1.0], dtype=np.float64)
        lat, lon, alt = self.matrix @ vec
        return float(lat), float(lon), float(alt)

    def fit(self, samples: List[Tuple[float, float, float, float, float]]) -> None:
        self.matrix = _fit_linear_3d(samples)

    def save(self) -> None:
        if self.matrix is None:
            raise RuntimeError("Model not fitted")
        np.savez(self.model_path, matrix=self.matrix)

    def load(self) -> None:
        data = np.load(self.model_path)
        self.matrix = data["matrix"]


# --------- Detection ---------


@dataclass
class Detection:
    bbox: Tuple[int, int, int, int]
    center: Tuple[int, int]
    area: float
    contour: np.ndarray
    label: str
    confidence: float
    lat: Optional[float] = None
    lon: Optional[float] = None
    alt: Optional[float] = None


class DroneDetector:
    """
    Combines contour-based detection with shape-code classification.
    """

    def __init__(
        self,
        shape_lib: Optional[ShapeCodeLibrary] = None,
        geo_regressor: Optional[GeolocationRegressor] = None,
    ) -> None:
        self.shape_lib = shape_lib or ShapeCodeLibrary()
        self.geo_regressor = geo_regressor

    def _find_contours(self, image: np.ndarray) -> Tuple[List[np.ndarray], np.ndarray]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(blurred, 40, 255, cv2.THRESH_BINARY)
        binary = cv2.bitwise_not(binary)
        dilated = cv2.dilate(binary, np.ones((3, 3), np.uint8), iterations=3)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours, dilated

    def detect(
        self,
        image: np.ndarray,
        *,
        debug: bool = False,
        min_area: int = 40,
        min_conf: float = 0.15,
    ) -> Tuple[List[Detection], np.ndarray]:
        detections: List[Detection] = []
        contours, dilated = self._find_contours(image)
        if not contours:
            return detections, dilated
        h, w = image.shape[:2]
        horizon = int(0.35 * h)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            x, y, cw, ch = cv2.boundingRect(cnt)
            if y > h - horizon:
                continue
            if ch > h * 0.5:
                continue
            center_x = int(x + cw / 2)
            center_y = int(y + ch / 2)
            label, confidence, _ = self.shape_lib.classify(cnt)
            if label != "drone" or confidence < min_conf:
                continue
            lat = lon = alt = None
            if self.geo_regressor:
                loc = self.geo_regressor.predict(center_x, center_y)
                if loc:
                    lat, lon, alt = loc
            detections.append(
                Detection(
                    bbox=(x, y, cw, ch),
                    center=(center_x, center_y),
                    area=float(area),
                    contour=cnt,
                    label=label,
                    confidence=confidence,
                    lat=lat,
                    lon=lon,
                    alt=alt,
                )
            )
        if debug:
            print(f"[detect] found {len(detections)} drones from {len(contours)} contours")
        return detections, dilated


# --------- Tracking for video ---------


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in meters."""
    r = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = phi2 - phi1
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * r * math.atan2(math.sqrt(a), math.sqrt(1 - a))


@dataclass
class Track:
    track_id: int
    color: Tuple[int, int, int]
    last_seen: int
    hits: int = 0
    misses: int = 0
    history_px: deque = field(default_factory=lambda: deque(maxlen=256))
    history_geo: deque = field(default_factory=lambda: deque(maxlen=256))
    last_bbox: Optional[Tuple[int, int, int, int]] = None
    speed_mps: Optional[float] = None


class CentroidTracker:
    """Simple nearest-centroid tracker."""

    def __init__(self, max_misses: int = 15, dist_thresh: float = 60.0) -> None:
        self.max_misses = max_misses
        self.dist_thresh = dist_thresh
        self.tracks: Dict[int, Track] = {}
        self._next_id = 0

    def _new_color(self, idx: int) -> Tuple[int, int, int]:
        rng = np.random.default_rng(idx + 13)
        return tuple(int(c) for c in rng.integers(0, 255, size=3))

    def update(
        self, detections: List[Detection], frame_idx: int, timestamp: float
    ) -> Dict[int, Track]:
        # Age tracks
        for trk in self.tracks.values():
            trk.misses += 1
        # Associate by nearest neighbor
        for det in detections:
            cx, cy = det.center
            best_id = None
            best_dist = self.dist_thresh
            for tid, trk in self.tracks.items():
                px, py = trk.history_px[-1] if trk.history_px else (cx, cy)
                dist = math.hypot(px - cx, py - cy)
                if dist < best_dist:
                    best_dist = dist
                    best_id = tid
            if best_id is None:
                tid = self._next_id
                self._next_id += 1
                trk = Track(track_id=tid, color=self._new_color(tid), last_seen=frame_idx)
                self.tracks[tid] = trk
            else:
                trk = self.tracks[best_id]
            trk.history_px.append((cx, cy))
            if det.lat is not None and det.lon is not None and det.alt is not None:
                trk.history_geo.append((timestamp, det.lat, det.lon, det.alt))
                if len(trk.history_geo) >= 2:
                    t0, la0, lo0, al0 = trk.history_geo[-2]
                    t1, la1, lo1, al1 = trk.history_geo[-1]
                    dt = max(t1 - t0, 1e-6)
                    dist = _haversine_m(la0, lo0, la1, lo1)
                    dist = math.hypot(dist, al1 - al0)
                    trk.speed_mps = dist / dt
            trk.last_bbox = det.bbox
            trk.last_seen = frame_idx
            trk.hits += 1
            trk.misses = 0
        # Remove stale tracks
        dead_ids = [tid for tid, trk in self.tracks.items() if trk.misses > self.max_misses]
        for tid in dead_ids:
            self.tracks.pop(tid, None)
        return self.tracks


def _draw_tracks(frame: np.ndarray, tracks: Dict[int, Track]) -> None:
    for trk in tracks.values():
        if trk.last_bbox:
            x, y, w, h = trk.last_bbox
            cv2.rectangle(frame, (x, y), (x + w, y + h), trk.color, 2)
            cv2.putText(
                frame,
                f"ID:{trk.track_id}",
                (x, max(15, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                trk.color,
                2,
                cv2.LINE_AA,
            )
        if len(trk.history_px) >= 2:
            pts = np.array(trk.history_px, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(frame, [pts], False, trk.color, 2, cv2.LINE_AA)
        if trk.speed_mps is not None and trk.last_bbox:
            x, y, _, _ = trk.last_bbox
            cv2.putText(
                frame,
                f"{trk.speed_mps:0.1f} m/s",
                (x, y + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                trk.color,
                1,
                cv2.LINE_AA,
            )


def _draw_overlay(frame: np.ndarray, tracks: Dict[int, Track], fps: float) -> None:
    y = 20
    for trk in sorted(tracks.values(), key=lambda t: t.track_id):
        if not trk.history_geo:
            continue
        _, lat, lon, alt = trk.history_geo[-1]
        cv2.putText(
            frame,
            f"track_id:{trk.track_id}",
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            trk.color,
            1,
            cv2.LINE_AA,
        )
        y += 15
        cv2.putText(frame, f"- lat:{lat:.5f}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, trk.color, 1, cv2.LINE_AA)
        y += 15
        cv2.putText(frame, f"- lon:{lon:.5f}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, trk.color, 1, cv2.LINE_AA)
        y += 15
        cv2.putText(frame, f"- alt:{alt:.2f}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, trk.color, 1, cv2.LINE_AA)
        y += 15
        if trk.speed_mps is not None:
            cv2.putText(
                frame,
                f"- speed:{trk.speed_mps:0.2f} m/s",
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                trk.color,
                1,
                cv2.LINE_AA,
            )
        y += 20
    cv2.putText(
        frame,
        f"{fps:0.1f} FPS",
        (frame.shape[1] - 90, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


# --------- Public API mirroring legacy name ---------


def image_process_drone(
    img_path: str,
    CSV: bool = False,
    visual: bool = True,
    dil: bool = False,
    DEBUG: bool = False,
    bgr: Tuple[int, int, int] = (0, 0, 255),
    DETECT: bool = False,
    shape_library: Optional[ShapeCodeLibrary] = None,
    geo_regressor: Optional[GeolocationRegressor] = None,
) -> Tuple[np.ndarray, Optional[str]]:
    """
    Detect drones in a still image. Returns (image, csv_text?) similar to the
    original signature. CSV contains rows: filename,center_x,center_y,width,height,lat,lon,alt.
    """
    if not img_path:
        raise ValueError("No path")
    image = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"No image at {img_path}")
    detector = DroneDetector(shape_library, geo_regressor)
    detections, dilated = detector.detect(image, debug=DEBUG)
    csv_lines: List[str] = []
    for det in detections:
        x, y, w, h = det.bbox
        if DETECT:
            cv2.rectangle(image, (x, y), (x + w, y + h), bgr, 2)
            cv2.putText(
                image,
                f"ID:{det.label}",
                (x, max(15, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                bgr,
                1,
                cv2.LINE_AA,
            )
        row = [
            Path(img_path).name,
            det.center[0],
            det.center[1],
            det.bbox[2],
            det.bbox[3],
        ]
        if det.lat is not None:
            row.extend([f"{det.lat:.6f}", f"{det.lon:.6f}", f"{det.alt:.2f}"])
        csv_lines.append(",".join(map(str, row)))
    if DEBUG:
        print(csv_lines)
    if visual:
        cv2.imshow("dilated" if dil else "Color Image with Rectangle", dilated if dil else image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    if CSV:
        return (dilated if dil else image), "\n".join(csv_lines) + ("\n" if csv_lines else "")
    return image, None


# --------- Training helpers ---------


def run_shape_build(args: argparse.Namespace) -> None:
    lib = ShapeCodeLibrary(args.drone_dir, args.bird_dir, args.cache)
    lib.build(max_samples=args.limit)
    lib.save()
    print(f"[ok] shape codes saved to {args.cache}")


def run_geo_train(args: argparse.Namespace) -> None:
    shape_lib = ShapeCodeLibrary(args.drone_dir, args.bird_dir, args.shape_cache)
    shape_lib.ensure_ready()
    detector = DroneDetector(shape_lib)
    samples: List[Tuple[float, float, float, float, float]] = []
    train_dir = Path(args.data_dir)
    for img_path in sorted(train_dir.glob("*.jpg")):
        csv_path = img_path.with_suffix(".csv")
        if not csv_path.exists():
            continue
        with open(csv_path, "r", encoding="utf-8") as fp:
            lines = fp.read().strip().splitlines()
        if len(lines) < 2:
            continue
        lat, lon, alt = map(float, lines[1].split(","))
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        detections, _ = detector.detect(image, debug=False)
        if not detections:
            continue
        det = max(detections, key=lambda d: d.area)
        samples.append((det.center[0], det.center[1], lat, lon, alt))
    if not samples:
        raise RuntimeError("No samples found for geolocation training")
    geo = GeolocationRegressor(args.output)
    geo.fit(samples)
    geo.save()
    print(f"[ok] geolocation model saved to {args.output} with {len(samples)} samples")


# --------- Video processing ---------


def run_video(args: argparse.Namespace) -> None:
    shape_lib = ShapeCodeLibrary(args.drone_dir, args.bird_dir, args.shape_cache)
    shape_lib.ensure_ready()
    geo = None
    if args.geo_model and Path(args.geo_model).exists():
        geo = GeolocationRegressor(args.geo_model)
        geo.load()
    detector = DroneDetector(shape_lib, geo)
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(
        str(args.output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    tracker = CentroidTracker(max_misses=15, dist_thresh=80.0)
    frame_idx = 0
    last_time = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        timestamp = frame_idx / fps
        detections, _ = detector.detect(frame, debug=False, min_conf=0.12)
        tracks = tracker.update(detections, frame_idx, timestamp)
        _draw_tracks(frame, tracks)
        now = time.time()
        inst_fps = 1.0 / max(now - last_time, 1e-6)
        last_time = now
        _draw_overlay(frame, tracks, inst_fps)
        writer.write(frame)
        if args.display:
            cv2.imshow("video", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        frame_idx += 1
    cap.release()
    writer.release()
    if args.display:
        cv2.destroyAllWindows()
    print(f"[ok] video saved to {args.output}")


# --------- CLI ---------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Drone detection toolkit")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ui = sub.add_parser("ui", help="เปิด UI ไม่ใช้ terminal")
    p_ui.set_defaults(func=run_ui)

    p_shape = sub.add_parser("build-shape", help="build shape code cache from Drones/Birds")
    p_shape.add_argument("--drone-dir", default="Drones")
    p_shape.add_argument("--bird-dir", default="Birds")
    p_shape.add_argument("--cache", default="shape_codes.json")
    p_shape.add_argument("--limit", type=int, default=None, help="max samples per class")
    p_shape.set_defaults(func=run_shape_build)

    p_geo = sub.add_parser("train-geo", help="train pixel->lat/lon/alt regressor")
    p_geo.add_argument("--data-dir", default="P2_DATA_TRAIN")
    p_geo.add_argument("--drone-dir", default="Drones")
    p_geo.add_argument("--bird-dir", default="Birds")
    p_geo.add_argument("--shape-cache", default="shape_codes.json")
    p_geo.add_argument("--output", default="geo_model.npz")
    p_geo.set_defaults(func=run_geo_train)

    p_img = sub.add_parser("detect-image", help="detect drone(s) on a still image")
    p_img.add_argument("--image", required=True)
    p_img.add_argument("--shape-cache", default="shape_codes.json")
    p_img.add_argument("--geo-model", default=None)
    p_img.add_argument("--save", default=None, help="optional output image")
    p_img.add_argument("--no-visual", action="store_true")
    p_img.set_defaults(func=lambda args: _run_detect_image(args))

    p_vid = sub.add_parser("detect-video", help="process video with tracking")
    p_vid.add_argument("--video", required=True)
    p_vid.add_argument("--output", default="output.mp4")
    p_vid.add_argument("--shape-cache", default="shape_codes.json")
    p_vid.add_argument("--geo-model", default=None)
    p_vid.add_argument("--display", action="store_true")
    p_vid.add_argument("--drone-dir", default="Drones")
    p_vid.add_argument("--bird-dir", default="Birds")
    p_vid.set_defaults(func=run_video)

    return parser


def _run_detect_image(args: argparse.Namespace) -> None:
    shape_lib = ShapeCodeLibrary(cache_path=args.shape_cache)
    shape_lib.ensure_ready()
    geo = None
    if args.geo_model and Path(args.geo_model).exists():
        geo = GeolocationRegressor(args.geo_model)
        geo.load()
    image, csv_text = image_process_drone(
        args.image,
        CSV=True,
        visual=not args.no_visual,
        DETECT=True,
        shape_library=shape_lib,
        geo_regressor=geo,
    )
    if args.save:
        cv2.imwrite(args.save, image)
        print(f"[ok] saved annotated image to {args.save}")
    if csv_text:
        print(csv_text.strip())


class UIApp:
    """Minimal Tkinter UI wrapper around the CLI actions."""

    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("Drone Detector UI")
        self.drone_dir = tk.StringVar(value="Drones")
        self.bird_dir = tk.StringVar(value="Birds")
        self.shape_cache = tk.StringVar(value="shape_codes.json")
        self.data_dir = tk.StringVar(value="P2_DATA_TRAIN")
        self.geo_model = tk.StringVar(value="geo_model.npz")
        self.image_path = tk.StringVar(value="")
        self.video_path = tk.StringVar(value="")
        self.video_out = tk.StringVar(value="annotated_video.mp4")
        self.preview_img = None
        self._build_layout()
        self._add_hint()

    def log(self, text: str) -> None:
        self.log_box.insert(tk.END, text + "\n")
        self.log_box.see(tk.END)

    def _browse_file(self, var: tk.StringVar, title: str, filetypes: tuple) -> None:
        path = filedialog.askopenfilename(title=title, filetypes=filetypes)
        if path:
            var.set(path)

    def _browse_dir(self, var: tk.StringVar, title: str) -> None:
        path = filedialog.askdirectory(title=title)
        if path:
            var.set(path)

    def _build_layout(self) -> None:
        frm = tk.Frame(self.root, padx=8, pady=8)
        frm.grid(row=0, column=0, sticky="nsew")

        def add_row(label: str, var: tk.StringVar, browse=None) -> int:
            row = add_row.idx
            tk.Label(frm, text=label, anchor="w").grid(row=row, column=0, sticky="w", pady=2)
            tk.Entry(frm, textvariable=var, width=60).grid(row=row, column=1, sticky="we", pady=2)
            if browse == "dir":
                tk.Button(frm, text="เลือก...", command=lambda v=var: self._browse_dir(v, f"เลือก {label}")).grid(
                    row=row, column=2, padx=4
                )
            elif browse == "file":
                tk.Button(
                    frm,
                    text="เลือก...",
                    command=lambda v=var: self._browse_file(v, f"เลือก {label}", (("All", "*.*"),)),
                ).grid(row=row, column=2, padx=4)
            add_row.idx += 1
            return row

        add_row.idx = 0
        add_row("โฟลเดอร์ Drones", self.drone_dir, browse="dir")
        add_row("โฟลเดอร์ Birds", self.bird_dir, browse="dir")
        add_row("ไฟล์ shape cache", self.shape_cache, browse="file")
        add_row("โฟลเดอร์ train geo", self.data_dir, browse="dir")
        add_row("ไฟล์ geo model", self.geo_model, browse="file")
        add_row("ไฟล์รูป", self.image_path, browse="file")
        add_row("ไฟล์วิดีโอ", self.video_path, browse="file")
        add_row("บันทึกวิดีโอออก", self.video_out, browse="file")

        tk.Button(frm, text="สร้าง Shape Codes", command=self._run_shape_codes).grid(row=add_row.idx, column=0, pady=6)
        tk.Button(frm, text="เทรน Geo Model", command=self._run_geo_train).grid(row=add_row.idx, column=1, pady=6)
        add_row.idx += 1
        tk.Button(frm, text="ตรวจจับรูป", command=self._run_detect_image).grid(row=add_row.idx, column=0, pady=6)
        tk.Button(frm, text="ตรวจจับวิดีโอ", command=self._run_detect_video).grid(row=add_row.idx, column=1, pady=6)
        add_row.idx += 1

        self.log_box = tk.Text(frm, height=10, width=90)
        self.log_box.grid(row=add_row.idx, column=0, columnspan=3, pady=4, sticky="we")
        add_row.idx += 1

        self.preview_label = tk.Label(frm, text="Preview จะโชว์ตรงนี้", anchor="center", relief="groove", height=16)
        self.preview_label.grid(row=add_row.idx, column=0, columnspan=3, pady=4, sticky="we")

        self.root.grid_columnconfigure(0, weight=1)
        frm.grid_columnconfigure(1, weight=1)
        self._maybe_bootstrap()

    def _add_hint(self) -> None:
        hint = (
            "ขั้นตอนง่าย ๆ: 1) เลือกรูปหรือวิดีโอ 2) ถ้ายังไม่เคย กดสร้าง Shape Codes/Geo Model "
            "3) กด ตรวจจับรูป หรือ ตรวจจับวิดีโอ"
        )
        self.log(hint)

    def _maybe_bootstrap(self) -> None:
        """ถ้ายังไม่มี shape cache สร้างให้อัตโนมัติ (จำกัดตัวอย่างเพื่อความเร็ว)."""
        if Path(self.shape_cache.get()).exists():
            return
        self.log("[info] ยังไม่มี shape cache กำลังสร้างให้อัตโนมัติ...")

        def work() -> None:
            try:
                args = SimpleNamespace(
                    drone_dir=self.drone_dir.get(),
                    bird_dir=self.bird_dir.get(),
                    cache=self.shape_cache.get(),
                    limit=150,  # เร็วขึ้นสำหรับครั้งแรก
                )
                run_shape_build(args)
                self.log(f"[ok] สร้าง shape cache -> {args.cache}")
            except Exception as exc:
                self.log(f"[error] bootstrap: {exc}")

        self._thread(work)

    def _thread(self, fn) -> None:
        threading.Thread(target=fn, daemon=True).start()

    def _update_preview(self, image: np.ndarray) -> None:
        """Render an OpenCV image into the preview label without saving."""
        try:
            success, buf = cv2.imencode(".png", image)
            if not success:
                self.log("[error] แปลงภาพไม่สำเร็จ")
                return
            b64 = base64.b64encode(buf).decode("ascii")
            photo = tk.PhotoImage(data=b64)
            self.preview_img = photo  # keep reference
            self.preview_label.configure(image=photo, text="")
        except Exception as exc:
            self.log(f"[error] preview: {exc}")

    def _run_shape_codes(self) -> None:
        def work() -> None:
            try:
                args = SimpleNamespace(
                    drone_dir=self.drone_dir.get(),
                    bird_dir=self.bird_dir.get(),
                    cache=self.shape_cache.get(),
                    limit=None,
                )
                run_shape_build(args)
                self.log(f"[ok] shape codes -> {args.cache}")
            except Exception as exc:
                messagebox.showerror("Error", str(exc))
                self.log(f"[error] {exc}")

        self._thread(work)

    def _run_geo_train(self) -> None:
        def work() -> None:
            try:
                args = SimpleNamespace(
                    data_dir=self.data_dir.get(),
                    drone_dir=self.drone_dir.get(),
                    bird_dir=self.bird_dir.get(),
                    shape_cache=self.shape_cache.get(),
                    output=self.geo_model.get(),
                )
                run_geo_train(args)
                self.log(f"[ok] geo model -> {args.output}")
            except Exception as exc:
                messagebox.showerror("Error", str(exc))
                self.log(f"[error] {exc}")

        self._thread(work)

    def _run_detect_image(self) -> None:
        if not self.image_path.get():
            messagebox.showwarning("ขาดข้อมูล", "เลือกไฟล์รูปก่อน")
            return

        def work() -> None:
            try:
                args = SimpleNamespace(
                    image=self.image_path.get(),
                    shape_cache=self.shape_cache.get(),
                    geo_model=self.geo_model.get() if Path(self.geo_model.get()).exists() else None,
                    no_visual=True,
                )
                image, _ = image_process_drone(
                    args.image,
                    CSV=True,
                    visual=False,
                    DETECT=True,
                    shape_library=ShapeCodeLibrary(cache_path=args.shape_cache),
                    geo_regressor=GeolocationRegressor(args.geo_model) if args.geo_model else None,
                )
                self.root.after(0, lambda img=image: self._update_preview(img))
                self.log("[ok] แสดง preview แล้ว (ไม่เซฟไฟล์)")
            except Exception as exc:
                messagebox.showerror("Error", str(exc))
                self.log(f"[error] {exc}")

        self._thread(work)

    def _run_detect_video(self) -> None:
        if not self.video_path.get():
            messagebox.showwarning("ขาดข้อมูล", "เลือกไฟล์วิดีโอก่อน")
            return

        def work() -> None:
            try:
                args = SimpleNamespace(
                    video=self.video_path.get(),
                    output=self.video_out.get(),
                    shape_cache=self.shape_cache.get(),
                    geo_model=self.geo_model.get() if Path(self.geo_model.get()).exists() else None,
                    display=False,
                    drone_dir=self.drone_dir.get(),
                    bird_dir=self.bird_dir.get(),
                )
                run_video(args)
                self.log(f"[ok] บันทึกวิดีโอที่ {args.output}")
            except Exception as exc:
                messagebox.showerror("Error", str(exc))
                self.log(f"[error] {exc}")

        self._thread(work)

    def run(self) -> None:
        self.root.mainloop()


def run_ui(_: argparse.Namespace | None = None) -> None:
    app = UIApp()
    app.run()


def main() -> None:
    parser = build_arg_parser()
    if len(sys.argv) == 1:
        run_ui(None)
        return
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

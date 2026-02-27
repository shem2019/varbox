"""OpenCV calibration CLI for chessboard captures."""

from __future__ import annotations

import argparse
import glob
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np

from boxing_analytics.calibration.models import CalibrationProfile, save_profile

NDArray = np.ndarray[Any, Any]


@dataclass(frozen=True, slots=True)
class CalibrationInputs:
    image_paths: list[str]
    image_size: tuple[int, int]
    object_points: list[NDArray]
    image_points: list[NDArray]


def _build_object_point_grid(board_cols: int, board_rows: int, square_size_mm: float) -> NDArray:
    grid = np.zeros((board_rows * board_cols, 3), np.float32)
    grid[:, :2] = np.mgrid[0:board_cols, 0:board_rows].T.reshape(-1, 2)
    grid *= float(square_size_mm)
    return grid


def _parse_ring_points(
    ring_points_path: str | None,
) -> tuple[NDArray, NDArray] | None:
    if not ring_points_path:
        return None
    path = Path(ring_points_path)
    payload = path.read_text(encoding="utf-8")
    import json

    data = json.loads(payload)
    if not isinstance(data, dict):
        raise ValueError("ring points file must be a JSON object")
    ring = np.asarray(data.get("ring_points", []), dtype=np.float32)
    image = np.asarray(data.get("image_points", []), dtype=np.float32)
    if ring.shape[0] < 4 or image.shape[0] < 4:
        raise ValueError("ring_points and image_points require at least 4 points each")
    if ring.shape != image.shape or ring.shape[1] != 2:
        raise ValueError("ring_points and image_points must have matching Nx2 shape")
    return ring, image


def collect_calibration_inputs(
    *,
    images_glob: str,
    board_cols: int,
    board_rows: int,
    square_size_mm: float,
) -> CalibrationInputs:
    image_paths = sorted(glob.glob(images_glob))
    if not image_paths:
        raise FileNotFoundError(f"No calibration images matched: {images_glob}")

    pattern_size = (board_cols, board_rows)
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.001,
    )
    object_template = _build_object_point_grid(board_cols, board_rows, square_size_mm)
    object_points: list[NDArray] = []
    image_points: list[NDArray] = []
    image_size: tuple[int, int] | None = None
    used_paths: list[str] = []

    for path in image_paths:
        image = cast(Any, cv2.imread(path))
        if image is None:
            continue
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(
            gray,
            pattern_size,
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
        )
        if not found:
            continue
        refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        object_points.append(object_template.copy())
        image_points.append(refined)
        image_size = (gray.shape[1], gray.shape[0])
        used_paths.append(path)

    if image_size is None or len(object_points) < 3:
        raise RuntimeError("Calibration needs at least 3 valid chessboard detections")

    return CalibrationInputs(
        image_paths=used_paths,
        image_size=image_size,
        object_points=object_points,
        image_points=image_points,
    )


def _reprojection_error(
    *,
    object_points: list[NDArray],
    image_points: list[NDArray],
    rvecs: Sequence[NDArray],
    tvecs: Sequence[NDArray],
    camera_matrix: NDArray,
    dist_coeffs: NDArray,
) -> float:
    total_error = 0.0
    total_points = 0
    for idx, obj in enumerate(object_points):
        projected, _ = cv2.projectPoints(obj, rvecs[idx], tvecs[idx], camera_matrix, dist_coeffs)
        err = cv2.norm(image_points[idx], projected, cv2.NORM_L2)
        total_error += float(err * err)
        total_points += int(len(obj))
    if total_points <= 0:
        return 0.0
    return float(np.sqrt(total_error / total_points))


def calibrate_camera_profile(
    *,
    profile_name: str,
    images_glob: str,
    board_cols: int,
    board_rows: int,
    square_size_mm: float,
    ring_points_path: str | None = None,
) -> CalibrationProfile:
    inputs = collect_calibration_inputs(
        images_glob=images_glob,
        board_cols=board_cols,
        board_rows=board_rows,
        square_size_mm=square_size_mm,
    )
    camera_seed: NDArray = np.eye(3, dtype=np.float64)
    dist_seed: NDArray = np.zeros((8, 1), dtype=np.float64)
    _, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        inputs.object_points,
        inputs.image_points,
        inputs.image_size,
        camera_seed,
        dist_seed,
    )
    reproj = _reprojection_error(
        object_points=inputs.object_points,
        image_points=inputs.image_points,
        rvecs=rvecs,
        tvecs=tvecs,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
    )

    ring_homography: list[list[float]] | None = None
    ring_pairs = _parse_ring_points(ring_points_path)
    if ring_pairs is not None:
        ring_points, image_points = ring_pairs
        homography, _ = cv2.findHomography(ring_points, image_points, method=0)
        if homography is not None:
            ring_homography = homography.astype(float).tolist()

    return CalibrationProfile(
        profile_name=profile_name,
        camera_matrix=camera_matrix.astype(float).tolist(),
        dist_coeffs=dist_coeffs.reshape(-1).astype(float).tolist(),
        image_width=inputs.image_size[0],
        image_height=inputs.image_size[1],
        reprojection_error=float(reproj),
        source_images=[str(Path(p).name) for p in inputs.image_paths],
        ring_homography=ring_homography,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Camera calibration utility")
    parser.add_argument("--images-glob", required=True, help="Glob pattern for chessboard images")
    parser.add_argument(
        "--board-cols", type=int, required=True, help="Inner chessboard corners per row"
    )
    parser.add_argument(
        "--board-rows", type=int, required=True, help="Inner chessboard corners per column"
    )
    parser.add_argument(
        "--square-size-mm", type=float, required=True, help="Chessboard square size in mm"
    )
    parser.add_argument(
        "--output", required=True, help="Calibration profile output path (.json/.yaml)"
    )
    parser.add_argument(
        "--profile-name",
        default="",
        help="Profile name. Defaults to output filename stem.",
    )
    parser.add_argument(
        "--ring-points-json",
        default="",
        help="Optional JSON with ring_points/image_points arrays for ring-plane homography.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    output_path = str(args.output)
    profile_name = str(args.profile_name).strip() or Path(output_path).stem
    profile = calibrate_camera_profile(
        profile_name=profile_name,
        images_glob=str(args.images_glob),
        board_cols=int(args.board_cols),
        board_rows=int(args.board_rows),
        square_size_mm=float(args.square_size_mm),
        ring_points_path=str(args.ring_points_json).strip() or None,
    )
    written = save_profile(profile, output_path)
    print(
        f"Saved calibration profile: {written}\n"
        f"Profile={profile.profile_name} "
        f"Resolution={profile.image_width}x{profile.image_height} "
        f"ReprojectionError={profile.reprojection_error:.4f} "
        f"FramesUsed={len(profile.source_images)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

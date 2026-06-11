#!/usr/bin/env python3
"""Render sample frame video and frame-synchronized 3D trajectory video.

Example:
  python scripts/render_sample_trajectory_video.py --sample-dir sample
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


FLOAT_RE = re.compile(r"[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?")


@dataclass(frozen=True)
class Pose:
    frame_id: int | None
    x: float
    y: float
    z: float
    roll: float
    pitch: float
    yaw: float
    gimbal_pitch: float


def numeric_stem(path: Path) -> int | None:
    try:
        return int(path.stem)
    except ValueError:
        return None


def collect_images(sample_dir: Path) -> list[Path]:
    images = [p for p in sample_dir.glob("*.png") if numeric_stem(p) is not None]
    images.sort(key=lambda p: int(p.stem))
    if not images:
        raise FileNotFoundError(f"No numeric PNG frames found in {sample_dir}")
    return images


def angle_values_to_degrees(values: list[float], angle_unit: str) -> list[float]:
    if angle_unit == "deg":
        return values
    if angle_unit == "rad":
        return [math.degrees(v) for v in values]

    max_abs = max((abs(v) for v in values), default=0.0)
    if max_abs <= math.tau + 1e-3:
        return [math.degrees(v) for v in values]
    return values


def pose_from_full_values(values: list[float], frame_id: int | None, angle_unit: str) -> Pose | None:
    if len(values) < 6:
        return None

    x, y, z = values[:3]
    angles = angle_values_to_degrees(values[3:7], angle_unit)
    roll = angles[0]
    pitch = angles[1]
    yaw = angles[2]
    gimbal_pitch = angles[3] if len(angles) >= 4 else pitch

    return Pose(
        frame_id=frame_id,
        x=x,
        y=y,
        z=z,
        roll=roll,
        pitch=pitch,
        yaw=yaw,
        gimbal_pitch=gimbal_pitch,
    )


def pose_from_state_item(item: object, angle_unit: str, state_angle_order: str) -> Pose | None:
    if not isinstance(item, dict) or "state" not in item:
        return None

    state = item["state"]
    if not isinstance(state, list) or len(state) < 2:
        return None

    position = state[0]
    rotation = state[1]
    if not isinstance(position, list) or not isinstance(rotation, list) or len(position) < 3 or len(rotation) < 3:
        return None

    frame_id = item.get("frame", item.get("frame_id", item.get("id")))
    try:
        frame_id = int(frame_id) if frame_id is not None else None
    except (TypeError, ValueError):
        frame_id = None

    rot = angle_values_to_degrees([float(v) for v in rotation[:3]], angle_unit)
    if state_angle_order == "roll-pitch-yaw":
        roll, pitch, yaw = rot
    elif state_angle_order == "yaw-pitch-roll":
        yaw, pitch, roll = rot
    else:
        pitch, yaw, roll = rot

    return Pose(
        frame_id=frame_id,
        x=float(position[0]),
        y=float(position[1]),
        z=float(position[2]),
        roll=roll,
        pitch=pitch,
        yaw=yaw,
        gimbal_pitch=pitch,
    )


def numeric_pose_rows_from_json(data: object, pose_key: str) -> list[object]:
    if isinstance(data, list):
        return data
    if not isinstance(data, dict):
        return []

    if pose_key != "auto":
        rows = data.get(pose_key, [])
        return rows if isinstance(rows, list) else []

    for key in ("raw_logs", "preprocessed_logs", "logs", "poses", "states"):
        rows = data.get(key)
        if isinstance(rows, list):
            return rows
    return []


def read_poses_from_log(log_json: Path, angle_unit: str, pose_key: str, state_angle_order: str) -> list[Pose]:
    with log_json.open("r", encoding="utf-8-sig") as f:
        data = json.load(f)

    poses: list[Pose] = []
    rows = numeric_pose_rows_from_json(data, pose_key)
    for idx, row in enumerate(rows):
        pose = pose_from_state_item(row, angle_unit, state_angle_order)
        if pose is not None:
            poses.append(pose)
            continue

        if isinstance(row, dict):
            values = None
            if all(key in row for key in ("x", "y", "z")):
                values = [
                    row.get("x", 0.0),
                    row.get("y", 0.0),
                    row.get("z", 0.0),
                    row.get("roll", 0.0),
                    row.get("pitch", row.get("gimbal_pitch", 0.0)),
                    row.get("yaw", 0.0),
                    row.get("gimbal_pitch", row.get("pitch", 0.0)),
                ]
                frame_id = row.get("frame", row.get("frame_id", row.get("id")))
                try:
                    frame_id = int(frame_id) if frame_id is not None else None
                except (TypeError, ValueError):
                    frame_id = None
            else:
                frame_id = None
            pose = pose_from_full_values([float(v) for v in values], frame_id, angle_unit) if values else None
        elif isinstance(row, list):
            pose = pose_from_full_values([float(v) for v in row], None, angle_unit)
        elif isinstance(row, str):
            values = [float(v) for v in FLOAT_RE.findall(row)]
            pose = pose_from_full_values(values, None, angle_unit)
        else:
            pose = None

        if pose is not None:
            poses.append(pose)

    if not poses:
        raise ValueError(f"No valid pose rows found in {log_json}")
    return poses


def align_poses(images: list[Path], poses: list[Pose]) -> list[Pose]:
    by_frame = {pose.frame_id: pose for pose in poses if pose.frame_id is not None}
    if by_frame:
        matched = [by_frame.get(int(img.stem)) for img in images]
        if all(pose is not None for pose in matched):
            return [pose for pose in matched if pose is not None]

    if len(poses) >= len(images):
        return poses[: len(images)]

    # If the log is shorter than the image sequence, sample the closest pose.
    if len(poses) == 1:
        return [poses[0] for _ in images]
    idx = np.linspace(0, len(poses) - 1, len(images)).round().astype(int)
    return [poses[i] for i in idx]


def convert_positions(
    poses: list[Pose],
    *,
    relative: bool,
    z_convention: str,
    swap_xy: bool,
    invert_x: bool,
    invert_y: bool,
) -> np.ndarray:
    xyz = np.array([[p.x, p.y, p.z] for p in poses], dtype=float)

    if relative:
        xyz = xyz - xyz[0]

    if swap_xy:
        xyz = xyz[:, [1, 0, 2]]

    if invert_x:
        xyz[:, 0] *= -1
    if invert_y:
        xyz[:, 1] *= -1

    if z_convention == "down":
        xyz[:, 2] *= -1

    return xyz


def camera_direction(
    pose: Pose,
    *,
    z_convention: str,
    swap_xy: bool,
    invert_x: bool,
    invert_y: bool,
    invert_yaw: bool,
    yaw_offset_deg: float,
) -> np.ndarray:
    yaw = pose.yaw + yaw_offset_deg
    if invert_yaw:
        yaw *= -1

    yaw_rad = math.radians(yaw)
    pitch_rad = math.radians(pose.gimbal_pitch)

    direction = np.array(
        [
            math.cos(pitch_rad) * math.cos(yaw_rad),
            math.cos(pitch_rad) * math.sin(yaw_rad),
            math.sin(pitch_rad),
        ],
        dtype=float,
    )

    if swap_xy:
        direction = direction[[1, 0, 2]]
    if invert_x:
        direction[0] *= -1
    if invert_y:
        direction[1] *= -1
    if z_convention == "down":
        direction[2] *= -1

    norm = np.linalg.norm(direction)
    return direction / norm if norm else direction


def axis_limits(points: np.ndarray) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) / 2
    span = float(np.max(maxs - mins))
    if span <= 1e-6:
        span = 1.0
    span *= 1.15
    half = span / 2
    return (
        (center[0] - half, center[0] + half),
        (center[1] - half, center[1] + half),
        (center[2] - half, center[2] + half),
    )


def render_trajectory_panel(
    points: np.ndarray,
    poses: list[Pose],
    frame_idx: int,
    panel_size: int,
    frame_duration: float,
    args: argparse.Namespace,
) -> np.ndarray:
    fig = plt.figure(figsize=(panel_size / 100, panel_size / 100), dpi=100)
    fig.patch.set_facecolor("white")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("white")

    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1.0, 1.0, 1.0, 1.0))
        axis.pane.set_edgecolor((0.72, 0.76, 0.8, 1.0))

    ax.grid(True, color="#d5dbe0", alpha=0.75)
    ax.tick_params(colors="#39434d", labelsize=8)
    ax.xaxis.label.set_color("#26323d")
    ax.yaxis.label.set_color("#26323d")
    ax.zaxis.label.set_color("#26323d")
    ax.set_xlabel("X (m)", labelpad=8)
    ax.set_ylabel("Y (m)", labelpad=8)
    ax.set_zlabel("Z (m)", labelpad=8)

    xlim, ylim, zlim = axis_limits(points)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_zlim(*zlim)
    ax.view_init(elev=args.elev, azim=args.azim)

    ax.plot(points[:, 0], points[:, 1], points[:, 2], color="#c7cdd3", linewidth=3.0, alpha=0.9)
    ax.plot(
        points[: frame_idx + 1, 0],
        points[: frame_idx + 1, 1],
        points[: frame_idx + 1, 2],
        color="#1f77b4",
        linewidth=4.0,
        solid_capstyle="round",
    )
    ax.scatter(*points[0], s=70, color="#2ca02c", edgecolor="#203020", linewidth=0.8, label="start")
    ax.scatter(*points[-1], s=70, color="#d62728", edgecolor="#4d1717", linewidth=0.8, label="end")
    ax.scatter(*points[frame_idx], s=95, color="#ffbf00", edgecolor="#5e4700", linewidth=0.8, label="current")

    span = max(np.ptp(points[:, 0]), np.ptp(points[:, 1]), np.ptp(points[:, 2]), 1.0)
    direction = camera_direction(
        poses[frame_idx],
        z_convention=args.z_convention,
        swap_xy=args.swap_xy,
        invert_x=args.invert_x,
        invert_y=args.invert_y,
        invert_yaw=args.invert_yaw,
        yaw_offset_deg=args.yaw_offset_deg,
    )
    current = points[frame_idx]
    arrow = direction * span * 0.48
    arrow_end = current + arrow
    ax.plot(
        [current[0], arrow_end[0]],
        [current[1], arrow_end[1]],
        [current[2], arrow_end[2]],
        color="#f2a900",
        linewidth=0,
        solid_capstyle="round",
        alpha=0.92,
    )
    ax.quiver(
        current[0],
        current[1],
        current[2],
        arrow[0],
        arrow[1],
        arrow[2],
        color="#f2a900",
        linewidth=0,
        arrow_length_ratio=0.52,
        pivot="tail",
    )
    t = frame_idx * frame_duration
    pose = poses[frame_idx]
    ax.set_title(
        f"3D trajectory | frame {frame_idx:03d} | t={t:.2f}s\n"
        f"yaw={pose.yaw:.1f} deg, gimbal/pitch={pose.gimbal_pitch:.1f} deg",
        color="#111820",
        fontsize=11,
        pad=16,
    )

    ax.legend(loc="upper left", fontsize=8, frameon=False)

    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.9)
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    rgb = cv2.cvtColor(rgba, cv2.COLOR_RGBA2RGB)
    plt.close(fig)
    return rgb


def read_frame_bgr(path: Path) -> np.ndarray:
    # np.fromfile keeps this robust on Windows paths that may contain non-ASCII characters.
    data = np.fromfile(str(path), dtype=np.uint8)
    frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError(f"Could not read image frame: {path}")
    return frame


def render_frames_video(images: list[Path], output: Path, frame_duration: float) -> None:
    first = read_frame_bgr(images[0])
    height, width = first.shape[:2]
    width = width if width % 2 == 0 else width - 1
    height = height if height % 2 == 0 else height - 1
    fps = 1.0 / frame_duration
    output.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open MP4 writer for {output}")

    try:
        for image_path in images:
            frame = read_frame_bgr(image_path)
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            writer.write(frame)
    finally:
        writer.release()

    print(f"Wrote frames video: {output}")


def render_trajectory_video(args: argparse.Namespace, images: list[Path], poses: list[Pose]) -> None:
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    points = convert_positions(
        poses,
        relative=args.relative,
        z_convention=args.z_convention,
        swap_xy=args.swap_xy,
        invert_x=args.invert_x,
        invert_y=args.invert_y,
    )

    fps = 1.0 / args.frame_duration
    panel_size = args.panel_size
    frame_size = (panel_size, panel_size)
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, frame_size)
    if not writer.isOpened():
        raise RuntimeError(f"Could not open MP4 writer for {output}")

    try:
        for idx in range(len(images)):
            trajectory = render_trajectory_panel(points, poses, idx, panel_size, args.frame_duration, args)
            writer.write(cv2.cvtColor(trajectory, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()

    print(f"Wrote trajectory video: {output}")


def render_video(args: argparse.Namespace) -> None:
    sample_dir = Path(args.sample_dir)
    log_json = Path(args.log_json) if args.log_json else sample_dir / "log.json"
    images = collect_images(sample_dir)
    poses = align_poses(
        images,
        read_poses_from_log(
            log_json,
            angle_unit=args.log_angle_unit,
            pose_key=args.log_pose_key,
            state_angle_order=args.state_angle_order,
        ),
    )
    fps = 1.0 / args.frame_duration

    if args.frames_output:
        render_frames_video(images, Path(args.frames_output), args.frame_duration)
    render_trajectory_video(args, images, poses)

    print(f"Frames: {len(images)}")
    print(f"FPS: {fps:.6g}")
    print(f"Duration: {len(images) * args.frame_duration:.3f}s")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-dir", default="sample", help="Folder containing numeric PNG frames.")
    parser.add_argument("--log-json", default=None, help="Pose log JSON. Defaults to <sample-dir>/log.json.")
    parser.add_argument(
        "--log-pose-key",
        default="auto",
        help="Pose list key for object-style logs, e.g. raw_logs or preprocessed_logs. Defaults to auto.",
    )
    parser.add_argument(
        "--log-angle-unit",
        choices=("auto", "deg", "rad"),
        default="auto",
        help="Angle unit in log.json. Auto treats small-magnitude angles as radians.",
    )
    parser.add_argument(
        "--state-angle-order",
        choices=("pitch-yaw-roll", "roll-pitch-yaw", "yaw-pitch-roll"),
        default="pitch-yaw-roll",
        help="Rotation order for log entries shaped like {'state': [[x,y,z], [a,b,c]]}.",
    )
    parser.add_argument("--output", default="static/sample/sample_3d_path.mp4", help="Output MP4 path.")
    parser.add_argument(
        "--frames-output",
        default="static/sample/sample_frames.mp4",
        help="Output MP4 for the raw image frames. Use an empty string to skip it.",
    )
    parser.add_argument("--frame-duration", type=float, default=0.15, help="Seconds per input frame.")
    parser.add_argument("--panel-size", type=int, default=720, help="Output video width and height in pixels.")
    parser.add_argument("--elev", type=float, default=24.0, help="3D plot elevation angle.")
    parser.add_argument("--azim", type=float, default=-58.0, help="3D plot azimuth angle.")
    parser.add_argument(
        "--z-convention",
        choices=("up", "down"),
        default="down",
        help="Use 'down' when raw z increases for downward UAV motion; use 'up' for standard z-up plotting.",
    )
    parser.add_argument(
        "--absolute",
        dest="relative",
        action="store_false",
        help="Plot raw coordinates instead of subtracting the first pose.",
    )
    parser.set_defaults(relative=True)
    parser.add_argument("--swap-xy", action="store_true", help="Swap x and y after loading positions.")
    parser.add_argument("--invert-x", action="store_true", help="Invert plotted x axis.")
    parser.add_argument("--invert-y", action="store_true", help="Invert plotted y axis.")
    parser.add_argument("--invert-yaw", action="store_true", help="Invert yaw before drawing the camera direction.")
    parser.add_argument("--yaw-offset-deg", type=float, default=0.0, help="Yaw offset applied to camera direction.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.frame_duration <= 0:
        raise ValueError("--frame-duration must be positive")
    render_video(args)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import re
from pathlib import Path

import cv2


def natural_key(path: Path) -> list[int | str]:
    """Sort names like step_2.png before step_10.png."""
    parts = re.split(r"(\d+)", path.stem)
    return [int(part) if part.isdigit() else part.lower() for part in parts]


def collect_pngs(source_dir: Path, sample_step: int, sample_offset: int) -> list[Path]:
    png_files = sorted(source_dir.glob("*.png"), key=natural_key)
    if not png_files:
        raise FileNotFoundError(f"No PNG images found in: {source_dir}")
    return png_files[sample_offset::sample_step]


def create_writer(first_image: Path, output_path: Path, interval: float) -> cv2.VideoWriter:
    first_frame = cv2.imread(str(first_image))
    if first_frame is None:
        raise ValueError(f"Failed to read image: {first_image}")

    height, width = first_frame.shape[:2]
    fps = 1.0 / interval
    output_path.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to create video file: {output_path}")
    return writer


def write_images_to_video(image_paths: list[Path], output_path: Path, interval: float) -> None:
    if not image_paths:
        raise ValueError("No images were selected for video generation.")

    first_frame = cv2.imread(str(image_paths[0]))
    if first_frame is None:
        raise ValueError(f"Failed to read image: {image_paths[0]}")

    height, width = first_frame.shape[:2]
    writer = create_writer(image_paths[0], output_path, interval)
    try:
        for image_path in image_paths:
            frame = cv2.imread(str(image_path))
            if frame is None:
                raise ValueError(f"Failed to read image: {image_path}")
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            writer.write(frame)
    finally:
        writer.release()

    print(f"Created video: {output_path}")
    print(f"Images: {len(image_paths)}")


def pngs_to_video(
    source_dir: Path,
    output_path: Path,
    interval: float,
    sample_step: int,
    sample_offset: int,
) -> None:
    image_paths = collect_pngs(source_dir, sample_step, sample_offset)
    write_images_to_video(image_paths, output_path, interval)
    print(f"Source folder: {source_dir}")
    print(f"Sample step: every {sample_step} image(s)")
    print(f"Sample offset: {sample_offset}")


def folder_sequence_to_video(
    parent_dir: Path,
    output_path: Path,
    interval: float,
    sample_step: int,
    sample_offset: int,
    folder_pattern: str,
    segments_dir: Path | None,
) -> None:
    folders = sorted(
        [path for path in parent_dir.glob(folder_pattern) if path.is_dir()],
        key=natural_key,
    )
    if not folders:
        raise FileNotFoundError(f"No folders matching {folder_pattern!r} found in: {parent_dir}")

    all_images: list[Path] = []
    for folder in folders:
        folder_images = collect_pngs(folder, sample_step, sample_offset)
        all_images.extend(folder_images)

        if segments_dir is not None:
            segment_path = segments_dir / f"{folder.name}.mp4"
            write_images_to_video(folder_images, segment_path, interval)

    write_images_to_video(all_images, output_path, interval)
    print(f"Parent folder: {parent_dir}")
    print(f"Folders: {len(folders)}")
    print(f"Sample step: every {sample_step} image(s)")
    print(f"Sample offset: {sample_offset}")
    print(f"Frame interval: {interval}s, FPS: {1.0 / interval:.6g}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate videos from PNG images in natural numeric order."
    )
    parser.add_argument(
        "source_dir",
        nargs="?",
        default=Path("."),
        type=Path,
        help=(
            "PNG folder, or parent folder containing out_0/out_1/... folders. "
            "Default: current folder, with pngsource fallback"
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        default=Path("output.mp4"),
        type=Path,
        help="Output video path. Default: output.mp4",
    )
    parser.add_argument(
        "-i",
        "--interval",
        default=0.15,
        type=float,
        help="Duration of each image in seconds. Default: 0.15",
    )
    parser.add_argument(
        "-s",
        "--sample-step",
        default=3,
        type=int,
        help="Use one image every N images. Default: 3",
    )
    parser.add_argument(
        "--sample-offset",
        default=0,
        type=int,
        help="Start offset for sampling. 0 means 1st image, 2 means 3rd image. Default: 0",
    )
    parser.add_argument(
        "--folder-pattern",
        default="out_*",
        help="Folder pattern for sequence mode. Default: out_*",
    )
    parser.add_argument(
        "--segments-dir",
        default=Path("video_segments"),
        type=Path,
        help=(
            "Directory for each out_x segment video. "
            "Use --no-segments to skip. Default: video_segments"
        ),
    )
    parser.add_argument(
        "--no-segments",
        action="store_true",
        help="Only create the final merged video.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.interval <= 0:
        raise ValueError("Interval must be greater than 0.")
    if args.sample_step <= 0:
        raise ValueError("Sample step must be greater than 0.")
    if args.sample_offset < 0 or args.sample_offset >= args.sample_step:
        raise ValueError("Sample offset must be >= 0 and smaller than sample step.")

    sequence_folders = [path for path in args.source_dir.glob(args.folder_pattern) if path.is_dir()]
    if sequence_folders:
        folder_sequence_to_video(
            args.source_dir,
            args.output,
            args.interval,
            args.sample_step,
            args.sample_offset,
            args.folder_pattern,
            None if args.no_segments else args.segments_dir,
        )
    else:
        if args.source_dir == Path(".") and not list(args.source_dir.glob("*.png")):
            fallback_dir = Path("pngsource")
            if fallback_dir.is_dir():
                args.source_dir = fallback_dir
        pngs_to_video(
            args.source_dir,
            args.output,
            args.interval,
            args.sample_step,
            args.sample_offset,
        )
        print(f"Frame interval: {args.interval}s, FPS: {1.0 / args.interval:.6g}")


if __name__ == "__main__":
    main()

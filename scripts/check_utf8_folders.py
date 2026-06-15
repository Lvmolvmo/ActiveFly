from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path


TEXT_EXTENSIONS = {
    ".csv",
    ".ini",
    ".json",
    ".jsonl",
    ".md",
    ".py",
    ".rst",
    ".toml",
    ".txt",
    ".xml",
    ".yaml",
    ".yml",
}

DEFAULT_SKIP_NAMES = {
    ".DS_Store",
    "Thumbs.db",
    "desktop.ini",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Find folders under real_outdoor_example that contain files whose "
            "contents cannot be decoded as UTF-8."
        )
    )
    parser.add_argument(
        "root",
        nargs="?",
        default="real_outdoor_example",
        type=Path,
        help="Folder to scan. Defaults to real_outdoor_example.",
    )
    parser.add_argument(
        "--all-files",
        action="store_true",
        help="Check every file instead of only common text file extensions.",
    )
    parser.add_argument(
        "--include-hidden",
        action="store_true",
        help="Include hidden/system metadata files such as .DS_Store.",
    )
    parser.add_argument(
        "--convert-gbk",
        action="store_true",
        help="Convert non-UTF-8 text files that decode as GBK to UTF-8.",
    )
    return parser.parse_args()


def should_check_file(path: Path, all_files: bool, include_hidden: bool) -> bool:
    if not include_hidden and path.name in DEFAULT_SKIP_NAMES:
        return False
    return all_files or path.suffix.lower() in TEXT_EXTENSIONS


def utf8_error(path: Path) -> str | None:
    try:
        path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        return f"byte {exc.start}: {exc.reason}"
    except OSError as exc:
        return f"read failed: {exc}"
    return None


def convert_gbk_to_utf8(path: Path) -> str | None:
    try:
        text = path.read_bytes().decode("gbk")
        path.write_text(text, encoding="utf-8")
    except (UnicodeDecodeError, OSError) as exc:
        return str(exc)
    return None


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    if not root.is_dir():
        print(f"Folder not found: {root}")
        return 2

    bad_by_folder: dict[Path, list[tuple[Path, str]]] = defaultdict(list)
    converted: list[Path] = []
    checked = 0

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if not should_check_file(path, args.all_files, args.include_hidden):
            continue

        checked += 1
        error = utf8_error(path)
        if error:
            if args.convert_gbk:
                conversion_error = convert_gbk_to_utf8(path)
                if conversion_error is None:
                    converted.append(path)
                    continue
                error = f"{error}; GBK conversion failed: {conversion_error}"
            bad_by_folder[path.parent].append((path, error))

    if converted:
        print(f"Converted {len(converted)} file(s) from GBK to UTF-8:")
        for path in converted:
            print(f"  - {path.relative_to(root)}")
        print()

    if not bad_by_folder:
        print(f"OK: checked {checked} file(s). No non-UTF-8 files found.")
        return 0

    print(f"Found {len(bad_by_folder)} folder(s) containing non-UTF-8 files:")
    for folder in sorted(bad_by_folder):
        print(f"\n{folder.relative_to(root)}")
        for file_path, error in bad_by_folder[folder]:
            print(f"  - {file_path.name}: {error}")

    return 1


if __name__ == "__main__":
    raise SystemExit(main())

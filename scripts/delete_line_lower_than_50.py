import os
from pathlib import Path

base_path = Path(__file__).resolve().parent
log_path = base_path / ".." / "output" / "map_results" / "log"
delete_rows_const = 250


if __name__ == "__main__":
    if not log_path.exists():
        print(f"[INFO] Log directory not found: {log_path}")
        raise SystemExit(0)

    files_to_delete = []

    # Recursively walk through all files under log_path
    for file_path in log_path.rglob("*"):
        if not file_path.is_file():
            continue

        try:
            # Count lines robustly, ignoring encoding errors
            with file_path.open("r", encoding="utf-8", errors="ignore") as f:
                line_count = sum(1 for _ in f)
        except OSError as e:
            print(f"[WARN] Cannot read file {file_path}: {e}")
            continue

        if line_count < delete_rows_const:
            files_to_delete.append((file_path, line_count))

    if not files_to_delete:
        print(
            f"[INFO] No files with line count < {delete_rows_const} "
            f"found under {log_path}"
        )
        raise SystemExit(0)

    print(f"[INFO] Files to delete (line count < {delete_rows_const}):")
    for path, count in files_to_delete:
        print(f"  {path}  ->  {count} lines")

    # Delete the files
    for path, _ in files_to_delete:
        try:
            path.unlink()
            print(f"[OK] Deleted {path}")
        except OSError as e:
            print(f"[ERROR] Failed to delete {path}: {e}")

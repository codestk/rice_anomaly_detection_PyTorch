"""
Utility for keeping two folders in sync by deleting entries that only exist
in the second folder.

The script treats the first folder as the source of truth. Anything that is
present in the second folder but missing from the first folder will be removed
so that both folders end up containing the same top-level names.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox


def remove_path(path: Path) -> None:
    """Delete a file or directory tree."""
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


def sync_folders(source: Path, secondary: Path, dry_run: bool = False) -> list[str]:
    """
    Delete items from ``secondary`` whose names are not present in ``source``.

    Returns a list with the names that were (or would be) deleted.
    """
    if not source.is_dir():
        raise ValueError(f"Source folder does not exist: {source}")
    if not secondary.is_dir():
        raise ValueError(f"Secondary folder does not exist: {secondary}")

    source_names = {entry.name for entry in source.iterdir()}
    deleted: list[str] = []

    for entry in secondary.iterdir():
        if entry.name not in source_names:
            deleted.append(entry.name)
            if not dry_run:
                remove_path(entry)

    return deleted


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Browse two folders and delete anything that only exists in the second "
            "folder so that both folders share the same contents."
        )
    )
    parser.add_argument(
        "source",
        type=Path,
        help="Reference folder (kept untouched).",
    )
    parser.add_argument(
        "secondary",
        type=Path,
        help="Folder that will have extra files removed.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deleted without removing anything.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    deleted = sync_folders(args.source, args.secondary, args.dry_run)

    if not deleted:
        print("Folders already contain the same top-level entries.")
        return

    action = "would be removed" if args.dry_run else "removed"
    print(f"The following entries {action}:")
    for name in deleted:
        print(f" - {name}")


def launch_ui() -> None:
    root = tk.Tk()
    root.title("Folder Sync Delete Tool")
    root.resizable(False, False)

    source_var = tk.StringVar()
    secondary_var = tk.StringVar()
    result_var = tk.StringVar()
    deleted_list = tk.StringVar()

    def browse_folder(target_var: tk.StringVar) -> None:
        path = filedialog.askdirectory(title="เลือกโฟลเดอร์")
        if path:
            target_var.set(path)

    def update_results(names: list[str]) -> None:
        if not names:
            result_var.set("ทั้งสองโฟลเดอร์มีไฟล์ตรงกันอยู่แล้ว")
            deleted_list.set("")
            return

        result_var.set(f"จะลบทั้งหมด {len(names)} รายการ:")
        deleted_list.set("\n".join(names))

    def handle_delete() -> None:
        try:
            source = Path(source_var.get()).expanduser()
            secondary = Path(secondary_var.get()).expanduser()
            preview = sync_folders(source, secondary, dry_run=True)
        except ValueError as exc:
            messagebox.showerror("Error", str(exc))
            return
        except Exception as exc:  # pragma: no cover - safety net for UI
            messagebox.showerror("Error", f"Unexpected error: {exc}")
            return

        update_results(preview)

        if not preview:
            messagebox.showinfo("Info", "ไม่มีไฟล์ที่ต้องลบ")
            return

        if not messagebox.askyesno(
            "Confirm",
            f"ยืนยันการลบ {len(preview)} รายการจากโฟลเดอร์ที่สองหรือไม่?",
        ):
            return

        try:
            deleted = sync_folders(source, secondary, dry_run=False)
        except Exception as exc:  # pragma: no cover
            messagebox.showerror("Error", f"ลบไม่สำเร็จ: {exc}")
            return

        update_results(deleted)
        messagebox.showinfo("Success", "ลบไฟล์เรียบร้อยแล้ว")

    padding = {"padx": 8, "pady": 4}

    tk.Label(root, text="โฟลเดอร์หลัก").grid(row=0, column=0, sticky="w", **padding)
    tk.Entry(root, textvariable=source_var, width=50).grid(
        row=0, column=1, **padding
    )
    tk.Button(root, text="Browse", command=lambda: browse_folder(source_var)).grid(
        row=0, column=2, **padding
    )

    tk.Label(root, text="โฟลเดอร์เป้าหมาย").grid(row=1, column=0, sticky="w", **padding)
    tk.Entry(root, textvariable=secondary_var, width=50).grid(
        row=1, column=1, **padding
    )
    tk.Button(root, text="Browse", command=lambda: browse_folder(secondary_var)).grid(
        row=1, column=2, **padding
    )

    tk.Button(root, text="Delete", command=handle_delete, width=20).grid(
        row=2, column=0, columnspan=3, **padding
    )

    tk.Label(root, textvariable=result_var, fg="blue", wraplength=400, justify="left").grid(
        row=3, column=0, columnspan=3, sticky="w", **padding
    )

    tk.Label(root, textvariable=deleted_list, justify="left", anchor="w").grid(
        row=4, column=0, columnspan=3, sticky="w", **padding
    )

    root.mainloop()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main()
    else:
        launch_ui()

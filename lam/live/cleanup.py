import json
import shutil
import uuid
from pathlib import Path
from typing import Iterable

from lam.live.settings import settings

def _resolve(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()

def _allowed_roots() -> list[Path]:
    return [
        _resolve(settings.output_root),
        _resolve(settings.oac_output_root),
        _resolve(settings.upload_dir),
        _resolve(settings.tracking_output_dir),
    ]

def _is_allowed(path: Path) -> bool:
    path = path.resolve()
    for root in _allowed_roots():
        try:
            path.relative_to(root)
            return True
        except ValueError:
            pass

    return False

def safe_remove_path(path: str | Path) -> bool:
    path = _resolve(path)
    if not _is_allowed(path):
        print("[CLEANUP] refused unsafe path:", path)

    if not path.exists():
        return False

    try:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()

        print("[CLEANUP] removed path:", path)
        return True
    except Exception as e:
        print("[CLEANUP] failed to remove path:", path, "error:", e)
        return False

def cleanup_paths(paths: Iterable[str | Path]) -> dict:
    removed = []
    failed = []

    for path in paths:
        try:
            ok = safe_remove_path(path)
            if ok:
                removed.append(str(path))
        except Exception as e:
            failed.append({"path": str(path), "error": str(e)})

    return {"removed": removed, "failed": failed}

def cleanup_registry_dir() -> Path:
    path = _resolve(settings.output_root) / "live_cleanup"
    path.mkdir(parents=True, exist_ok=True)
    return path

def register_cleanup(paths: Iterable[str | Path], kind: str = "temporary_export") -> str:
    cleanup_id = uuid.uuid4().hex

    manifest = {
        "cleanup_id": cleanup_id,
        "kind": kind,
        "paths": [str(_resolve(p)) for p in paths],
    }

    manifest_path = cleanup_registry_dir() / f"{cleanup_id}.json"

    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("[CLEANUP] registered:", cleanup_id, manifest["paths"])

    return cleanup_id


def cleanup_by_id(cleanup_id: str) -> dict:
    cleanup_id = "".join(c for c in cleanup_id if c.isalnum())

    if not cleanup_id:
        return {
            "ok": False,
            "error": "empty cleanup_id",
            "removed": [],
            "failed": [],
        }

    manifest_path = cleanup_registry_dir() / f"{cleanup_id}.json"

    if not manifest_path.exists():
        return {
            "ok": False,
            "error": "cleanup manifest not found",
            "cleanup_id": cleanup_id,
            "removed": [],
            "failed": [],
        }

    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    paths = manifest.get("paths", [])

    result = cleanup_paths(paths)

    try:
        manifest_path.unlink()
    except Exception:
        pass

    return {
        "ok": True,
        "cleanup_id": cleanup_id,
        **result,
    }


def processed_export_dir_from_image(processed_image_path: str | Path) -> Path:
    """
    processed image:
      tracking_output_live/export/<id>/images/00000_00.png

    export dir:
      tracking_output_live/export/<id>
    """
    return _resolve(processed_image_path).parents[1]


def oac_avatar_dir_from_zip(zip_path: str | Path) -> Path:
    zip_path = _resolve(zip_path)
    return zip_path.with_suffix("")
        
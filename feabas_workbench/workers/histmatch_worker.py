"""Histogram-match all tiles of a source folder to a template (mirrors folder structure)."""

from __future__ import annotations

from pathlib import Path

from feabas_workbench.core import histmatch
from feabas_workbench.core.tiles import list_image_files
from feabas_workbench.workers.common import load_spec, progress, result, log, run


def main() -> int:
    spec = load_spec("histogram matching")
    in_root = Path(spec["in_root"])
    out_root = Path(spec["out_root"])
    template = Path(spec["template"])
    ext = spec.get("ext", "tif")
    files = [Path(p) for p in spec.get("files", [])] or list_image_files(in_root, ext, spec.get("recursive", True))
    log(f"{len(files)} files, template {template.name}")
    failures = histmatch.match_folder(
        files, in_root, out_root, template,
        ignore_black=bool(spec.get("ignore_black", True)),
        ignore_white=bool(spec.get("ignore_white", False)),
        workers=int(spec.get("workers", 8)),
        progress=progress,
        skip_existing=bool(spec.get("skip_existing", True)),
    )
    for p, err in failures[:20]:
        log(f"FAILED {p}: {err}")
    result({"n_files": len(files), "n_failed": len(failures), "out_root": str(out_root)})
    return 1 if failures else 0


if __name__ == "__main__":
    run(main)

"""
End-to-end check: build a small synthetic dataset and run the whole FEABAS pipeline on it.

    python tools/run_demo_pipeline.py D:/demo --feabas-python C:/envs/fw-feabas/python.exe
    python tools/run_demo_pipeline.py D:/demo --render          # also render the aligned PNG stack
    python tools/run_demo_pipeline.py D:/stack --render --masks --rows 1 --cols 1 --tile 1024
                                        # an image stack (one image per section) with segmentation masks

Run it with any interpreter that has the workbench installed; --feabas-python is the one with
FEABAS (default: the same interpreter, which is how CI runs it). Exit code 0 only if every step
ran and produced what it should - output files for every section and pair, and matches in them:
every montage connected, coarse and fine matches with several points per section pair. With
--masks the input images themselves and 16-bit block labels are then carried through the
alignment as segmentation masks and checked against the aligned images: same tile layout at
every mip level, only input labels, no shift. A project that already exists is reused
(finished steps are skipped), so a failed run can be repeated after a fix.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from feabas_workbench.core.pipeline import (mask_problems, match_problems, render_masks,       # noqa: E402
                                            run_standard_pipeline, summary)
from feabas_workbench.core.project import Project                              # noqa: E402
from feabas_workbench.core.synthetic import make_demo_project                  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("project", help="project folder (created with a synthetic dataset if it does not exist)")
    ap.add_argument("--feabas-python", default=sys.executable, help="interpreter with FEABAS installed")
    ap.add_argument("--render", action="store_true", help="also render the aligned stack as PNG tiles + mipmaps")
    ap.add_argument("--masks", action="store_true", help="with --render: carry segmentation masks through the "
                    "alignment and check them against the aligned images")
    ap.add_argument("--sections", type=int, default=4)
    ap.add_argument("--rows", type=int, default=2, help="tiles per section vertically (1 x 1: an image stack)")
    ap.add_argument("--cols", type=int, default=2)
    ap.add_argument("--tile", type=int, default=384)
    ap.add_argument("--timeout", type=float, default=1800, help="seconds allowed per step")
    a = ap.parse_args(argv)
    root = Path(a.project)
    if Project.exists(root):
        print(f"reusing project {root}")
    else:
        p = make_demo_project(root, n_sections=a.sections, tile=a.tile, rows=a.rows, cols=a.cols)
        v = p.state.volume
        print(f"created {root}: {v.n_sections} sections, {v.grid_rows}x{v.grid_cols} tiles of {v.tile_w} px")
    t0 = time.time()
    logfile = open(root / "workbench.log", "a", encoding="utf-8")

    def log(text: str) -> None:
        print(text, flush=True)
        logfile.write(text + "\n")

    runs = run_standard_pipeline(root, a.feabas_python, render=a.render, log=log, timeout_per_step=a.timeout)
    logfile.close()
    print()
    print(summary(runs))
    ok = bool(runs) and all(r.ok for r in runs) and len(runs) >= 10
    problems = match_problems(root)
    if ok and a.masks:
        problems += check_masks(root, a.feabas_python, a.timeout) if a.render else ["--masks needs --render"]
    for line in problems:
        print("!! " + line)
    ok = ok and not problems
    print(f"\n{'PIPELINE OK' if ok else 'PIPELINE FAILED'} in {(time.time() - t0) / 60:.1f} min")
    return 0 if ok else 1


def check_masks(root: Path, python: str, timeout: float) -> list[str]:
    """Masks made from the input images (8-bit, matched by name) and 16-bit block labels (matched in
    order when every section is one image), carried through the alignment and checked."""
    import cv2
    import numpy as np
    from feabas_workbench.core.configs import ConfigStore
    from feabas_workbench.core.project import Project
    from feabas_workbench.core.segmentation import read_stitch_coord
    from feabas_workbench.core.steps import aligned_dir
    project = Project.load(root)
    images_dir, labels_dir = root / "demo_masks" / "images", root / "demo_masks" / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    tiles = [(s, t) for s in project.section_names() for t in read_stitch_coord(project.stitch_coord_dir / f"{s}.txt")[0]]
    single = len(tiles) == len(project.section_names())
    ids = set()
    for k, (s, t) in enumerate(tiles):
        a = cv2.imread(str(t), cv2.IMREAD_UNCHANGED)
        cv2.imwrite(str(images_dir / f"{t.stem}.png"), a)
        yy, xx = np.mgrid[0:a.shape[0], 0:a.shape[1]]
        lab = (40000 + 97 * k + (yy // 48) * 7 + xx // 48).astype(np.uint16)
        lab[a == 0] = 0                                   # labels only where the image has data
        ids |= set(np.unique(lab).tolist())
        cv2.imwrite(str(labels_dir / (f"seg_{k:03d}.png" if single else f"{t.stem}.png")), lab)
    base = aligned_dir(root, ConfigStore(root / "configs"))
    problems = []
    for name, folder, kw in (("images_as_masks", images_dir, dict(images_as_masks=True)),
                             ("labels16", labels_dir, dict(labels=ids))):
        code, out = render_masks(root, folder, name, python, timeout=timeout)
        if code != 0:
            problems.append(f"aligned masks '{name}': exit {code}")
            continue
        problems += [f"aligned masks '{name}': {line}" for line in mask_problems(base, out, **kw)]
    if not problems:
        print(f"aligned masks OK: images and 16-bit labels, {len(tiles)} mask files"
              + (", labels matched in section order" if single else ""))
    return problems


if __name__ == "__main__":
    raise SystemExit(main())

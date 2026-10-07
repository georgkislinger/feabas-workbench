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
every mip level, only input labels, no shift. Both kinds of masks also go through once more
exported at --mask-mip (images: block means; labels: majority), which must give a stack that
starts at that level, lines up with the images there and agrees with the full-resolution labels'
own mipmap. --render-mip renders the aligned images at a coarser level (rendering.mip_level) when
the project is created. A project that already exists is reused (finished steps are skipped), so a
failed run can be repeated after a fix.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from feabas_workbench.core.pipeline import (label_agreement, label_shift, mask_problems,       # noqa: E402
                                            match_problems, render_masks, run_standard_pipeline, summary)
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
    ap.add_argument("--render-mip", type=int, default=0, help="render the aligned images at this mip level "
                    "(rendering.mip_level of a new project)")
    ap.add_argument("--mask-mip", type=int, default=2, help="with --masks: also carry masks exported at this mip level")
    ap.add_argument("--timeout", type=float, default=1800, help="seconds allowed per step")
    a = ap.parse_args(argv)
    root = Path(a.project)
    if Project.exists(root):
        print(f"reusing project {root}")
    else:
        p = make_demo_project(root, n_sections=a.sections, tile=a.tile, rows=a.rows, cols=a.cols)
        v = p.state.volume
        print(f"created {root}: {v.n_sections} sections, {v.grid_rows}x{v.grid_cols} tiles of {v.tile_w} px")
        if a.render_mip:
            from feabas_workbench.core.configs import ConfigStore
            cs = ConfigStore(root / "configs")
            cs.set("alignment", "rendering.mip_level", a.render_mip)
            cs.save("alignment")
            print(f"aligned images rendered at mip {a.render_mip}")
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
        problems += check_masks(root, a.feabas_python, a.timeout, a.mask_mip) if a.render else ["--masks needs --render"]
    for line in problems:
        print("!! " + line)
    ok = ok and not problems
    print(f"\n{'PIPELINE OK' if ok else 'PIPELINE FAILED'} in {(time.time() - t0) / 60:.1f} min")
    return 0 if ok else 1


def check_masks(root: Path, python: str, timeout: float, mask_mip: int = 2) -> list[str]:
    """Masks made from the input images (8-bit, matched by name) and 16-bit block labels (matched in
    order when every section is one image), carried through the alignment and checked - as they are,
    and once more exported at *mask_mip* (images: block means, labels: majority)."""
    import cv2
    import numpy as np
    from feabas_workbench.core.configs import ConfigStore
    from feabas_workbench.core.project import Project
    from feabas_workbench.core.segmentation import first_level, read_stitch_coord, reduce_labels, stack_first_mip
    from feabas_workbench.core.steps import aligned_dir, aligned_render_mip
    project = Project.load(root)
    configs = ConfigStore(root / "configs")
    render_mip = aligned_render_mip(configs)
    folders = {key: root / "demo_masks" / key for key in ("images", "labels", "images_mip", "labels_mip")}
    for folder in folders.values():
        folder.mkdir(parents=True, exist_ok=True)
    tiles = [(s, t) for s in project.section_names() for t in read_stitch_coord(project.stitch_coord_dir / f"{s}.txt")[0]]
    single = len(tiles) == len(project.section_names())
    ids = set()
    f = 2 ** mask_mip
    for k, (s, t) in enumerate(tiles):
        a = cv2.imread(str(t), cv2.IMREAD_UNCHANGED)
        cv2.imwrite(str(folders["images"] / f"{t.stem}.png"), a)
        small = cv2.resize(a, (-(-a.shape[1] // f), -(-a.shape[0] // f)), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(folders["images_mip"] / f"{t.stem}.png"), small)
        yy, xx = np.mgrid[0:a.shape[0], 0:a.shape[1]]
        # 48 px blocks, which a mip export up to mip 4 keeps exactly, with shuffled labels: a tie in
        # the majority mipmaps (the smaller label wins) then goes either way, as with real
        # segmentations, instead of moving every edge of a section in one direction
        blocks = (yy // 48) * 64 + xx // 48
        shuffled = np.random.default_rng(k).permutation(int(blocks.max()) + 1)
        lab = (1 + shuffled[blocks]).astype(np.uint16)
        lab[a == 0] = 0                                   # labels only where the image has data
        ids |= set(np.unique(lab).tolist())
        name = f"seg_{k:03d}.png" if single else f"{t.stem}.png"
        cv2.imwrite(str(folders["labels"] / name), lab)
        cv2.imwrite(str(folders["labels_mip"] / name), reduce_labels(lab, mask_mip))
    base = aligned_dir(root, configs)
    problems, out = [], {}
    cases = (("images_as_masks", "images", 0, dict(images_as_masks=True)),
             ("labels16", "labels", 0, dict(labels=ids)),
             # grey images pass through majority mipmaps here, which is no precise measure: a sanity check
             (f"images_as_masks_mip{mask_mip}", "images_mip", mask_mip, dict(images_as_masks=True)),
             (f"labels16_mip{mask_mip}", "labels_mip", mask_mip, dict(labels=ids)))
    for name, key, level, kw in cases:
        code, out[key] = render_masks(root, folders[key], name, python, timeout=timeout)
        if key == "labels_mip" and code == 0:       # once more, rendered at full resolution
            code, out["labels_mip_full"] = render_masks(root, folders[key], name + "_rendered_at_mip0", python,
                                                        timeout=timeout, supersample=mask_mip)
        if code != 0:
            problems.append(f"aligned masks '{name}': exit {code}")
            continue
        problems += [f"aligned masks '{name}': {line}" for line in mask_problems(base, out[key], **kw)]
        want, got = first_level(level, render_mip), stack_first_mip(out[key])
        if got != want:
            problems.append(f"aligned masks '{name}': start at mip {got}, not at mip {want}")
    level = first_level(mask_mip, render_mip)
    if not problems:
        # the labels exported at mip k, rendered at full resolution, must give exactly what the
        # full-resolution labels give (they hold the same blocks): every coordinate convention agrees
        exact, _ = label_agreement(out["labels"], out["labels_mip_full"], level)
        ex, ey = label_shift(out["labels"], out["labels_mip_full"], level)
        if exact < 0.999 or max(abs(ex), abs(ey)) > 0.02:
            problems.append(f"16-bit labels exported at mip {mask_mip} and rendered at full resolution differ from the "
                            f"full-resolution ones at mip {level}: {exact:.2%} agree, shift ({ex:+.3f}, {ey:+.3f}) px")
        # rendered as by default (two levels finer than the masks): within rounding at that level
        agree, best = label_agreement(out["labels"], out["labels_mip"], level)
        dx, dy = label_shift(out["labels"], out["labels_mip"], level)
        if best != (0, 0) or agree < 0.85 or max(abs(dx), abs(dy)) > 0.35:
            problems.append(f"16-bit labels exported at mip {mask_mip} against the full-resolution ones at mip {level}: "
                            f"{agree:.1%} of the labelled pixels agree, best at a shift of {best}, sub-pixel shift "
                            f"({dx:+.2f}, {dy:+.2f}) px")
    if not problems:
        print(f"aligned masks OK: images and 16-bit labels, {len(tiles)} mask files"
              + (", labels matched in section order" if single else "")
              + f"; exported at mip {mask_mip}: from mip {level}; rendered at full resolution {exact:.2%} identical to "
                f"the full-resolution labels, as by default {agree:.1%} with a shift of ({dx:+.2f}, {dy:+.2f}) px")
    return problems


if __name__ == "__main__":
    raise SystemExit(main())

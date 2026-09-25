"""Batch mask operations using the same implementation as the interactive editor."""
from pathlib import Path
from feabas_workbench.core.project import Project, VolumeInfo
from feabas_workbench.core.maskstore import MaskStore
from feabas_workbench.core.masks import TissueParams, dark_regions, write_mask
from feabas_workbench.workers.common import load_spec, progress, result, run


def main():
    spec = load_spec("cluster mask operation")
    project = Project(Path(spec["root"]))
    project.state.volume = VolumeInfo(**spec["volume"])
    store = MaskStore(project)
    sections = spec["sections"]
    params = TissueParams.from_dict(spec["tissue"])
    results = []
    for i, section in enumerate(sections, 1):
        if not store.thumbnail_path(section):
            raise RuntimeError(f"Missing thumbnail: {section}")
        if spec["operation"] == "tissue":
            store.compute_tissue(section, params, save=True)
        elif spec["operation"] == "dark":
            mask = dark_regions(store.thumbnail(section), spec["max_grey"], spec["min_px"], spec["dilate"])
            write_mask(store.folds_dir / f"{section}.png", mask.astype("uint8") * 255)
        elif spec["operation"] == "compose":
            if spec["skip_edited"] and store.is_hand_edited(section):
                continue
            if store.tissue_mask(section) is None:
                store.compute_tissue(section, params, save=True)
            results.append(store.compose(section, **spec["compose"]))
        else:
            raise ValueError("Unknown mask operation.")
        progress(i, len(sections), section)
    result({"sections": len(sections), "masks": results})
    return 0


if __name__ == "__main__":
    run(main)

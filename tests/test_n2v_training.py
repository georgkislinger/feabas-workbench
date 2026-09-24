import json

import numpy as np
import pytest
import tifffile

from feabas_workbench.core.n2v_training import Tile, describe, inspect_selection, prepare_data, sample_regions


def test_minimum_counts_pixels_not_files():
    assert describe([Tile('large.tif', 2560, 2560, 'uint8')])["valid"]
    assert not describe([Tile(str(i), 128, 128, 'uint8') for i in range(24)])["valid"]
    assert not describe([Tile('long.tif', 32, 1000000, 'uint8')])["valid"]
    assert not describe([Tile('large.tif', 2560, 2560, 'uint8')], patch=1024)["valid"]


def test_spatial_splits_are_disjoint_unique_and_reproducible():
    tiles = [Tile('a', 2560, 2560, 'uint8'), Tile('b', 128, 8192, 'uint8')]
    records = sample_regions(tiles, 128, (300, 70))
    assert records == sample_regions(tiles, 128, (300, 70))
    cells = [(r['tile'], r['y'], r['x']) for r in records]
    assert len(cells) == len(set(cells))
    for r in records:
        tile = tiles[r['tile']]
        assert r['y'] + 128 <= tile.height and r['x'] + 128 <= tile.width


def test_headers_deduplicate_reject_stacks_and_auto_select_pixels(tmp_path):
    a = tmp_path / 'a.tif'; tifffile.imwrite(a, np.zeros((2560, 2560), np.uint8))
    info = inspect_selection([a, a], auto=True)
    assert info['valid'] and len(info['tiles']) == 1
    b = tmp_path / 'stack.tif'; tifffile.imwrite(b, np.zeros((4, 100, 100), np.uint8), photometric='minisblack')
    with pytest.raises(ValueError, match='single-plane'):
        inspect_selection([b])
    with pytest.raises(FileNotFoundError):
        inspect_selection([tmp_path / 'absent.tif'])


def test_sample_data_matches_manifest_and_does_not_change_source(tmp_path):
    path = tmp_path / 'image.tif'
    source = np.broadcast_to(np.arange(2560, dtype=np.uint16)[:, None], (2560, 2560))
    tifffile.imwrite(path, source)
    train, val, manifest = prepare_data([path], 128, 12, log=lambda _: None)
    assert train.dtype == val.dtype == np.float32
    for r in manifest['patches']:
        sample = (train if r['split'] == 'train' else val)[r['sample']]
        np.testing.assert_array_equal(sample, source[r['y']:r['y'] + 128, r['x']:r['x'] + 128])
    np.testing.assert_array_equal(tifffile.imread(path), source)


def test_gui_plots_both_losses_and_loads_best_preview(tmp_path):
    from PySide6.QtWidgets import QApplication
    from feabas_workbench.core.envs import Settings
    from feabas_workbench.core.synthetic import make_demo_project
    from feabas_workbench.core.configs import ConfigStore
    from feabas_workbench.ui.bridge import AppContext
    from feabas_workbench.ui.pages.page_preprocess import PreprocessPage
    app = QApplication.instance() or QApplication([])
    project = make_demo_project(tmp_path / 'project', n_sections=2, tile=128)
    ctx = AppContext(Settings()); ctx.project = ctx.local_project = project
    ctx.configs = ConfigStore(project.configs_dir)
    page = PreprocessPage(ctx)
    work = project.models_dir / 'n2v/run'; work.mkdir(parents=True)
    history = [dict(epoch=1, train_loss=.8, val_loss=1.), dict(epoch=2, train_loss=.4, val_loss=.6)]
    (work / 'training_status.json').write_text(json.dumps(dict(state='stopped', history=history,
        best_epoch=2, best_loss=.6, preview='training_preview.npz')))
    np.savez(work / 'training_preview.npz', original=np.arange(1024).reshape(32,32),
             denoised=np.arange(1024).reshape(32,32) + 2, epoch=2)
    (work / 'best.ckpt').touch(); (work / 'zz-other.ckpt').touch()
    page._dn_training_work = work
    page._dn_poll_training()
    assert page.dn_loss.history == history
    assert page._dn_preview_epoch == 2 and not page.dn_stop.isEnabled()
    page._refresh_models()
    assert page.dn_models.currentData() == str(work / 'best.ckpt')
    page.dn_list.addItem('placeholder'); page._save_dn(); page._dn_clear()
    assert project.state.preprocessing.denoise['training_tiles'] == []
    page._dn_check_timer.stop()
    page.dn_live.show(); page.dn_live.resize(740, 580)
    assert page.dn_live.grab().save(str(tmp_path / 'live-preview.png'))
    page.shutdown(); page.close(); app.processEvents()

import json
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

from feabas_workbench.core.local_parallel import (input_sections, mipmap_plan, mipmap_worker_gib, selected_indices,
                                                  plan, options)
from feabas_workbench.core.project import Project
from feabas_workbench.core.synthetic import make_demo_project
from feabas_workbench.core.configs import ConfigStore


# GUI-only installations deliberately do not include the separate FEABAS runtime.
# CI also runs this file in the FEABAS end-to-end job, where these checks are mandatory.
requires_feabas = pytest.mark.skipif(importlib.util.find_spec("feabas") is None,
                                    reason="requires the separate FEABAS runtime")


@pytest.fixture
def project(tmp_path):
    return make_demo_project(tmp_path / 'project', n_sections=3, tile=128)


def test_cpu_and_ram_limits_and_persistence(project):
    project.state.local_execution = dict(ram_budget_gib=32, steps={
        'stitch.matching': dict(mode='both', sections=10, measured_gib=10)})
    project.save()
    loaded = Project.load(project.root)
    setting = dict(mode='both', workers=8, sections=10, measured_gib=10)
    p = plan(loaded, 'stitch.matching', setting, available=(64, 512), count=100)
    assert (p.sections, p.workers) == (3, 8)
    assert options(loaded, 'stitch.matching')['mode'] == 'both'
    assert plan(loaded, 'stitch.matching', setting, available=(4, 512)).workers == 4
    assert plan(loaded, 'stitch.matching', setting, available=(64, 10)).sections == 1


def test_a_set_cpu_budget_may_use_hyper_threads(project):
    """32 cores / 64 logical CPUs, 4 sections x 12 workers asked for: the physical cores allow 2 at
    once and the note says so; a budget of 64 that was set must not be cut back to 32."""
    s = dict(mode='both', workers=12, sections=4, measured_gib=1)
    machine = dict(available=(32, 455), threads=64, count=100)
    auto = plan(project, 'stitch.rendering', s, **machine)
    assert (auto.cpu_budget, auto.sections, auto.workers) == (32, 2, 12)
    assert auto.note.startswith("2 of 4 sections at once: CPU budget 32 physical cores") and "64 logical CPUs" in auto.note
    project.state.local_execution["cpu_budget"] = 64
    full = plan(project, 'stitch.rendering', s, **machine)
    assert (full.cpu_budget, full.sections, full.workers) == (64, 4, 12) and full.note == ""
    assert plan(project, 'stitch.rendering', s, cpu_override=128, **machine).cpu_budget == 64   # no more than there is
    assert plan(project, 'stitch.rendering', s, cpu_override=0, **machine).cpu_budget == 32     # 'All physical cores'
    ram_bound = plan(project, 'stitch.rendering', dict(s, measured_gib=200), cpu_override=64, **machine)
    assert ram_bound.sections == 1 and "RAM budget 364 GiB" in ram_bound.note


def test_unknown_dimensions_never_multiply_unknown_ram(project):
    project.state.volume.tile_w = 0
    s = dict(mode='across', workers=12, sections=10, measured_gib=0)
    assert plan(project, 'stitch.matching', s, available=(64, 512)).sections == 1
    s['measured_gib'] = 2
    assert plan(project, 'stitch.matching', s, available=(64, 512)).sections == 3


def test_optimization_does_not_multiply_useless_inner_workers(project):
    p = plan(project, 'stitch.optimization', dict(mode='both', workers=16, sections=2, measured_gib=1), available=(4, 20))
    assert (p.sections, p.workers) == (2, 1)


def test_ranges_match_actual_partial_inputs_and_filtering(project):
    folder = project.root / 'stitch/match_h5'; folder.mkdir(exist_ok=True, parents=True)
    for name in ['s0003.h5', 's0001.h5']:
        (folder / name).touch()
    paths = input_sections(project.root, 'stitch.optimization')
    assert [p.stem for p in paths] == ['s0001', 's0003']
    filtered, indices = selected_indices(paths, 'stitch.optimization', start=0, stop=0, filt='0003')
    assert indices == [0] and filtered[0].stem == 's0003'
    assert selected_indices(paths, 'align.rendering', stop=0)[1] == []
    assert selected_indices(list(range(8)), 'align.downsample', 1, 7, 2)[1] == [1, 3, 5]
    with pytest.raises(ValueError):
        selected_indices(paths, 'stitch.matching', stride=0)


def test_gui_routes_only_opted_in_local_steps(project, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QApplication
    from feabas_workbench.core.envs import Settings
    from feabas_workbench.ui.bridge import AppContext
    from feabas_workbench.ui.local_parallel_dialog import LocalParallelDialog
    from feabas_workbench.core.steps import STEPS_BY_KEY
    app = QApplication.instance() or QApplication([])
    ctx = AppContext(Settings()); ctx.project = ctx.local_project = project; ctx.configs = ConfigStore(project.configs_dir)
    monkeypatch.setattr(ctx, 'require_feabas_python', lambda: sys.executable)
    step = STEPS_BY_KEY['stitch.matching']
    assert 'local_parallel' not in ' '.join(ctx.feabas_step_spec(step).argv)
    dialog = LocalParallelDialog(ctx, ['stitch.matching'])
    original = {str(f): f.read_bytes() for f in project.configs_dir.glob('*.yaml')}
    mode, sections, workers, measured, _ = dialog.rows['stitch.matching']
    mode.setCurrentIndex(mode.findData('both')); sections.setValue(2); workers.setValue(2); measured.setValue(1)
    dialog.save()
    assert original == {str(f): f.read_bytes() for f in project.configs_dir.glob('*.yaml')}
    spec = ctx.feabas_step_spec(step, start=1, stop=3, stride=1)
    assert 'feabas_workbench.workers.local_parallel' in spec.argv
    payload = json.loads(Path(spec.argv[-1]).read_text())
    assert payload['start'] == 1 and payload['settings']['workers'] == 2
    assert spec.count_outputs is None and spec.progress_fn is None
    zero_end = ctx.feabas_step_spec(step, stop=0)
    assert json.loads(Path(zero_end.argv[-1]).read_text())['stop'] is None
    ctx.cluster_enabled = True
    remote = ctx.feabas_step_spec(step)
    assert remote.remote['step'] == step.key and 'local_parallel' not in ' '.join(remote.argv)
    dialog.close(); app.processEvents()


@requires_feabas
def test_spawn_resource_override_keeps_project_config_untouched(project, tmp_path):
    from feabas_workbench.core.jobs import feabas_env
    original = {str(f): f.read_bytes() for f in project.configs_dir.glob('*.yaml')}
    resource = tmp_path / 'resource.json'
    resource.write_text(json.dumps(dict(kind='stitching', field='matching', workers=2, logging_directory=str(tmp_path))))
    env = dict(os.environ, **feabas_env(), FW_LOCAL_RESOURCES_FILE=str(resource))
    code = ('from feabas import config; import json; '
            "print(json.dumps([config.general_settings()['cpu_budget'], config.stitch_configs()['matching']['num_workers'], config.parallel_framework()]))")
    p = subprocess.run([sys.executable, '-c', code], cwd=project.root, env=env, text=True, capture_output=True, timeout=30)
    assert p.returncode == 0, p.stderr
    assert json.loads(p.stdout.strip().splitlines()[-1]) == [2, 2, 'process']
    assert original == {str(f): f.read_bytes() for f in project.configs_dir.glob('*.yaml')}


@requires_feabas
def test_thumbnail_runtime_listing_uses_names_with_partial_mips(project, tmp_path):
    from feabas_workbench.core.jobs import feabas_env
    folder = project.root / 'stitched_sections/mip2'
    selected = folder / 's0002/metadata.txt'; selected.parent.mkdir(parents=True); selected.touch()
    # s0001 has not finished mipmapping yet. Index 1 would incorrectly be empty.
    resource = tmp_path / 'resource.json'
    resource.write_text(json.dumps(dict(kind='thumbnail', field='downsample', workers=1,
        logging_directory=str(tmp_path), section_name='s0002')))
    env = dict(os.environ, **feabas_env(), FW_LOCAL_RESOURCES_FILE=str(resource))
    code = ("from feabas import storage; from pathlib import Path; import json; "
            "print(json.dumps(storage.list_folder_content(str(Path('stitched_sections/mip2/**/metadata.txt')),recursive=True)))")
    proc = subprocess.run([sys.executable, '-c', code], cwd=project.root, env=env, text=True, capture_output=True, timeout=30)
    assert proc.returncode == 0, proc.stderr
    assert [Path(f).parent.name for f in json.loads(proc.stdout)] == ['s0002']
    assert selected_indices(list(range(8)), 'thumbnail.downsample', 1, 6, 2, reverse=True)[1] == [6, 4, 2]


def test_failed_section_stops_other_processes_and_pending_sections(project, tmp_path, monkeypatch):
    import psutil
    from feabas_workbench.workers import local_parallel as worker
    script = tmp_path / 'driver.py'
    script.write_text("import sys,time\nfrom pathlib import Path\n"
        "index=int(sys.argv[1])\nPath(sys.argv[2], str(index)+'.started').touch()\n"
        "time.sleep(.5 if index==0 else 30)\nsys.exit(2 if index==0 else 0)\n")
    monkeypatch.setattr(worker, 'step_argv', lambda python, step, start, stop, stride, filt:
        [python, str(script), str(start), str(tmp_path)])
    allocation = plan(project, 'stitch.matching', dict(mode='across', workers=1, sections=2, measured_gib=1),
                      available=(4, 16))
    monkeypatch.setattr(worker, 'plan', lambda *a, **kw: allocation)
    processes = []
    original_popen = subprocess.Popen
    def popen(*args, **kwargs):
        child = original_popen(*args, **kwargs)
        if str(script) in args[0]:
            processes.append(child)
        return child
    monkeypatch.setattr(worker.subprocess, 'Popen', popen)
    with pytest.raises(RuntimeError, match='exited 2'):
        worker.execute(dict(root=str(project.root), project=str(project.root), step='stitch.matching', settings={}))
    assert len(processes) == 2
    assert all(not psutil.pid_exists(p.pid) for p in processes)
    assert not (tmp_path / '2.started').exists()


def _aligned_sections(project, n, tiles):
    """*n* rendered aligned sections at mip 0, each listing *tiles* PNG tiles in its metadata.txt."""
    for s in range(n):
        d = project.root / 'aligned_stack/mip0' / f'{s:03d}_s{s:04d}'
        d.mkdir(parents=True, exist_ok=True)
        rows = [f'{{ROOT_DIR}}\t{d}', '{RESOLUTION}\t10.0'] + [f's{s:04d}_tr1-tc{c + 1}.png\t0\t0\t4096\t4096'
                                                              for c in range(tiles)]
        (d / 'metadata.txt').write_text('\n'.join(rows) + '\n')


def test_mipmaps_default_to_automatic_and_small_sections_run_side_by_side(project):
    """FEABAS mipmaps one section at a time and starts a process pool for every mip level, with a
    job per tile of that level: 64-tile sections left a 32-core PC mostly idle. Automatic gives
    whole sections to workers when that is faster, and a read cache that stops tile re-reads."""
    machine = dict(available=(32, 455), threads=64)
    assert options(project, 'align.downsample')['mode'] == 'auto'
    assert options(project, 'thumbnail.downsample')['mode'] == 'auto'
    assert options(project, 'stitch.rendering')['mode'] == 'existing'
    _aligned_sections(project, 20, 64)
    many = mipmap_plan(project, 'align.downsample', **machine)
    assert (many.across, many.sections, many.workers, many.cpu_budget) == (True, 20, 20, 32)
    assert many.values == dict(cache_size=16, parallel_within_section=False, num_workers=20)
    assert "20 sections, 20 at once with one worker each; 16-tile read cache" in many.note
    assert json.loads(many.env()['FW_STAGE_SETTINGS']) == dict(kind='alignment', field='downsample',
                                                               values=many.values, cpu_budget=32)
    two = mipmap_plan(project, 'align.downsample', stop=2, **machine)
    assert (two.across, two.values, two.cpu_budget) == (False, dict(cache_size=16), None)
    assert "2 sections, one at a time with FEABAS's 15 workers each" in two.note
    assert mipmap_plan(project, 'align.downsample', stop=5, **machine).across      # 5 small ones
    _aligned_sections(project, 20, 5000)
    assert not mipmap_plan(project, 'align.downsample', stop=5, **machine).across  # 5 large: FEABAS's way
    assert mipmap_plan(project, 'align.downsample', **machine).across              # 20 >= 15 workers
    per = mipmap_worker_gib(project, ConfigStore(project.configs_dir), 'align.downsample')
    tight = mipmap_plan(project, 'align.downsample', available=(32, 10), threads=64)
    assert 1.5 < per < 2 and tight.workers == int(8 // per) and "RAM budget 8 GiB" in tight.note


def test_automatic_mipmaps_keep_what_the_project_sets(project):
    machine = dict(available=(32, 455), threads=64)
    _aligned_sections(project, 20, 64)
    cs = ConfigStore(project.configs_dir)
    cs['alignment'].set('downsample.cache_size', 8)
    cs['alignment'].set('downsample.num_workers', 6)
    cs.save('alignment')
    own = mipmap_plan(project, 'align.downsample', **machine)
    assert own.values == dict(parallel_within_section=False, num_workers=6) and "8-tile read cache" in own.note
    cs['alignment'].set('downsample.parallel_within_section', False); cs.save('alignment')
    assert mipmap_plan(project, 'align.downsample', stop=2, **machine).across     # chosen in the settings
    project.state.local_execution = dict(steps={'align.downsample': dict(mode='existing')})
    project.save()
    from feabas_workbench.core.pipeline import _automatic_mipmaps
    from feabas_workbench.core.steps import STEPS_BY_KEY
    assert _automatic_mipmaps(project.root, STEPS_BY_KEY['align.downsample']) is None      # FEABAS as set up
    assert _automatic_mipmaps(project.root, STEPS_BY_KEY['stitch.rendering']) is None
    stitched = project.root / 'stitched_sections/mip0/s0001'; stitched.mkdir(parents=True)
    (stitched / 'metadata.txt').write_text('{ROOT_DIR}\tx\n{RESOLUTION}\t10.0\ns0001_tr1-tc1.png\t0\t0\t9\t9\n')
    thumbs = _automatic_mipmaps(project.root, STEPS_BY_KEY['thumbnail.downsample'])
    assert (thumbs.kind, thumbs.field, thumbs.sections) == ('thumbnail', 'downsample', 1)
    cs['stitching'].set('rendering.driver', 'neuroglancer_precomputed'); cs.save('stitching')
    assert mipmap_plan(project, 'thumbnail.downsample', **machine) is None       # no PNG tiles to mipmap


def test_gui_runs_automatic_mipmaps_with_this_runs_settings(project, monkeypatch):
    from PySide6.QtWidgets import QApplication
    from feabas_workbench.core.envs import Settings
    from feabas_workbench.core.steps import STEPS_BY_KEY
    from feabas_workbench.ui.bridge import AppContext
    from feabas_workbench.ui.local_parallel_dialog import LocalParallelDialog
    app = QApplication.instance() or QApplication([])
    ctx = AppContext(Settings()); ctx.project = ctx.local_project = project; ctx.configs = ConfigStore(project.configs_dir)
    monkeypatch.setattr(ctx, 'require_feabas_python', lambda: sys.executable)
    logged = []
    monkeypatch.setattr(ctx, 'log', lambda text: logged.append(text))
    _aligned_sections(project, 6, 64)
    step = STEPS_BY_KEY['align.downsample']
    spec = ctx.feabas_step_spec(step)
    setting = json.loads(spec.env['FW_STAGE_SETTINGS'])
    assert setting['kind'] == 'alignment' and setting['values']['cache_size'] == 16
    assert 'local_parallel' not in ' '.join(spec.argv) and any('Automatic: 6 sections' in t for t in logged)
    dialog = LocalParallelDialog(ctx, ['align.rendering', 'align.downsample'])
    mode, *_, label = dialog.rows['align.downsample']
    assert mode.currentData() == 'auto' and label.text().startswith('Automatic: 6 sections')
    assert dialog.rows['align.rendering'][0].findData('auto') == -1
    mode.setCurrentIndex(mode.findData('existing')); dialog.save()
    assert 'FW_STAGE_SETTINGS' not in ctx.feabas_step_spec(step).env
    dialog.close(); app.processEvents()


def test_explicit_modes_give_mipmap_sections_the_read_cache(project, tmp_path, monkeypatch):
    from feabas_workbench.workers import local_parallel as worker
    _aligned_sections(project, 2, 4)
    script = tmp_path / 'driver.py'
    script.write_text("import os,sys\nfrom pathlib import Path\n"
                      "Path(sys.argv[2], sys.argv[1]+'.env').write_text(os.environ.get('FW_STAGE_SETTINGS',''))\n")
    monkeypatch.setattr(worker, 'step_argv', lambda python, step, start, stop, stride, filt:
        [python, str(script), str(start), str(tmp_path)])
    monkeypatch.setattr(worker, 'verify_section', lambda *a: None)
    settings = dict(mode='across', workers=1, sections=2, measured_gib=1)
    worker.execute(dict(root=str(project.root), project=str(project.root), step='align.downsample', settings=settings))
    got = json.loads((tmp_path / '0.env').read_text())
    assert got == dict(kind='alignment', field='downsample', values=dict(cache_size=16))


@requires_feabas
def test_stage_settings_reach_feabas_and_leave_the_yaml_alone(project):
    from feabas_workbench.core.jobs import feabas_env
    original = {str(f): f.read_bytes() for f in project.configs_dir.glob('*.yaml')}
    setting = dict(kind='alignment', field='downsample', cpu_budget=7,
                   values=dict(parallel_within_section=False, num_workers=3, cache_size=16))
    env = dict(os.environ, **feabas_env(), FW_STAGE_SETTINGS=json.dumps(setting))
    code = ("from feabas import config; import json; d = config.align_configs()['downsample']; "
            "print(json.dumps([config.general_settings()['cpu_budget'], d['parallel_within_section'], "
            "d['num_workers'], d['cache_size'], d['max_mip']]))")
    p = subprocess.run([sys.executable, '-c', code], cwd=project.root, env=env, text=True, capture_output=True, timeout=30)
    assert p.returncode == 0, p.stderr
    assert json.loads(p.stdout.strip().splitlines()[-1]) == [7, False, 3, 16, 7]
    assert original == {str(f): f.read_bytes() for f in project.configs_dir.glob('*.yaml')}

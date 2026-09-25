import json
from pathlib import Path

import pytest

from feabas_workbench.core.cluster_storage import (estimate_project_bytes, fit_warning, parse_dss_storage,
                                                   preprocessing_copies, project_storage_path, raw_image_bytes,
                                                   storage_message)
from feabas_workbench.core.cluster_transport import ClusterClient, ClusterError, ConnectionSettings


@pytest.fixture
def no_storage():
    return (Path(__file__).parent / "fixtures/dss-no-containers.txt").read_text()


def test_mounted_filesystems_are_not_allocations_but_home_and_its_quota_are_reported(no_storage):
    info = parse_dss_storage(no_storage)
    assert info["directories"] == [] and info["storage_state"] == "none"
    assert info["home"] == "/dss/dsshome1/04/testuser"
    assert (info["home_used"], info["home_limit"]) == (0, 100 * 10**9)
    message = storage_message(info)
    assert "no DSS project container assigned" in message and "home folder (100 GB)" in message


def test_only_accessible_container_paths_are_offered(no_storage):
    text = no_storage.replace("* You have access to the following DSS Containers: *",
        "* You have access to the following DSS Containers: *\n"
        "* project-a on /dss/dssfs02/project-a *\n* /dss/dssfs02/project-a *\n"
        "* project-b on /dss/dssfs03/project-b *")
    info = parse_dss_storage(text)
    assert info["storage_state"] == "found"
    assert info["directories"] == ["/dss/dssfs02/project-a", "/dss/dssfs03/project-b"]
    assert info["home"] == "/dss/dsshome1/04/testuser"


@pytest.mark.parametrize("text", ["", "permission denied", "/dss/dssfs02/project-a",
    "You have access to the following DSS Containers:\n",  # truncated
    "You have access to the following DSS Containers:\nunknown format\nDSS Container usage and limits:"])
def test_unknown_output_does_not_claim_no_allocation(text):
    info = parse_dss_storage(text)
    assert info["storage_state"] == "unknown"
    assert "could not identify" in storage_message(info)


@pytest.mark.parametrize("path", ["/dss", "/dss/dssfs02", "/dss/dssfs02/",
    "/dss/dsshome1/04/user", "/dss/lxclscratch/04/user", "/tmp/project", "/dss/dssfs02/../home"])
def test_invalid_project_storage_is_rejected(path):
    with pytest.raises(ValueError):
        project_storage_path(path)


def test_manual_allocation_path_is_allowed():
    assert project_storage_path(" /dss/dssfs02/project-a/work/ ") == "/dss/dssfs02/project-a/work"


def test_own_home_is_project_storage_but_other_homes_are_not():
    home = "/dss/dsshome1/04/testuser"
    assert project_storage_path(home + "/", home) == home
    assert project_storage_path(home + "/feabas-workbench", home) == home + "/feabas-workbench"
    for other in ("/dss/dsshome1/04/otheruser", "/dss/dsshome1/04/testuser2", "/dss/dsshome1", "/dss/dsshome1/04"):
        with pytest.raises(ValueError):
            project_storage_path(other, home)


def test_storage_estimate_warns_only_when_the_free_quota_is_too_small():
    raw = 15 * 10**9
    assert estimate_project_bytes(raw) == 75 * 10**9
    assert estimate_project_bytes(raw, copies=1) == 90 * 10**9
    assert fit_warning(75 * 10**9, 0, 100 * 10**9) == ""
    warning = fit_warning(75 * 10**9, 80 * 10**9, 100 * 10**9)
    assert "about 75 GB" in warning and "only 20 GB of 100 GB" in warning
    assert fit_warning(75 * 10**9, None, None) == ""   # unknown quota: no false alarm


def test_raw_size_and_preprocessing_copies(tmp_path):
    from feabas_workbench.core.project import Project
    tiles = tmp_path / "tiles"
    (tiles / "s1").mkdir(parents=True)
    (tiles / "s1" / "a.tif").write_bytes(b"x" * 1000)
    (tiles / "b.tif").write_bytes(b"x" * 24)
    (tiles / ".workbench-cluster").mkdir()
    (tiles / ".workbench-cluster" / "ignored").write_bytes(b"x" * 500)
    assert raw_image_bytes(tiles) == 1024
    state = Project.create(tmp_path / "project").state
    assert preprocessing_copies(state) == 0
    state.preprocessing.histmatch["enabled"] = True
    state.preprocessing.active_source = "denoise"
    assert preprocessing_copies(state) == 2


@pytest.mark.parametrize("failed", [False, True])
def test_discovery_separates_empty_storage_from_failed_command(tmp_path, monkeypatch, no_storage, failed):
    client = ClusterClient(ConnectionSettings(username="test"), lambda *_: "", lambda *_: False, tmp_path / "hosts")
    def execute(command):
        if command == "dssusrinfo all":
            if failed:
                raise ClusterError("Cannot inspect /dss/dssfs02/project-a")
            return no_storage
        return json.dumps(dict(home="/dss/dsshome1/04/test", user="test"))
    monkeypatch.setattr(client, "execute", execute)
    info = client.discover()
    assert info["directories"] == []
    assert info["storage_state"] == ("error" if failed else "none")
    assert info["home"].startswith("/dss/dsshome1/04/test")   # home stays usable either way

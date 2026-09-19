import json
from pathlib import Path

import pytest

from feabas_workbench.core.cluster_storage import parse_dss_storage, project_storage_path, storage_message
from feabas_workbench.core.cluster_transport import ClusterClient, ClusterError, ConnectionSettings


@pytest.fixture
def no_storage():
    return (Path(__file__).parent / "fixtures/dss-no-containers.txt").read_text()


def test_mounted_filesystems_and_home_are_not_allocations(no_storage):
    info = parse_dss_storage(no_storage)
    assert info == dict(directories=[], storage_state="none")
    assert "no DSS project containers assigned" in storage_message(info)


def test_only_accessible_container_paths_are_offered(no_storage):
    text = no_storage.replace("* You have access to the following DSS Containers: *",
        "* You have access to the following DSS Containers: *\n"
        "* project-a on /dss/dssfs02/project-a *\n* /dss/dssfs02/project-a *\n"
        "* project-b on /dss/dssfs03/project-b *")
    assert parse_dss_storage(text) == dict(storage_state="found",
        directories=["/dss/dssfs02/project-a", "/dss/dssfs03/project-b"])


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

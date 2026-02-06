# This file is adapted from https://github.com/jennyzzt/dgm.

from unittest.mock import Mock

import docker
import pytest

from utils.docker_utils import get_docker_client


def test_get_docker_client_success(monkeypatch):
    mock_client = Mock()
    mock_client.ping.return_value = True
    monkeypatch.setattr(docker, "from_env", lambda: mock_client)

    client = get_docker_client("unit test")

    assert client is mock_client
    mock_client.ping.assert_called_once()


def test_get_docker_client_wraps_docker_exception(monkeypatch):
    def raise_docker_exception():
        raise docker.errors.DockerException("Permission denied")

    monkeypatch.setattr(docker, "from_env", raise_docker_exception)

    with pytest.raises(RuntimeError) as exc:
        get_docker_client("SWE-bench harness setup")

    message = str(exc.value)
    assert "SWE-bench harness setup requires Docker daemon access" in message
    assert "Permission denied" in message
    assert "newgrp docker" in message

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import pytest


def _load_hm_cuda_container_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "hm_cuda_container.py"
    spec = importlib.util.spec_from_file_location("hm_cuda_container", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _build_args(*, network: str | None) -> argparse.Namespace:
    return argparse.Namespace(
        tag="hm-cuda-test",
        network=network,
        username="colivier",
        uid=1000,
        gid=1000,
        cuda_base="nvidia/cuda:12.4.1-devel-ubuntu22.04",
        torch_index_url="https://download.pytorch.org/whl/cu128",
        torch_version="2.7.1+cu128",
        torchvision_version="0.22.1+cu128",
        torchaudio_version="2.7.1+cu128",
    )


def should_detect_transient_dns_failures():
    hm_cuda_container = _load_hm_cuda_container_module()

    assert hm_cuda_container._is_transient_dns_failure("Temporary failure in name resolution")
    assert hm_cuda_container._is_transient_dns_failure("curl: (6) Could not resolve host")
    assert not hm_cuda_container._is_transient_dns_failure("Successfully built image")


def should_retry_build_with_host_network_when_auto_default_hits_dns_failure(monkeypatch):
    hm_cuda_container = _load_hm_cuda_container_module()
    commands: list[list[str]] = []

    monkeypatch.setattr(hm_cuda_container, "_repo_root", lambda: Path("/repo"))
    monkeypatch.setattr(hm_cuda_container, "_default_build_network", lambda: "default")
    monkeypatch.setattr(hm_cuda_container, "_docker_env", lambda: {"DOCKER_BUILDKIT": "1"})

    def _fake_run_streaming(cmd: list[str], *, cwd=None, env=None):
        commands.append(cmd)
        if len(commands) == 1:
            return 1, True
        return 0, False

    monkeypatch.setattr(hm_cuda_container, "_run_streaming", _fake_run_streaming)

    hm_cuda_container.cmd_build(_build_args(network=None))

    assert len(commands) == 2
    assert commands[0][commands[0].index("--network") + 1] == "default"
    assert commands[1][commands[1].index("--network") + 1] == "host"


def should_not_retry_build_when_network_was_explicit(monkeypatch):
    hm_cuda_container = _load_hm_cuda_container_module()
    commands: list[list[str]] = []

    monkeypatch.setattr(hm_cuda_container, "_repo_root", lambda: Path("/repo"))
    monkeypatch.setattr(hm_cuda_container, "_docker_env", lambda: {"DOCKER_BUILDKIT": "1"})

    def _fake_run_streaming(cmd: list[str], *, cwd=None, env=None):
        commands.append(cmd)
        return 1, True

    monkeypatch.setattr(hm_cuda_container, "_run_streaming", _fake_run_streaming)

    with pytest.raises(SystemExit) as exc_info:
        hm_cuda_container.cmd_build(_build_args(network="default"))

    assert exc_info.value.code == 1
    assert len(commands) == 1
    assert commands[0][commands[0].index("--network") + 1] == "default"

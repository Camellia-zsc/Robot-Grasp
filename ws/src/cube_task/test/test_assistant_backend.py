"""Explicit backend selection, including dependency failures; no model calls."""
import builtins
import sys
from types import SimpleNamespace

import pytest

from cube_task.assistant import server, simulation
from cube_task.tools import assistant_ui


def _stub_server(monkeypatch):
    created = []
    class Controller:
        def __init__(self, **kwargs):
            created.append(kwargs)
        def close(self):
            pass
    class Server:
        def __init__(self, address, controller):
            pass
        def serve_forever(self, **kwargs):
            pass
        def server_close(self):
            pass
    monkeypatch.setattr(server, 'DashboardController', Controller)
    monkeypatch.setattr(server, 'DashboardServer', Server)
    return created


@pytest.mark.parametrize('args,backend', [([], 'mujoco'), (['--backend', 'sim'], 'sim')])
def test_cli_passes_explicit_backend_without_silent_fallback(monkeypatch, args, backend):
    created = _stub_server(monkeypatch)
    monkeypatch.setitem(sys.modules, 'mujoco', SimpleNamespace(__version__='3.3.7'))
    assert assistant_ui.main(args) == 0
    assert len(created) == 1 and created[0]['backend'] == backend


def test_cli_refuses_missing_physics_before_starting_server(monkeypatch, capsys):
    created = _stub_server(monkeypatch)
    original_import = builtins.__import__
    def without_mujoco(name, *args, **kwargs):
        if name == 'mujoco':
            raise ImportError('not installed')
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', without_mujoco)
    with pytest.raises(SystemExit) as exc:
        assistant_ui.main([])
    assert exc.value.code == 2 and not created
    assert 'requirements-physics.txt' in capsys.readouterr().err
    assert assistant_ui.main(['--backend', 'sim']) == 0


def test_cli_refuses_incompatible_physics_version(monkeypatch, capsys):
    created = _stub_server(monkeypatch)
    monkeypatch.setitem(sys.modules, 'mujoco', SimpleNamespace(__version__='4.0.1'))
    with pytest.raises(SystemExit) as exc:
        assistant_ui.main([])
    assert exc.value.code == 2 and not created
    assert 'mujoco==3.3.7' in capsys.readouterr().err


@pytest.mark.parametrize('backend', ['sim', 'mujoco'])
def test_controller_default_factory_receives_selected_backend(monkeypatch, backend):
    received = []
    class Session:
        def __init__(self, **kwargs):
            received.append(kwargs)
        def set_speed(self, speed):
            pass
        def prepare(self):
            return {'scene_version': 1}
        def close(self):
            pass
        def stop(self):
            pass
    class Preview:
        def __init__(self, session):
            pass
        def start(self):
            pass
        def close(self):
            pass
    monkeypatch.setattr(simulation, 'SimulationSession', Session)
    settings = SimpleNamespace(public_info=lambda: {'configured': True})
    controller = server.DashboardController(backend=backend, settings=settings,
        client=object(), preview_factory=Preview)
    try:
        assert controller.wait_idle(3)
        assert len(received) == 1 and received[0]['backend'] == backend
        assert controller._status == 'ready'
    finally:
        controller.close()

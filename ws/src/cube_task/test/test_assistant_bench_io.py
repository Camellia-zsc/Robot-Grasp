"""An external Windows checkpoint reader cannot abort an ongoing trial."""
import ctypes
import json
import os
import threading

import pytest

from cube_task.tools.assistant_bench import write_json


@pytest.mark.skipif(os.name != 'nt', reason='Windows sharing contract')
def test_checkpoint_replace_survives_a_brief_real_windows_read_lock(tmp_path):
    path = tmp_path / 'progress-state.json'
    write_json(path, {'complete': False})
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.CreateFileW.restype = ctypes.c_void_p
    kernel.CreateFileW.argtypes = [ctypes.c_wchar_p, ctypes.c_uint32, ctypes.c_uint32,
        ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p]
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    handle = kernel.CreateFileW(str(path), 0x80000000, 1, None, 3, 0, None)
    assert handle != ctypes.c_void_p(-1).value
    released = threading.Event()
    def unlock():
        kernel.CloseHandle(handle)
        released.set()
    timer = threading.Timer(.06, unlock)
    timer.start()
    try:
        write_json(path, {'complete': True})
        assert released.is_set()
        assert json.loads(path.read_text(encoding='utf8')) == {'complete': True}
        assert not path.with_suffix('.json.tmp').exists()
    finally:
        timer.join()

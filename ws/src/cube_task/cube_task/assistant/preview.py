"""Live wrist-camera thumbnails rendered from detached viewer snapshots.

This viewer never captures perception frames or reads the mutable simulation
world. Its private render world cannot advance time or change a later detection.
"""
from __future__ import annotations

import logging
import math
import threading
import time
from typing import Optional

import cv2
import numpy as np

from ..sim.render import render_rgbd
from ..sim.world import SimObject, SimWorld, SimZone, check_pose


_LOG = logging.getLogger(__name__)
_MAX_FPS = 3.0


class LivePreview:
    """Cache a small live image without using the session's capture path.

    ``start()`` is idempotent and returns this preview. ``close()`` permanently
    stops its worker and is also idempotent. Requested rates above 3 fps are
    capped; width preserves the source camera's aspect ratio.
    """

    def __init__(self, session, fps: float = 3, width: int = 320):
        fps = float(fps)
        if not math.isfinite(fps) or fps <= 0.0:
            raise ValueError('preview fps must be finite and positive')
        if type(width) is not int or width < 1:
            raise ValueError('preview width must be a positive integer')
        intrinsics = session.camera.intrinsics()
        if intrinsics is None:
            raise ValueError('preview needs camera intrinsics')
        height = max(1, round(width * intrinsics.height / intrinsics.width))
        self._intrinsics = intrinsics.scaled(width, height)
        self._T_flange_camera = check_pose(
            session.camera.T_flange_camera, 'preview T_flange_camera').copy()
        self._session = session
        self._fps = min(fps, _MAX_FPS)
        self._stop = threading.Event()
        self._lifecycle_lock = threading.Lock()
        self._cache_lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._jpeg: Optional[bytes] = None
        self._info = {'available': False, 'sim_time': None, 'width': width,
                      'height': height, 'updated_at': None, 'fps': self._fps,
                      'mode': 'live'}

    def start(self) -> 'LivePreview':
        """Start one preview worker; a closed preview cannot be restarted."""
        with self._lifecycle_lock:
            if self._stop.is_set():
                raise RuntimeError('preview is closed')
            if self._thread is None:
                self._thread = threading.Thread(
                    target=self._run, name='assistant-live-preview', daemon=True)
                self._thread.start()
        return self

    def close(self) -> None:
        """Stop promptly after any render in progress, retaining the last image."""
        with self._lifecycle_lock:
            self._stop.set()
            worker = self._thread
        if worker is not None and worker is not threading.current_thread():
            worker.join()

    def frame_jpeg(self) -> Optional[bytes]:
        """Read the latest completed JPEG without waiting for a render."""
        with self._cache_lock:
            return self._jpeg

    def info(self) -> dict:
        """Describe the cached image, including its snapshot simulation time."""
        with self._cache_lock:
            return dict(self._info)

    def _run(self) -> None:
        previous = None
        failed = False
        period = 1.0 / self._fps
        while not self._stop.is_set():
            started = time.monotonic()
            try:
                snapshot = self._session.snapshot()
                if snapshot and (failed or snapshot != previous) and not self._stop.is_set():
                    jpeg = self._render(snapshot)
                    with self._cache_lock:
                        if not self._stop.is_set():
                            self._jpeg = jpeg
                            self._info.update(available=True,
                                              sim_time=float(snapshot['sim_time']),
                                              updated_at=time.time())
                    previous = snapshot
                    failed = False
            except Exception:
                with self._cache_lock:
                    self._info['available'] = False
                if not failed and not self._stop.is_set():
                    _LOG.exception('live camera preview could not render a snapshot')
                failed = True
            self._stop.wait(max(0.0, period - (time.monotonic() - started)))

    def _render(self, snapshot: dict) -> bytes:
        objects = [SimObject(
            name=obj['id'], label=obj['label'], shape=obj['shape'], size=obj['size'],
            color_bgr=list(reversed(obj['color_rgb'])), T_base_obj=obj['transform'])
            for obj in snapshot['objects']]
        zones = [SimZone(
            name=zone['name'], center=zone['center'], size=zone['size'],
            yaw=zone.get('yaw', 0.0), label=zone.get('label', ''),
            color_bgr=list(reversed(zone['color_rgb'])),
            wall_height=zone.get('wall_height', 0.0))
            for zone in snapshot['zones']]
        world = SimWorld(table_z=snapshot['table_z'], objects=objects, zones=zones)
        # Store the gripper first, then mark any held object. The object's exact
        # snapshot transform must survive reconstruction without a grasp update.
        world.update_tcp(snapshot['tcp_transform'], snapshot['gripper_width'] or 0.0)
        world.held = next((obj for obj, state in zip(objects, snapshot['objects'])
                           if state.get('held')), None)
        T_base_camera = (np.asarray(snapshot['link_frames'][-1], dtype=float)
                         @ self._T_flange_camera)
        color, _depth = render_rgbd(
            world, self._intrinsics, T_base_camera,
            rng=None, noise_std=0.0, dropout=0.0)
        ok, encoded = cv2.imencode('.jpg', color, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            raise RuntimeError('preview JPEG encoding failed')
        return encoded.tobytes()

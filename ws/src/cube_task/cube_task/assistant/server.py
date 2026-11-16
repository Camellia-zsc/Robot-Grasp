"""Loopback-only dashboard and single-owner simulation task controller."""
from __future__ import annotations

import copy
import json
import logging
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

_LOG = logging.getLogger(__name__)
STATIC = Path(__file__).with_name('static')


class BusyError(ValueError):
    pass


class DashboardController:
    """One worker owns all robot operations; web readers see cached state."""

    def __init__(self, seed=0, realtime=True, session_factory=None, client=None,
                 settings=None, log_dir=None, preview_factory=None):
        from .llm import ModelClient, load_settings
        from .preview import LivePreview
        from .simulation import SimulationSession

        self.seed = int(seed)
        self.realtime = bool(realtime)
        self.log_dir = log_dir
        self._session_factory = session_factory or SimulationSession
        self._preview_factory = preview_factory or (LivePreview if session_factory is None else None)
        self._lock = threading.RLock()
        self._cancel = threading.Event()
        self._worker = None
        self._session = None
        self._preview = None
        self._closed = False
        self._busy = False
        self._status = 'starting'
        self._stage = '正在准备仿真'
        self._error = ''
        self._scene = {}
        self._messages = []
        self._events = deque(maxlen=160)
        self._next_id = 0
        self._last_result = None
        self._speed = 1.0
        self._needs_reset = False
        self._recovery = {'attempt': 0, 'max_attempts': 3, 'status': 'idle',
                          'reason': '', 'message': '', 'history': []}
        self._metrics = {'requests': 0, 'successes': 0, 'model_requests': 0}
        self._last_usage = {}
        self._model = {'configured': False, 'model': '', 'provider': '', 'api_style': ''}
        self._client = client
        try:
            settings = settings or load_settings()
            self._model = settings.public_info()
            self._client = client or ModelClient(settings)
        except (ValueError, OSError) as exc:
            self._error = '模型配置尚未就绪：%s' % exc
        self._launch(self._reset_impl, 'preparing', '正在准备 L1 仿真')

    def _event(self, kind, payload=None):
        with self._lock:
            self._next_id += 1
            event = {'id': self._next_id, 'kind': kind, 'time': time.strftime('%H:%M:%S')}
            if isinstance(payload, dict):
                event.update(copy.deepcopy(payload))
            elif payload is not None:
                event['message'] = str(payload)
            self._events.append(event)

    def _on_sim_event(self, kind, payload=None):
        self._event(kind, payload)
        if isinstance(payload, dict) and payload.get('message'):
            with self._lock:
                if self._status == 'executing' or (
                        kind == 'scan_view' and self._status in ('observing', 'preparing')):
                    self._stage = str(payload['message'])

    def _set_status(self, status, stage):
        with self._lock:
            self._status = status
            self._stage = stage

    def _launch(self, fn, status, stage):
        with self._lock:
            if self._closed:
                raise ValueError('界面服务已关闭')
            if self._busy:
                raise BusyError('上一项任务还在进行，请等待完成或先停止')
            self._cancel.clear()
            self._busy = True
            self._status, self._stage = status, stage
            self._error = ''

            def work():
                try:
                    fn()
                except Exception as exc:
                    # API errors are redacted by llm.py; never log response bodies.
                    with self._lock:
                        cancelled = self._cancel.is_set()
                        if cancelled:
                            self._status, self._stage = 'stopped', '任务已停止，可以重置场景'
                        else:
                            safe = (str(exc) if isinstance(exc, (ValueError, RuntimeError))
                                    else '操作失败（%s）' % type(exc).__name__)
                            self._error = safe
                            self._status, self._stage = 'error', safe
                        for message in self._messages:
                            if message.get('status') == 'streaming':
                                message['status'] = 'cancelled' if cancelled else 'error'
                        if self._session is not None:
                            sim = self._session.snapshot()
                            if sim.get('status') == 'stopped':
                                self._needs_reset = True
                    self._event('stopped' if cancelled else 'error', {'message': self._stage})
                finally:
                    with self._lock:
                        if self._cancel.is_set():
                            self._status, self._stage = 'stopped', '任务已停止，可以重置场景'
                            self._needs_reset = True
                        self._busy = False
                    if self._closed:
                        self._dispose_session()

            self._worker = threading.Thread(target=work, name='grasp-task', daemon=True)
            self._worker.start()

    def _reset_impl(self):
        with self._lock:
            previous = self._session
            previous_preview = self._preview
            self._session = None
            self._preview = None
            self._scene = {}
            self._needs_reset = True
            self._recovery = {'attempt': 0, 'max_attempts': 3, 'status': 'idle',
                              'reason': '', 'message': '', 'history': []}
        if previous_preview is not None:
            previous_preview.close()
        if previous is not None:
            previous.close()
        if self._cancel.is_set():
            raise RuntimeError('场景准备已取消')
        session = self._session_factory(seed=self.seed, realtime=self.realtime,
                                        log_dir=self.log_dir, on_event=self._on_sim_event)
        with self._lock:
            self._session = session
            if self._cancel.is_set():
                session.stop()
                raise RuntimeError('场景准备已取消')
            if self._preview_factory is not None:
                self._preview = self._preview_factory(session)
                self._preview.start()
        session.set_speed(self._speed)
        scene = session.prepare()
        with self._lock:
            if self._cancel.is_set():
                session.stop()
                raise RuntimeError('场景准备已取消')
            self._scene = scene
            self._last_result = None
            self._needs_reset = False
            self._status, self._stage = 'ready', '场景已就绪，输入一句中文开始抓取'
        self._event('ready', {'message': 'L1 场景已就绪', 'scene_version': scene.get('scene_version')})

    def reset(self):
        self._launch(self._reset_impl, 'preparing', '正在重置仿真场景')

    def submit(self, instruction):
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError('请输入抓取指令')
        instruction = instruction.strip()
        if len(instruction) > 1500:
            raise ValueError('指令最多 1500 个字符')
        if not self._model.get('configured') or self._client is None:
            raise ValueError('未找到可用的模型配置，请按使用说明配置本机 API')
        if self._needs_reset:
            raise ValueError('任务已停止，请先重置场景再执行新指令')
        if self._session is None or not self._scene:
            raise ValueError('仿真尚未准备好，请先重置场景')

        def run():
            with self._lock:
                self._next_id += 1
                self._messages.append({'id': self._next_id, 'role': 'user',
                                       'content': instruction, 'status': 'complete'})
                self._messages = self._messages[-60:]
                self._last_result = None
                self._recovery = {'attempt': 0, 'max_attempts': 3, 'status': 'idle',
                                  'reason': '', 'message': '', 'history': []}
                self._metrics['requests'] += 1
            self._set_status('observing', '正在观察桌面并生成物体列表')
            scene = self._session.observe()
            if self._cancel.is_set():
                raise RuntimeError('任务已停止')
            with self._lock:
                self._scene = scene
                self._next_id += 1
                reply = {'id': self._next_id, 'role': 'assistant', 'content': '',
                         'status': 'streaming'}
                self._messages.append(reply)
            self._set_status('thinking', '模型正在理解指令，回复将实时显示')

            def on_delta(text):
                if self._cancel.is_set():
                    return
                with self._lock:
                    reply['content'] += text

            with self._lock:
                self._metrics['model_requests'] += 1
            decision = self._client.decide(instruction, scene, on_delta,
                                            cancel_event=self._cancel)
            if self._cancel.is_set():
                raise RuntimeError('任务已停止')
            from .llm import validate_decision
            decision = validate_decision(decision, scene)
            with self._lock:
                if self._cancel.is_set():
                    raise RuntimeError('任务已停止')
                reply['status'] = 'complete'
                reply['decision'] = decision
                self._last_usage = copy.deepcopy(getattr(self._client, 'last_usage', {}))
            self._event('decision', {'message': decision['message'], 'decision': decision})
            if decision['action'] != 'pick':
                with self._lock:
                    if self._cancel.is_set():
                        raise RuntimeError('任务已停止')
                    self._status, self._stage = 'ready', decision['message']
                return
            with self._lock:
                if self._cancel.is_set():
                    raise RuntimeError('任务已停止')
                self._status, self._stage = 'executing', '正在执行接近、抓取、抬升和放置'
            result = self._session.execute(decision, cancel_event=self._cancel)
            if self._cancel.is_set():
                raise RuntimeError('任务已停止')
            with self._lock:
                self._last_result = result
                self._scene = self._session.scene
            if not result.get('ok') and not result.get('aborted'):
                selected = next(obj for obj in scene['objects']
                                if obj['object_id'] == decision['object_id'])
                goal = {'label': selected['label'], 'zone_name': decision['zone_name']}
                result = self._recover_task(instruction, goal, result)
            with self._lock:
                if self._cancel.is_set():
                    raise RuntimeError('任务已停止')
                self._last_result = result
                self._scene = self._session.scene
                ok = bool(result.get('ok'))
                if ok:
                    self._metrics['successes'] += 1
                if result.get('holding') or result.get('aborted') or result.get('fault'):
                    self._needs_reset = True
                self._status = 'complete' if ok else 'error'
                self._stage = result.get('message') or ('抓取和放置已完成' if ok else '本次抓取失败')
                if not ok:
                    self._error = self._stage
            self._event('result', {'message': self._stage, 'result': result})

        self._launch(run, 'observing', '正在开始任务')

    def _recover_task(self, instruction, goal, initial_result):
        """Ask the real model to select bounded local repairs, checking every result."""
        from .recovery import MAX_RECOVERY_ATTEMPTS, validate_recovery_decision

        if not (hasattr(self._session, 'prepare_recovery') and
                hasattr(self._client, 'recover')):
            return initial_result
        result = initial_result
        history = []
        reason = ''
        for attempt in range(1, MAX_RECOVERY_ATTEMPTS + 1):
            if self._cancel.is_set():
                raise RuntimeError('任务已停止')
            diagnostic = result.get('diagnostics') or result.get('error') or result.get('message')
            with self._lock:
                self._recovery.update(attempt=attempt, status='observing',
                                      reason=str(diagnostic or ''),
                                      message='正在重新核对现场并生成恢复方案',
                                      history=copy.deepcopy(history))
            self._set_status('observing', '正在为第 %d 次恢复重新观测现场' % attempt)
            self._event('recovery_start', {'attempt': attempt,
                         'message': '执行异常，开始第 %d/%d 次恢复' %
                                    (attempt, MAX_RECOVERY_ATTEMPTS)})
            reply = None
            try:
                context = self._session.prepare_recovery(result, goal, attempt, history)
                if self._cancel.is_set():
                    raise RuntimeError('任务已停止')
                with self._lock:
                    self._scene = self._session.scene
                if not context.get('options'):
                    reason = '当前没有满足原目标的可执行恢复方案，已保留现场'
                    break
                with self._lock:
                    self._next_id += 1
                    reply = {'id': self._next_id, 'role': 'assistant', 'kind': 'recovery',
                             'attempt': attempt, 'content': '', 'status': 'streaming'}
                    self._messages.append(reply)
                    self._messages = self._messages[-60:]
                    self._recovery.update(status='thinking', message='模型正在分析异常和恢复方案')
                    self._metrics['model_requests'] += 1
                self._set_status('thinking', '模型正在给出第 %d 次恢复方案' % attempt)

                def on_delta(text):
                    if not self._cancel.is_set():
                        with self._lock:
                            reply['content'] += text

                chosen = self._client.recover(instruction, context, on_delta,
                                               cancel_event=self._cancel)
                if self._cancel.is_set():
                    raise RuntimeError('任务已停止')
                chosen = validate_recovery_decision(chosen, context)
                with self._lock:
                    if self._cancel.is_set():
                        raise RuntimeError('任务已停止')
                    reply['status'], reply['decision'] = 'complete', chosen
                    self._last_usage = copy.deepcopy(getattr(self._client, 'last_usage', {}))
                if chosen['action'] == 'stop':
                    reason = chosen['message']
                    break
                with self._lock:
                    if self._cancel.is_set():
                        raise RuntimeError('任务已停止')
                    self._recovery.update(status='executing', message=chosen['message'])
                    self._status, self._stage = 'executing', '正在执行第 %d 次恢复：%s' % (
                        attempt, chosen['message'])
                if self._cancel.is_set():
                    raise RuntimeError('任务已停止')
                result = self._session.execute_recovery(chosen, cancel_event=self._cancel)
                if self._cancel.is_set():
                    raise RuntimeError('任务已停止')
                history.append({'attempt': attempt, 'option_id': chosen['option_id'],
                                'action': chosen['action'], 'ok': bool(result.get('ok')),
                                'stage': result.get('stage', ''), 'error': result.get('error', ''),
                                'holding': bool(result.get('holding')),
                                'message': result.get('message', '')})
                with self._lock:
                    self._last_result = result
                    self._scene = self._session.scene
                    self._recovery['history'] = copy.deepcopy(history)
                self._event('recovery_result', {'attempt': attempt, 'ok': bool(result.get('ok')),
                             'message': '第 %d 次恢复%s：%s' % (attempt,
                                '核验成功' if result.get('ok') else '仍未完成',
                                result.get('message', ''))})
                if result.get('ok'):
                    with self._lock:
                        self._recovery.update(status='succeeded', message='恢复后已通过落位核验')
                    return dict(result, recovery_history=history)
                if result.get('aborted'):
                    raise RuntimeError('任务已停止')
                if result.get('stage') == 'plan':
                    reason = '本地规划没有找到新的可执行动作，已停止重复尝试并保留现场'
                    break
            except Exception as exc:
                if self._cancel.is_set():
                    raise
                reason = (str(exc) if isinstance(exc, (ValueError, RuntimeError)) else
                          '恢复过程失败（%s）' % type(exc).__name__)
                if reply is not None and reply.get('status') == 'streaming':
                    with self._lock:
                        reply['status'] = 'error'
                # The failed action may have changed physical state or latched
                # a session stop. Keep conservative flags even if no new result
                # made it back from the adapter.
                current = self._session.snapshot()
                holding = (bool(current['held_object']) if 'held_object' in current
                           else bool(result.get('holding')))
                result = dict(result, ok=False, holding=holding, fault=True)
                with self._lock:
                    self._needs_reset = True
                break
        reason = reason or '已尝试 3 次恢复，仍未通过落位核验，已保留现场'
        with self._lock:
            self._recovery.update(status='exhausted', message=reason,
                                  history=copy.deepcopy(history))
        self._event('recovery_exhausted', {'message': reason})
        return dict(result, ok=False, message=reason, recovery_history=history,
                    recovery_reason=reason)

    def stop(self):
        with self._lock:
            if not self._busy:
                return
            self._cancel.set()
            self._needs_reset = True
            self._status, self._stage = 'stopping', '正在停止当前任务'
            if self._recovery['status'] != 'idle':
                self._recovery.update(status='stopped', message='恢复已停止，保留当前夹持与位置')
            session = self._session
        if session is not None:
            session.stop()

    def set_speed(self, speed):
        if isinstance(speed, bool) or speed not in (0.5, 1, 2, 4):
            raise ValueError('速度只能为 0.5、1、2 或 4 倍')
        with self._lock:
            self._speed = float(speed)
            session = self._session
        if session is not None:
            session.set_speed(self._speed)

    def snapshot(self):
        with self._lock:
            session = self._session
            preview = self._preview
            state = {'status': self._status, 'stage': self._stage, 'busy': self._busy,
                     'error': self._error, 'model': copy.deepcopy(self._model),
                     'scene': copy.deepcopy(self._scene), 'speed': self._speed,
                     'needs_reset': self._needs_reset,
                     'recovery': copy.deepcopy(self._recovery),
                     'metrics': copy.deepcopy(self._metrics),
                     'usage': copy.deepcopy(self._last_usage),
                     'messages': copy.deepcopy(self._messages),
                     'events': copy.deepcopy(list(self._events)),
                     'last_result': copy.deepcopy(self._last_result)}
        state['simulation'] = session.snapshot() if session is not None else {}
        state['camera'] = (preview.info() if preview is not None else
                           {'available': bool(state['simulation'].get('camera_available')),
                            'mode': 'observation'})
        state['camera_available'] = state['camera'].get('available', False)
        state['observation_views'] = (session.observation_views()
                                      if session is not None and hasattr(session, 'observation_views')
                                      else [])
        return state

    def camera_jpeg(self, mode='live', view=None):
        with self._lock:
            session, preview = self._session, self._preview
        if mode == 'live':
            return preview.frame_jpeg() if preview is not None else None
        if mode != 'observation':
            raise ValueError('未知相机模式')
        if session is None:
            return None
        if view is not None:
            if not hasattr(session, 'observation_frame_jpeg'):
                return None
            return session.observation_frame_jpeg(view)
        return session.frame_jpeg()

    def wait_idle(self, timeout=30):
        worker = self._worker
        if worker:
            worker.join(timeout)
        return not self._busy

    def close(self):
        self._closed = True
        self.stop()
        if self.wait_idle(3):
            self._dispose_session()

    def _dispose_session(self):
        with self._lock:
            session, self._session = self._session, None
            preview, self._preview = self._preview, None
        if preview is not None:
            preview.close()
        if session is not None:
            try:
                session.close()
            except Exception as exc:
                _LOG.warning('closing simulation failed (%s)', type(exc).__name__)


class DashboardServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, controller):
        self.controller = controller
        super().__init__(address, DashboardHandler)


class DashboardHandler(BaseHTTPRequestHandler):
    server_version = 'GraspDashboard/1'

    def log_message(self, fmt, *args):
        # No noisy camera/state polling logs, credentials are never in URLs.
        if len(args) > 1 and str(args[1]).startswith('5'):
            _LOG.warning(fmt, *args)

    def _send(self, code, payload, content_type='application/json; charset=utf-8'):
        body = payload if isinstance(payload, bytes) else json.dumps(
            payload, ensure_ascii=False, allow_nan=False).encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Referrer-Policy', 'no-referrer')
        self.send_header('Content-Security-Policy',
                         "default-src 'self'; img-src 'self' data:; "
                         "script-src 'self'; style-src 'self'; connect-src 'self'; "
                         "frame-ancestors 'none'; base-uri 'none'")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass

    def _local_request(self):
        port = self.server.server_port
        allowed = {'127.0.0.1:%s' % port, 'localhost:%s' % port}
        if self.headers.get('Host') not in allowed:
            self._send(403, {'error': '仅允许从本机界面访问'})
            return False
        origin = self.headers.get('Origin')
        if origin and origin not in {'http://' + host for host in allowed}:
            self._send(403, {'error': '不允许其他网站提交操作'})
            return False
        return True

    def do_GET(self):
        if not self._local_request():
            return
        path = urlsplit(self.path).path
        if path == '/api/state':
            self._send(200, self.server.controller.snapshot())
        elif path == '/api/camera.jpg':
            params = parse_qs(urlsplit(self.path).query)
            mode = params.get('mode', ['live'])[0]
            if mode not in ('live', 'observation'):
                self._send(400, {'error': '未知相机模式'})
                return
            view = params.get('view', [None])[0]
            if view is not None:
                try:
                    view = int(view)
                except ValueError:
                    self._send(400, {'error': '相机视角编号无效'})
                    return
            frame = self.server.controller.camera_jpeg(mode=mode, view=view)
            self._send(200 if frame else 204, frame or b'', 'image/jpeg')
        elif path == '/favicon.ico':
            self._send(204, b'', 'image/x-icon')
        else:
            files = {'/': ('index.html', 'text/html; charset=utf-8'),
                     '/static/app.js': ('app.js', 'text/javascript; charset=utf-8'),
                     '/static/style.css': ('style.css', 'text/css; charset=utf-8')}
            entry = files.get(path)
            if entry is None or not (STATIC / entry[0]).is_file():
                self._send(404, {'error': '页面不存在'})
                return
            self._send(200, (STATIC / entry[0]).read_bytes(), entry[1])

    def do_POST(self):
        if not self._local_request():
            return
        if self.headers.get_content_type() != 'application/json':
            self._send(415, {'error': '请使用 JSON 提交操作'})
            return
        try:
            size = int(self.headers.get('Content-Length', '0'))
        except ValueError:
            size = -1
        if size < 1 or size > 8192:
            self._send(413, {'error': '请求内容为空或过长'})
            return
        try:
            data = json.loads(self.rfile.read(size))
            if not isinstance(data, dict):
                raise ValueError('请求需要为 JSON 对象')
            path = urlsplit(self.path).path
            ctl = self.server.controller
            if path == '/api/command':
                ctl.submit(data.get('instruction'))
            elif path == '/api/reset':
                ctl.reset()
            elif path == '/api/stop':
                ctl.stop()
            elif path == '/api/speed':
                ctl.set_speed(data.get('speed'))
            else:
                self._send(404, {'error': '接口不存在'})
                return
            self._send(202, {'ok': True})
        except BusyError as exc:
            self._send(409, {'error': str(exc)})
        except (ValueError, TypeError, UnicodeError) as exc:
            self._send(400, {'error': str(exc) or '请求无效'})
        except Exception:
            self._send(500, {'error': '本机服务处理失败，请重置场景后重试'})

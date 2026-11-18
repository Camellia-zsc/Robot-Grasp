"""Loopback-only dashboard and single-owner simulation task controller."""
from __future__ import annotations

import copy
import json
import logging
import threading
import time
import uuid
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

from .batch import (bind_item, build_items, completed_position, idle_batch,
                    whole_footprint_at_goal)

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
        self._conversation_after_id = 0
        self._last_result = None
        from .skill_orchestrator import idle_skill_task
        self._skill_task = idle_skill_task()
        self._skill_memory = None
        self._approval_event = threading.Event()
        self._recovery_approval = {'status': 'none'}
        self._batch = idle_batch()
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
        self._launch(self._reset_impl, 'preparing', '正在准备桌面抓取仿真')

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

    def _approval_scene_current(self, version):
        snapshot = self._session.snapshot()
        current = snapshot.get('scene_version', self._session.scene.get('scene_version'))
        return (current == version and
                not getattr(self._session, '_interventions', ()) and
                not snapshot.get('editor', {}).get('intervention_pending', 0))

    def approve_recovery(self, request):
        if (not isinstance(request, dict) or set(request) !=
                {'request_id', 'scene_version', 'approved'} or
                type(request.get('approved')) is not bool or
                type(request.get('scene_version')) is not int):
            raise ValueError('请明确选择同意或拒绝当前方案')
        with self._lock:
            pending = self._recovery_approval
            if (not self._busy or self._cancel.is_set() or pending.get('status') != 'pending'
                    or request['request_id'] != pending.get('request_id')
                    or request['scene_version'] != pending.get('scene_version')):
                raise ValueError('确认请求已过期，请查看当前方案')
            if not self._approval_scene_current(pending['scene_version']):
                pending.update(status='stale', message='现场已变化，原方案失效，未执行')
                self._approval_event.set()
                raise ValueError('现场已变化，原方案已过期')
            pending.update(status='approved' if request['approved'] else 'declined',
                           message='已同意，准备重新检查并执行' if request['approved'] else '已拒绝，保留现场并停止')
            self._approval_event.set()

    def _await_recovery_approval(self, context, chosen):
        option = next(item for item in context['options'] if item['id'] == chosen['option_id'])
        if not option.get('requires_confirmation'):
            return True
        with self._lock:
            self._approval_event.clear()
            self._recovery_approval = dict(status='pending', request_id=uuid.uuid4().hex,
                scene_version=context['scene_version'],
                question=option.get('question') or '是否按以下方案调整现场并继续原任务？',
                steps=copy.deepcopy(option.get('steps') or [option['description']]),
                message=option['description'])
            self._status, self._stage = 'awaiting_approval', '等待你确认恢复方案，机械臂保持当前状态'
            self._recovery.update(status='awaiting_approval', message=self._stage)
        self._event('recovery_approval', {'message': self._recovery_approval['question']})
        while not self._approval_event.wait(.2):
            if self._cancel.is_set():
                break
            if not self._approval_scene_current(context['scene_version']):
                with self._lock:
                    self._recovery_approval.update(status='stale', message='现场已变化，方案失效，未执行')
                break
        with self._lock:
            if self._cancel.is_set():
                self._recovery_approval.update(status='declined', message='任务已停止，未执行方案')
                raise RuntimeError('任务已停止')
            accepted = self._recovery_approval['status'] == 'approved'
            if accepted and not self._approval_scene_current(context['scene_version']):
                self._recovery_approval.update(status='stale', message='确认后现场又变化，未执行方案')
                accepted = False
            return accepted

    def resume_recovery(self):
        """Explicitly continue a recoverable held failure, preserving its goal."""
        if hasattr(self._client, 'plan_skills'):
            return self._resume_skills()
        with self._lock:
            session = self._session
            if (self._busy or self._closed or self._cancel.is_set() or session is None
                    or not getattr(getattr(session, 'runner', None), 'has_held_recovery', False)
                    or not session.snapshot().get('held_object')
                    or session.cancel_event.is_set()):
                raise ValueError('当前没有可以继续处理的持物任务；已停止的任务需先重置')
            goal = copy.deepcopy(session._recovery_goal)
            failure = copy.deepcopy(self._last_result or {})
            instruction = next((m['content'] for m in reversed(self._messages)
                                if m.get('role') == 'user' and isinstance(m.get('content'), str)),
                               '继续完成当前持物的原目标')
            protected = [self._goal(item['label'], item) for item in self._batch.get('items', [])
                         if item['status'] in ('succeeded', 'skipped')]
            active = next((item for item in self._batch.get('items', [])
                           if item['label'] == goal['label'] and
                           item['status'] not in ('succeeded', 'skipped')), None)
            if active is not None:
                active.update(status='recovering', message='继续分析原持物任务的恢复方案')
                self._batch.update(status='running', current_index=active['index'])
            self._needs_reset = False

            def run():
                result = self._recover_task(instruction, goal, failure, protected_goals=protected)
                with self._lock:
                    self._last_result = copy.deepcopy(result)
                    self._scene = session.scene
                    if self._cancel.is_set():
                        self._last_result.update(ok=False, aborted=True)
                        raise RuntimeError('任务已停止')
                target_ok = bool(result.get('ok'))
                if target_ok and getattr(session, 'targeted_observation', False):
                    changed = [item for item in self._batch.get('items', [])
                               if item['status'] in ('succeeded', 'skipped') and
                               (item.get('result') or {}).get('verified_revision') != session._environment_revision]
                    if changed:
                        checked = session.verify_goals([self._goal(item['label'], item) for item in changed])
                        if not checked['ok']:
                            failed = {v['goal']['label'] for v in checked.get('items', []) if not v['ok']}
                            with self._lock:
                                for item in changed:
                                    if not failed or item['label'] in failed:
                                        item.update(status='failed', message='现场变化后原已完成目标未通过核验')
                                self._recovery.update(status='exhausted', message='当前物品已恢复，但部分原已完成目标发生变化')
                            result = dict(result, ok=False, target_ok=True, final_goal_verification=checked,
                                          message='当前物品已恢复，但部分原已完成目标发生变化')
                with self._lock:
                    if self._cancel.is_set():
                        self._last_result = dict(result, ok=False, aborted=True)
                        raise RuntimeError('任务已停止')
                    if active is not None:
                        active.update(status='succeeded' if target_ok else 'failed',
                                      message=result.get('message', ''), result=copy.deepcopy(result))
                        completed = sum(item['status'] in ('succeeded', 'skipped') for item in self._batch['items'])
                        complete = completed == self._batch['total']
                        self._batch.update(completed=completed, status='complete' if complete else 'failed',
                                           message='全部目标已完成' if complete else
                                           '当前恢复已结束；后续取消的任务未自动重启（完成 %d/%d 件）' % (completed, self._batch['total']))
                        self._last_result = self._batch_result(complete, result)
                    else:
                        self._last_result = result
                    self._scene = session.scene
                    self._needs_reset = bool(result.get('holding') or result.get('fault') or result.get('aborted'))
                    final_ok = bool(self._last_result.get('ok'))
                    self._status = 'complete' if final_ok else 'error'
                    self._stage = (self._batch['message'] if active is not None else
                                   result.get('message') or '恢复尚未完成，保留当前状态')
                    if final_ok:
                        self._metrics['successes'] += 1
                self._event('result', {'message': self._stage, 'result': self._last_result})
            self._launch(run, 'observing', '继续核对当前持物，保留原任务与现场')

    def _resume_skills(self):
        from .skill_orchestrator import SkillOrchestrator
        with self._lock:
            session = self._session
            saved = self._skill_memory
            if saved and self._skill_task.get('status') not in ('failed', 'declined'):
                raise ValueError('当前技能任务无需继续；请等待完成或发送新指令')
            legacy_held = bool(session is not None and session.snapshot().get('held_object')
                and getattr(getattr(session, 'runner', None), 'has_held_recovery', False))
            if (self._busy or self._closed or self._cancel.is_set() or session is None
                    or session.cancel_event.is_set() or not ((saved and saved.get('goals')) or legacy_held)):
                raise ValueError('当前没有可继续的任务；明确停止后需先重置')
            goals = copy.deepcopy(saved['goals'] if saved else [session._recovery_goal])
            if not all(isinstance(g, dict) and g.get('label') for g in goals):
                raise ValueError('未找到原始目标，不能猜测恢复任务')
            protected = (copy.deepcopy(saved['protected_goals']) if saved else
                [self._goal(item['label'], item) for item in self._batch.get('items', [])
                 if item['status'] in ('succeeded', 'skipped') and item['label'] not in {g['label'] for g in goals}])
            instruction = (saved['instruction'] if saved else next(
                (m['content'] for m in reversed(self._messages) if m.get('role') == 'user'),
                '继续完成当前持物的原目标'))
            failure = copy.deepcopy(self._last_result or {})
            self._needs_reset = False
            self._recovery = {'attempt': 0, 'max_attempts': 3, 'status': 'idle',
                              'reason': '', 'message': '', 'history': []}
            self._launch(lambda: SkillOrchestrator(self).run(instruction, goals=goals,
                protected_goals=protected, failure=failure, resume=bool(saved)),
                'observing', '保留原目标，重新规划未完成的基本技能')

    def _stopped_message(self):
        if (self._last_result or {}).get('pending_obligations'):
            return '任务已停止；仍有承诺的原位复位未完成，已保留各步记录和当前现场'
        transaction = (self._last_result or {}).get('pending_rearrangement')
        if transaction:
            return '任务已停止；暂放或区域调整尚未完成，已保留各步记录，请先核对物品位置'
        pending = (self._last_result or {}).get('pending_restore')
        if pending:
            names = {'red': '红色方块', 'green': '绿色方块', 'yellow': '黄色方块',
                     'apple': '苹果', 'banana': '香蕉', 'pear': '梨'}
            label = names.get(pending.get('label'), '临时移开的物品')
            return '任务已停止；%s尚未复位，已保留原位置与朝向，可以重置场景' % label
        return '任务已停止，可以重置场景'

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
                            self._status, self._stage = 'stopped', self._stopped_message()
                        else:
                            safe = (str(exc) if isinstance(exc, (ValueError, RuntimeError))
                                    else '操作失败（%s）' % type(exc).__name__)
                            self._error = safe
                            self._status, self._stage = 'error', safe
                        for message in self._messages:
                            if message.get('status') == 'streaming':
                                message['status'] = 'cancelled' if cancelled else 'error'
                        self._end_batch_on_error(cancelled, self._stage)
                        if self._session is not None:
                            sim = self._session.snapshot()
                            if sim.get('status') == 'stopped':
                                self._needs_reset = True
                    self._event('stopped' if cancelled else 'error', {'message': self._stage})
                finally:
                    with self._lock:
                        if self._cancel.is_set():
                            self._status, self._stage = 'stopped', self._stopped_message()
                            self._needs_reset = True
                        self._busy = False
                    if self._closed:
                        self._dispose_session()

            self._worker = threading.Thread(target=work, name='grasp-task', daemon=True)
            self._worker.start()

    def _reset_impl(self):
        with self._lock:
            # Keep visible chat, but a reset starts a new planning conversation.
            self._conversation_after_id = self._next_id
            previous = self._session
            previous_preview = self._preview
            self._session = None
            self._preview = None
            self._scene = {}
            self._needs_reset = True
            self._batch = idle_batch()
            self._recovery_approval = {'status': 'none'}
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
            from .skill_orchestrator import idle_skill_task
            self._skill_task = idle_skill_task()
            self._skill_memory = None
            self._needs_reset = False
            self._status, self._stage = 'ready', '场景已就绪，输入一句中文开始抓取'
        self._event('ready', {'message': '可编辑场景已就绪', 'scene_version': scene.get('scene_version')})

    def reset(self):
        self._launch(self._reset_impl, 'preparing', '正在重置仿真场景')

    def edit_scene(self, request):
        if not isinstance(request, dict):
            raise ValueError('布局操作需要为 JSON 对象')
        request = copy.deepcopy(request)

        def editable_session():
            if self._needs_reset:
                raise ValueError('任务已停止，请先重置场景再调整布局')
            session = self._session
            if session is None or not self._scene:
                raise ValueError('仿真尚未准备好，请先重置场景')
            sim = session.snapshot()
            if sim.get('held_object'):
                raise ValueError('机械臂仍夹持物品，请先完成放置或重置场景')
            if sim.get('status') == 'stopped':
                raise ValueError('任务已停止，请先重置场景再调整布局')
            return session

        def run():
            with self._lock:
                session = editable_session()
            scene = session.edit_scene(request)
            with self._lock:
                if self._cancel.is_set():
                    raise RuntimeError('布局调整已停止')
                self._scene = scene
                self._last_result = None
                self._batch = idle_batch()
                self._recovery = {'attempt': 0, 'max_attempts': 3, 'status': 'idle',
                                  'reason': '', 'message': '', 'history': []}
                self._error = ''
                self._status, self._stage = 'ready', '布局已变更，下次指令将重新扫描桌面'
            self._event('scene_edited', {'message': self._stage,
                                      'scene_version': scene.get('scene_version')})

        with self._lock:
            if self._busy:
                raise BusyError('上一项任务还在进行，请等待完成或先停止')
            editable_session()
            self._launch(run, 'editing', '正在调整桌面布局')

    def intervene_scene(self, request):
        """Queue a physical scene change without entering the robot worker lock."""
        if not isinstance(request, dict):
            raise ValueError('干预操作需要为 JSON 对象')
        if (request.get('action') != 'move'
                or set(request) - {'action', 'object_id', 'xy', 'yaw', 'scene_version'}):
            raise ValueError('运行中干预只允许移动或旋转已有物品')
        with self._lock:
            session = self._session
            if self._closed or self._needs_reset:
                raise ValueError('任务已停止，请先重置场景再干预')
            if session is None or not self._scene:
                raise ValueError('仿真尚未准备好，请先重置场景')
            if not hasattr(session, 'intervene_scene'):
                raise ValueError('当前仿真不支持运行中干预')
        # The session queues this request and applies it at an atomic simulation
        # step; it must never wait for the complete pick-and-place operation.
        result = session.intervene_scene(copy.deepcopy(request))
        result = result if isinstance(result, dict) else {'queued': True}
        self._event('scene_intervention_queued', {
            'message': '人为干预已受理；生效后将通过相机局部确认变化',
            'request_id': result.get('request_id')})
        return result

    def submit(self, instruction, destination=None, scene_version=None):
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
        selected_destination = None
        with self._lock:
            if destination is not None:
                from .destinations import validate_destination
                if (type(scene_version) is not int
                        or scene_version != self._scene.get('scene_version')):
                    raise ValueError('选定位置的场景已变化，请重新选择放置点')
                selected_destination = validate_destination(destination, self._scene)
                if selected_destination['type'] != 'point':
                    raise ValueError('界面选择的放置目标必须是桌面点')
            elif scene_version is not None:
                raise ValueError('提交场景版本时必须同时选择放置点')

        def run():
            with self._lock:
                from .conversation import conversation_summary
                conversation = conversation_summary(
                    self._messages, after_id=self._conversation_after_id)
                self._next_id += 1
                self._messages.append({'id': self._next_id, 'role': 'user',
                                       'content': instruction, 'status': 'complete'})
                self._messages = self._messages[-60:]
                self._last_result = None
                self._batch = idle_batch()
                self._recovery = {'attempt': 0, 'max_attempts': 3, 'status': 'idle',
                                  'reason': '', 'message': '', 'history': []}
                self._metrics['requests'] += 1
            if hasattr(self._client, 'plan_skills'):
                from .skill_orchestrator import SkillOrchestrator
                return SkillOrchestrator(self).run(instruction, selected_destination=selected_destination,
                                                   conversation=conversation)
            self._set_status('observing', '复用桌面观测生成任务计划，执行前局部确认')
            scene = self._session.scene
            if not getattr(self._session, 'targeted_observation', False) or not scene.get('objects'):
                scene = self._session.observe()
            if selected_destination is not None:
                from .destinations import validate_destination
                scene = copy.deepcopy(scene)
                scene['selected_destination'] = validate_destination(selected_destination, scene)
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
            if decision['action'] == 'pick_batch':
                self._execute_batch(instruction, decision, scene)
                return
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
            if (not result.get('ok') and not result.get('aborted')
                    and result.get('stage') != 'validation'):
                selected = next(obj for obj in scene['objects']
                                if obj['object_id'] == decision['object_id'])
                goal = self._goal(selected['label'], decision)
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

        with self._lock:
            if (selected_destination is not None
                    and scene_version != self._scene.get('scene_version')):
                raise ValueError('选定位置的场景已变化，请重新选择放置点')
            self._launch(run, 'observing', '正在开始任务')

    @staticmethod
    def _goal(label, decision):
        goal = {'label': label, 'zone_name': decision['zone_name']}
        if decision.get('destination') is not None:
            goal['destination'] = copy.deepcopy(decision['destination'])
        return goal

    def _end_batch_on_error(self, cancelled, message):
        """Called under the controller lock, including unexpected worker failures."""
        if self._batch['status'] != 'running':
            return
        self._batch.update(status='stopped' if cancelled else 'failed', message=message)
        for item in self._batch['items']:
            if item['status'] not in ('succeeded', 'skipped', 'failed'):
                active_failure = not cancelled and item['index'] == self._batch['current_index']
                item.update(status='failed' if active_failure else 'cancelled',
                            message=message if active_failure else '任务已停止' if cancelled else '前序任务未完成')
        sim = self._session.snapshot() if self._session is not None else {}
        self._last_result = self._batch_result(False, dict(
            self._last_result or {}, holding=bool(sim.get('held_object')),
            aborted=cancelled, fault=not cancelled))
        if not cancelled:
            self._needs_reset = True

    def _batch_result(self, ok, result=None):
        return dict(result or {}, ok=bool(ok), batch=True,
                    completed=self._batch['completed'], total=self._batch['total'],
                    items=copy.deepcopy(self._batch['items']), message=self._batch['message'])

    def _execute_batch(self, instruction, decision, initial_scene):
        items = build_items(decision, initial_scene)
        with self._lock:
            self._batch = dict(status='running', total=len(items), completed=0,
                               current_index=None, items=items, message='按顺序逐件抓放')
        previous_version = None
        need_scan = False
        last_result = {}
        for item in items:
            if self._cancel.is_set():
                raise RuntimeError('任务已停止')
            with self._lock:
                self._batch['current_index'] = item['index']
                self._batch['message'] = '正在处理第 %d/%d 件物品' % (item['index'], len(items))
                item.update(status='observing', message='正在根据最新观测确认目标')
            self._set_status('observing', '正在确认第 %d/%d 件物品' % (item['index'], len(items)))
            moving = False
            try:
                scene = self._session.scene
                if need_scan and (last_result.get('observation_error')
                                  or scene.get('scene_version') == previous_version):
                    moving = True
                    scene = self._session.observe()
                    moving = False
                if self._cancel.is_set():
                    raise RuntimeError('任务已停止')
                with self._lock:
                    self._scene = scene
                completed_zones = tuple(dict.fromkeys(
                    done['zone_name'] for done in items[:item['index'] - 1]
                    if (done['label'] == item['label'] and done['status'] in ('succeeded', 'skipped')
                        and done.get('destination') is None)))
                prior_extended = [done for done in items[:item['index'] - 1]
                                  if (done['label'] == item['label']
                                      and done['status'] in ('succeeded', 'skipped')
                                      and done.get('destination') is not None)]
                if any(done.get('completion_xy') is None for done in prior_extended):
                    raise ValueError('已完成同类物品的落位尚未可靠观察，无法确认下一件身份，已停止后续任务')
                completed_positions = tuple(done['completion_xy'] for done in prior_extended)
                try:
                    single, target = bind_item(item, scene, completed_zones, completed_positions)
                except ValueError:
                    if not getattr(self._session, 'targeted_observation', False):
                        raise
                    scene = self._session.observe(target={'label': item['label'],
                                                  'xy': item['reference_xy']},
                                                  purpose='recover_queued_target')
                    if self._session.editor.metadata().get('one_per_class'):
                        fresh = [d for d in self._session._last_fresh
                                 if d.valid and d.label == item['label'] and d.center_base is not None]
                        if len(fresh) == 1:
                            item['reference_xy'] = fresh[0].center_base[:2].tolist()
                    single, target = bind_item(item, scene, completed_zones, completed_positions)
                with self._lock:
                    item['object_id'] = single['object_id']
                if whole_footprint_at_goal(target, scene, item):
                    # Skipping is still a claim about current reality, so confirm it locally.
                    if getattr(self._session, 'targeted_observation', False):
                        scene = self._session.observe(target=dict(target), purpose='confirm_completed')
                        single, target = bind_item(item, scene, completed_zones, completed_positions)
                        fresh = self._session._last_fresh
                        if not any(d.valid and d.label == target['label'] and d.center_base is not None
                                   and sum((float(a)-float(b))**2 for a,b in
                                           zip(d.center_base[:2], target['center_base'][:2])) < .04**2
                                   for d in fresh):
                            raise ValueError('未在本次局部观测中确认已完成目标，不能跳过执行')
                if whole_footprint_at_goal(target, scene, item):
                    last_result = {'ok': True, 'skipped': True, 'observed_in_zone': True,
                                   'message': '实测完整足迹已满足指定目标，无需搬运'}
                    if getattr(self._session, 'targeted_observation', False):
                        last_result['verified_revision'] = self._session._observation_revision
                    if item.get('destination') is not None:
                        last_result['destination'] = copy.deepcopy(item['destination'])
                    need_scan = False
                    final_status = 'skipped'
                else:
                    with self._lock:
                        item.update(status='executing', message='正在抓取并放到指定区域')
                        self._status = 'executing'
                        self._stage = '正在执行第 %d/%d 件物品' % (item['index'], len(items))
                        self._recovery = {'attempt': 0, 'max_attempts': 3, 'status': 'idle',
                                          'reason': '', 'message': '', 'history': []}
                    self._event('batch_item_start', {'message': self._stage, 'index': item['index']})
                    previous_version = single['scene_version']
                    moving = True
                    last_result = self._session.execute(single, cancel_event=self._cancel)
                    if self._cancel.is_set():
                        raise RuntimeError('任务已停止')
                    with self._lock:
                        self._last_result = last_result
                        self._scene = self._session.scene
                    if (not last_result.get('ok') and not last_result.get('aborted')
                            and last_result.get('stage') != 'validation'
                            and not (last_result.get('truth') or {}).get('success')):
                        with self._lock:
                            item.update(status='recovering', message='正在恢复本件任务')
                        goal = self._goal(item['label'], item)
                        last_result = self._recover_task(instruction, goal, last_result,
                                                         reference_xy=item['reference_xy'],
                                                         completed_zones=completed_zones,
                                                         completed_positions=completed_positions,
                                                         protected_goals=[self._goal(done['label'], done) for done in items
                                                                          if done['status'] in ('succeeded', 'skipped')])
                    elif not last_result.get('ok') and (last_result.get('truth') or {}).get('success'):
                        last_result = dict(last_result, fault=True,
                                           message='本件已落位，但机械臂后续动作异常，已停止后续任务')
                    need_scan = True
                    final_status = 'succeeded' if last_result.get('ok') else 'failed'
            except ValueError as exc:
                last_result = {'ok': False, 'error': 'batch_target_unavailable', 'message': str(exc)}
                if moving:
                    last_result.update(fault=True, holding=bool(self._session.snapshot().get('held_object')))
                final_status = 'failed'
            if self._cancel.is_set():
                raise RuntimeError('任务已停止')
            with self._lock:
                if last_result.get('ok') and item.get('destination') is not None:
                    item['completion_xy'] = completed_position(
                        target, self._session.scene, item, last_result)
                item.update(status=final_status, message=last_result.get('message', ''),
                            result=copy.deepcopy(last_result))
                self._scene = self._session.scene
                if last_result.get('ok'):
                    self._batch['completed'] += 1
                else:
                    self._batch.update(status='failed', message='已完成 %d/%d 件；%s' % (
                        self._batch['completed'], len(items), item['message']))
                    for pending in items[item['index']:]:
                        pending.update(status='cancelled', message='前序任务未完成，不再执行')
                    break
            self._event('batch_item_result', {'index': item['index'], 'ok': last_result.get('ok'),
                                             'message': item['message']})
        if (self._batch['completed'] == len(items) and
                getattr(self._session, 'targeted_observation', False)):
            changed = [item for item in items if (item.get('result') or {}).get('verified_revision')
                       != self._session._environment_revision]
            if changed:
                checked = self._session.verify_goals(changed)
                last_result = dict(last_result, final_verification=checked)
                if not checked['ok']:
                    by_index = {v['goal']['index']: v for v in checked['items']}
                    for item in changed:
                        verified = by_index[item['index']]
                        if not verified['ok']:
                            item.update(status='failed', message='运行中变化后，目标已不在指定区域')
                            self._batch['completed'] -= 1
                    self._batch.update(status='failed', message=checked['message'])
        with self._lock:
            if self._cancel.is_set():
                raise RuntimeError('任务已停止')
            ok = self._batch['completed'] == len(items)
            if ok:
                self._batch.update(status='complete', message='已依次完成 %d/%d 件物品任务' % (len(items), len(items)))
                self._metrics['successes'] += 1
            if last_result.get('holding') or last_result.get('aborted') or last_result.get('fault'):
                self._needs_reset = True
            self._last_result = self._batch_result(ok, last_result)
            self._status = 'complete' if ok else 'error'
            self._stage = self._batch['message']
            if not ok:
                self._error = self._stage
        self._event('result', {'message': self._stage, 'result': self._last_result})

    def _recover_task(self, instruction, goal, initial_result, reference_xy=None,
                      completed_zones=(), completed_positions=(), protected_goals=()):
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
                if reference_xy is None:
                    extras = ({'protected_goals': protected_goals} if protected_goals and
                              getattr(self._session, 'targeted_observation', False) else {})
                    context = self._session.prepare_recovery(result, goal, attempt, history, **extras)
                else:
                    identity = {'reference_xy': reference_xy}
                    if completed_zones:
                        identity['completed_zones'] = completed_zones
                    if completed_positions:
                        identity['completed_positions'] = completed_positions
                    if protected_goals and getattr(self._session, 'targeted_observation', False):
                        identity['protected_goals'] = protected_goals
                    context = self._session.prepare_recovery(result, goal, attempt, history,
                                                              **identity)
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
                if not self._await_recovery_approval(context, chosen):
                    reason = self._recovery_approval.get('message') or '未获得人工同意，已停止'
                    break
                with self._lock:
                    if self._cancel.is_set():
                        raise RuntimeError('任务已停止')
                    self._recovery.update(status='executing', message=chosen['message'])
                    self._status, self._stage = 'executing', '正在执行第 %d 次恢复：%s' % (
                        attempt, chosen['message'])
                if self._cancel.is_set():
                    raise RuntimeError('任务已停止')
                selected = next(item for item in context['options'] if item['id'] == chosen['option_id'])
                if selected.get('requires_confirmation'):
                    result = self._session.execute_recovery(chosen, cancel_event=self._cancel, confirmed=True)
                else:
                    result = self._session.execute_recovery(chosen, cancel_event=self._cancel)
                history.append({'attempt': attempt, 'option_id': chosen['option_id'],
                                'action': chosen['action'], 'ok': bool(result.get('ok')),
                                'stage': result.get('stage', ''), 'error': result.get('error', ''),
                                'holding': bool(result.get('holding')),
                                'message': result.get('message', '')})
                result = dict(result, recovery_history=copy.deepcopy(history))
                with self._lock:
                    self._last_result = result
                    self._scene = self._session.scene
                    self._recovery['history'] = copy.deepcopy(history)
                    if self._cancel.is_set():
                        self._recovery.update(status='stopped', message=self._stopped_message())
                # A cancelled clearance can still return a restoration obligation.
                # Persist that physical result before ending the worker; never
                # start another repair or queue item after this cancellation.
                if self._cancel.is_set():
                    raise RuntimeError('任务已停止')
                self._event('recovery_result', {'attempt': attempt, 'ok': bool(result.get('ok')),
                             'message': '第 %d 次恢复%s：%s' % (attempt,
                                '核验成功' if result.get('ok') else '仍未完成',
                                result.get('message', ''))})
                if result.get('ok'):
                    with self._lock:
                        self._recovery.update(status='succeeded', message='恢复后已通过落位核验')
                    return dict(result, recovery_history=history)
                if result.get('pending_restore'):
                    reason = '移障物品尚未复位，已停止后续任务并保留复位目标'
                    break
                if result.get('pending_rearrangement'):
                    reason = result.get('message') or '多步重排尚未完成，已停止并保留各步现场'
                    break
                if reference_xy is not None and (result.get('truth') or {}).get('success'):
                    result = dict(result, fault=True)
                    reason = '本件已落位，但机械臂后续动作异常，已停止后续任务'
                    break
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
            self._approval_event.set()
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
                     'recovery_approval': copy.deepcopy(self._recovery_approval),
                     'metrics': copy.deepcopy(self._metrics),
                     'usage': copy.deepcopy(self._last_usage),
                     'messages': copy.deepcopy(self._messages),
                     'events': copy.deepcopy(list(self._events)),
                     'last_result': copy.deepcopy(self._last_result)}
            state['skill_task'] = copy.deepcopy(self._skill_task)
            state['batch'] = copy.deepcopy(self._batch)
            state['simulation'] = session.snapshot() if session is not None else {}
            state['can_resume_recovery'] = bool(session is not None and not self._busy
                and not self._cancel.is_set() and state['simulation'].get('held_object')
                and getattr(getattr(session, 'runner', None), 'has_held_recovery', False)
                and not session.cancel_event.is_set())
            if (hasattr(self._client, 'plan_skills') and self._skill_memory
                    and self._skill_memory.get('goals') and not self._busy and session is not None
                    and not self._cancel.is_set() and not session.cancel_event.is_set()
                    and self._skill_task.get('status') in ('failed', 'declined')):
                state['can_resume_recovery'] = True
            editor = state['simulation'].setdefault('editor', {})
            editor['editable'] = bool(
                editor.get('editable', True) and session is not None and self._scene
                and not self._closed and not self._busy and not self._needs_reset
                and not state['simulation'].get('held_object')
                and state['simulation'].get('status') != 'stopped')
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
                     '/static/editor.js': ('editor.js', 'text/javascript; charset=utf-8'),
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

        def unique_fields(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError('请求包含重复字段')
                result[key] = value
            return result

        def reject_constant(_value):
            raise ValueError('请求包含非有限数值')

        try:
            data = json.loads(self.rfile.read(size), object_pairs_hook=unique_fields,
                              parse_constant=reject_constant)
            if not isinstance(data, dict):
                raise ValueError('请求需要为 JSON 对象')
            path = urlsplit(self.path).path
            ctl = self.server.controller
            if path == '/api/command':
                if set(data) == {'instruction'}:
                    ctl.submit(data['instruction'])
                elif set(data) == {'instruction', 'destination', 'scene_version'}:
                    if data['destination'] is None:
                        raise ValueError('请选择有效的桌面放置点')
                    ctl.submit(data['instruction'], destination=data['destination'],
                               scene_version=data['scene_version'])
                else:
                    raise ValueError('指令字段不符合约定；选择位置时需同时提交目标与场景版本')
            elif path == '/api/scene-edit':
                ctl.edit_scene(data)
            elif path == '/api/scene-intervene':
                result = ctl.intervene_scene(data)
                self._send(202, dict(result, ok=True))
                return
            elif path == '/api/recovery-approval':
                ctl.approve_recovery(data)
            elif path == '/api/resume-recovery':
                if data:
                    raise ValueError('继续恢复不接受新目标或坐标')
                ctl.resume_recovery()
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

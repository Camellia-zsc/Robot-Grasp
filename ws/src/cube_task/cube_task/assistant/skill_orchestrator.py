"""Model-composed skills with immutable goals, explicit approval and feedback.

Only the dashboard's worker calls this module. It supplies no robot geometry and
never substitutes a model's completion message for measured final verification.
"""
from __future__ import annotations

import copy
import math
from contextlib import nullcontext

from .skill_protocol import MAX_SKILL_STEPS, validate_skill_plan

MIN_PLANS = 8
MAX_PLANS = 16
MAX_EXECUTED = 64
MAX_STAGNATION = 3
NAMES = {'red': '红色方块', 'green': '绿色方块', 'yellow': '黄色方块',
         'apple': '苹果', 'banana': '香蕉', 'pear': '梨'}
SKILLS = {'observe': '观察', 'pick': '抓取并保持', 'place': '放置',
          'move': '搬移', 'verify': '验证目标', 'ask': '询问人工', 'finish': '最终核验'}


def idle_skill_task():
    return dict(status='idle', plan_number=0, max_plans=MIN_PLANS, stagnation=0,
                executed=0, max_executed=MAX_EXECUTED, current_index=None,
                goals=[], steps=[], history=[], message='')


def plan_limit(goals, initial_steps=()):
    """Allow bounded recovery room proportional to the original task size."""
    return min(MAX_PLANS, max(MIN_PLANS, 2*len(goals)+4, len(initial_steps)//2+4))


class SkillOrchestrator:
    def __init__(self, controller, executor_factory=None):
        if executor_factory is None:
            from .skill_execution import SkillExecutor
            executor_factory = SkillExecutor
        self.ctl = controller
        self.session = controller._session
        self.executor_factory = executor_factory

    def _check_stop(self):
        if self.ctl._cancel.is_set() or self.session.cancel_event.is_set():
            raise RuntimeError('任务已停止')

    def _publish(self, **changes):
        with self.ctl._lock:
            self.ctl._skill_task.update(copy.deepcopy(changes))
            self.ctl._scene = self.session.scene

    def _revision(self):
        return getattr(self.session, '_environment_revision',
                       self.session.snapshot().get('scene_version'))

    def _pending_interventions(self):
        return bool(getattr(self.session, '_interventions', ()) or
                    self.session.snapshot().get('editor', {}).get('intervention_pending', 0))

    def _verification_current(self, result):
        return bool(result and result.get('verified_revision') == self._revision()
                    and not self._pending_interventions())

    def _held_label(self):
        if not self.session.snapshot().get('held_object'):
            return None
        context = getattr(self.session.runner, '_held_recovery_context', None)
        if context and getattr(context.get('det'), 'label', None):
            return context['det'].label
        return None

    def _context(self, memory, failure, executor=None):
        snap = self.session.snapshot()
        scene = self.session.scene
        scene['scene_version'] = snap.get('scene_version', scene['scene_version'])
        if memory.get('selected_destination') is not None:
            scene['selected_destination'] = copy.deepcopy(memory['selected_destination'])
        capability_fn = getattr(executor, 'capabilities', None)
        capabilities = (capability_fn() if callable(capability_fn) else
                        dict(skills=list(SKILLS), observe_modes=['stationary'],
                             holding_allowed_skills=['observe', 'place', 'ask']))
        if not isinstance(capabilities, dict):
            raise ValueError('执行器能力说明无效')
        task = self.ctl._skill_task
        remaining = MAX_EXECUTED-task['executed']
        budget = dict(plan_number=task['plan_number'], max_plans=task['max_plans'],
                      remaining_plans=max(0, task['max_plans']-task['plan_number']),
                      remaining_steps=max(0, remaining), stagnation=memory['stagnation'])
        return dict(scene=scene, goals=copy.deepcopy(memory['goals']),
                    originals=copy.deepcopy(memory['originals']),
                    protected_goals=copy.deepcopy(memory['protected_goals']),
                    holding=bool(snap.get('held_object')), held_label=self._held_label(),
                    history=copy.deepcopy(self.ctl._skill_task['history'][-MAX_EXECUTED:]),
                    pending_obligations=list(memory['obligations']),
                    capabilities=copy.deepcopy(capabilities), planning_budget=budget,
                    last_action_failure=copy.deepcopy(memory.get('last_action_failure', {})),
                    failure=copy.deepcopy(failure), max_steps=MAX_SKILL_STEPS,
                    remaining_steps=min(MAX_SKILL_STEPS, remaining))

    def _request(self, instruction, context, number):
        ctl = self.ctl
        with ctl._lock:
            ctl._next_id += 1
            reply = dict(id=ctl._next_id, role='assistant', kind='skill_plan',
                         attempt=number, content='', status='streaming')
            ctl._messages.append(reply)
            ctl._messages = ctl._messages[-60:]
            ctl._metrics['model_requests'] += 1
        ctl._set_status('thinking', '模型正在组合基本技能，回复实时显示')

        def on_delta(text):
            if not ctl._cancel.is_set():
                with ctl._lock:
                    reply['content'] += text

        plan = ctl._client.plan_skills(instruction, context, on_delta,
                                       cancel_event=ctl._cancel)
        self._check_stop()
        plan = validate_skill_plan(plan, context)
        with ctl._lock:
            reply.update(status='complete', decision=copy.deepcopy(plan))
            ctl._last_usage = copy.deepcopy(getattr(ctl._client, 'last_usage', {}))
        return plan

    @staticmethod
    def _final_destination(step, goals):
        dest = step.get('destination') or {}
        own = next((g for g in goals if g['label'] == step.get('label')), None)
        if own is None:
            return False
        if dest.get('type') == 'goal':
            return goals[dest['index']]['label'] == step.get('label')
        if dest.get('type') == 'zone':
            return own.get('zone_name') == dest['name']
        if dest.get('type') == 'region':
            return own.get('destination') == dest
        return False

    def _risk(self, steps, goals):
        labels = {g['label'] for g in goals}
        return any((s['skill'] == 'pick' and s['label'] not in labels)
                   or (s['skill'] in ('move', 'place') and
                       not self._final_destination(s, goals)) for s in steps)

    @staticmethod
    def _describe(step):
        text = '%s%s' % (SKILLS[step['skill']],
                        '：'+NAMES.get(step['label'], step['label']) if step.get('label') else '')
        if step['skill'] == 'observe' and step.get('mode') == 'active':
            text += '（经安全检查调整视角）'
        dest = step.get('destination') or {}
        if dest.get('type') == 'goal':
            text += ' → 原定目标 %d' % (dest['index']+1)
        elif dest.get('type') == 'original':
            text += ' → 任务开始时的原位置与朝向'
        elif dest.get('type') == 'auto_free':
            text += ' → 本地检查可放下的桌面空位'
        elif dest.get('type') in ('zone', 'region'):
            text += ' → '+{'zone_red': '红色区域', 'zone_green': '绿色区域',
                           'zone_yellow': '黄色区域', 'pick_area': '抓取区'}.get(dest['name'], dest['name'])
        if step['skill'] == 'ask':
            text += '：'+step['question']
        return text

    def _approve(self, steps, question):
        self._check_stop()
        version = self.session.snapshot()['scene_version']
        context = dict(scene_version=version, options=[dict(
            id='composed_skill_plan', requires_confirmation=True,
            question=question, description='仅执行列出的技能；拒绝保留现场。',
            steps=[self._describe(s) for s in steps])])
        self._publish(status='awaiting_approval')
        accepted = self.ctl._await_recovery_approval(context, dict(option_id='composed_skill_plan'))
        self._check_stop()
        return accepted

    def _feedback(self, step, result, number):
        entry = dict(id=step['id'], plan=number, skill=step['skill'],
                     label=step.get('label'), destination=copy.deepcopy(step.get('destination')),
                     ok=bool(result.get('ok')), stage=result.get('stage', ''),
                     error=result.get('error', ''), message=result.get('message', ''),
                     holding=bool(self.session.snapshot().get('held_object')))
        for name in ('diagnostics', 'observation_mode', 'reason'):
            if isinstance(result.get(name), str):
                entry[name] = result[name][:1000]
        for name in ('changed', 'progress'):
            if type(result.get(name)) is bool:
                entry[name] = result[name]
        with self.ctl._lock:
            self.ctl._skill_task['history'].append(entry)
            self.ctl._last_result = copy.deepcopy(result)
            self.ctl._skill_task['executed'] += 1
        self.ctl._event('skill_feedback', dict(message='%s：%s' %
            (self._describe(step), '完成' if result.get('ok') else result.get('message') or '未完成'),
            feedback=entry))

    def _obligations(self, step, result, memory, steps):
        if step['skill'] not in ('pick', 'place', 'move'):
            return
        label = step['label']
        if label in {g['label'] for g in memory['goals']}:
            return
        # An explicit later original reference is a promise, not a loose hint.
        promised = any(s.get('label') == label and
                       (s.get('destination') or {}).get('type') == 'original' for s in steps)
        if promised:
            memory['obligations'][label] = 'pending'
            if label not in memory['restorations']:
                memory['restorations'].append(label)
        if (result.get('ok') and step['skill'] in ('move', 'place')
                and (step.get('destination') or {}).get('type') == 'original'):
            memory['obligations'].pop(label, None)

    def _verify_all(self, executor, memory, cached=None):
        goals = copy.deepcopy(memory['goals']+memory['protected_goals'])
        for label in memory['restorations']:
            originals = [o for o in memory['originals'] if o['label'] == label]
            if len(originals) != 1:
                return dict(ok=False, error='original_not_measured', message='缺少承诺复位物品的原始测量')
            goals.append(dict(label=label, zone_name=None,
                destination=dict(type='point', xy=list(originals[0]['center_base'][:2]))))
        result = (cached if self._verification_current(cached) and not memory['protected_goals'] and not memory['restorations'] else
            executor.execute(dict(id='final-check', skill='verify'), goals, cancel_event=self.ctl._cancel))
        if not self._verification_current(result):
            return dict(result, ok=False, error='scene_changed', message='最终核验期间现场变化，结果已过期')
        for label in memory['restorations']:
            original = next(o for o in memory['originals'] if o['label'] == label)
            measured = [o for o in self.session.scene.get('objects', [])
                        if o.get('valid') and o.get('label') == label]
            # A square's observed principal axis is only defined modulo 90°.
            # Retain the full-pose requirement for asymmetric footprints.
            from .catalog import catalog_entries
            entry = next((e for e in catalog_entries() if e['label'] == label), {})
            dims = entry.get('dimensions', [])
            square = (entry.get('shape') in ('box', 'cube') and len(dims) >= 2
                      and abs(dims[0]-dims[1]) < .001)
            period = math.pi/2 if square else math.pi
            if (len(measured) != 1 or measured[0].get('yaw') is None or original.get('yaw') is None
                    or abs(math.remainder(measured[0]['yaw']-original['yaw'], period)) > math.radians(10)):
                result = dict(result, ok=False, error='restoration_yaw_unconfirmed', message='原位复位朝向未通过相机核验')
                break
        if result.get('ok'):
            memory['obligations'].clear()
        return result

    def _finish(self, ok, message, memory, result=None):
        # Publish success atomically with respect to application/queuing of edits.
        with getattr(self.session, '_operation_lock', nullcontext()):
            with getattr(self.session, '_intervention_lock', nullcontext()):
                if ok and not self._verification_current(result):
                    ok, message = False, '最终核验后现场发生变化，旧结果不能计作完成'
                return self._finish_locked(ok, message, memory, result)

    def _finish_locked(self, ok, message, memory, result=None):
        ctl, snap = self.ctl, self.session.snapshot()
        pending = dict(memory.get('obligations', {}))
        held = bool(snap.get('held_object'))
        ok = bool(ok and not held and not pending and not ctl._cancel.is_set()
                  and result and result.get('camera_verified') is True
                  and result.get('truth', {}).get('success') is True)
        if not ok and message.startswith('全部原始目标'):
            message = '最终相机、几何或复位核验未全部通过，已保留现场'
        result = dict(result or {}, ok=ok, holding=held, skill_task=True,
                      goals=copy.deepcopy(memory['goals']),
                      pending_obligations=pending,
                      message=message if not pending else message+'；仍有物品尚未按承诺复位')
        with ctl._lock:
            ctl._last_result = result
            ctl._scene = self.session.scene
            ctl._needs_reset = not ok and bool(memory['goals'])
            ctl._status = 'complete' if ok else 'stopped' if ctl._cancel.is_set() else 'error'
            ctl._stage = result['message']
            ctl._error = '' if ok else result['message']
            ctl._skill_task.update(status='complete' if ok else 'stopped' if ctl._cancel.is_set() else 'failed',
                                   message=result['message'], current_index=None)
            if ok:
                ctl._metrics['successes'] += 1
            if ctl._batch.get('total'):
                ctl._batch.update(status='complete' if ok else 'failed',
                                  completed=ctl._batch['total'] if ok else ctl._batch['completed'],
                                  message=result['message'])
                if ok:
                    for item in ctl._batch['items']:
                        item.update(status='succeeded', message='已通过最终核验')
        ctl._event('result', dict(message=result['message'], result=result))
        return result

    def run(self, instruction, *, goals=None, protected_goals=(), failure=None,
            selected_destination=None, resume=False):
        ctl = self.ctl
        previous = getattr(ctl, '_skill_memory', None)
        if resume and previous and previous.get('goals'):
            memory = previous
        else:
            scene = self.session.scene
            originals = [copy.deepcopy(o) for o in scene.get('objects', []) if o.get('valid')]
            memory = dict(instruction=instruction, goals=copy.deepcopy(goals or []),
                          protected_goals=copy.deepcopy(list(protected_goals)),
                          originals=originals, obligations={}, restorations=[],
                          selected_destination=copy.deepcopy(selected_destination))
        memory.setdefault('restorations', [])
        initial_failure = failure if isinstance(failure, dict) else {}
        action_failed = initial_failure.get('ok') is not True and (
            initial_failure.get('skill') in ('pick', 'place', 'move')
            or initial_failure.get('stage') in ('pick', 'place', 'move', 'inspect_place',
                                               'release', 'transit')
            or initial_failure.get('error') in ('placement_changed', 'placement_infeasible',
                                               'placement_planning_budget_exceeded')
            or (initial_failure.get('holding') is True and bool(initial_failure.get('diagnostics'))))
        memory.setdefault('last_action_failure', copy.deepcopy(initial_failure) if action_failed else {})
        memory['stagnation'] = 0
        ctl._skill_memory = memory
        ctl._skill_task = idle_skill_task()
        self._publish(goals=memory['goals'],
                      max_plans=plan_limit(memory['goals'], memory.get('initial_steps', ())))
        executor = self.executor_factory(self.session, originals=memory['originals'])
        try:
            self._check_stop()
            if not self.session.scene.get('objects') and not self.session.snapshot().get('held_object'):
                self.session.observe()
                memory['originals'] = [copy.deepcopy(o) for o in self.session.scene.get('objects', []) if o.get('valid')]
                executor.originals = copy.deepcopy(memory['originals'])
            number = 0
            while number < ctl._skill_task['max_plans']:
                number += 1
                self._check_stop()
                remaining = MAX_EXECUTED-ctl._skill_task['executed']
                if remaining <= 0:
                    break
                self._publish(status='thinking' if number == 1 else 'replanning', plan_number=number)
                context, revision = self._context(memory, failure, executor), self._revision()
                plan = self._request(instruction, context, number)
                if revision != self._revision() or self._pending_interventions():
                    failure = dict(ok=False, stage='plan', error='scene_changed', message='模型规划期间现场发生变化，旧计划未执行')
                    executor.execute(dict(id='refresh', skill='observe'), memory['goals'], cancel_event=ctl._cancel)
                    continue
                if plan['action'] != 'plan':
                    if not memory['goals'] and plan['action'] in ('answer', 'clarify'):
                        self._publish(status='idle', message=plan['message'])
                        ctl._set_status('ready', plan['message'])
                        return dict(ok=True, readonly=True, message=plan['message'])
                    return self._finish(False, plan['message'], memory)
                if not memory['goals']:
                    memory['goals'] = copy.deepcopy(plan['goals'])
                steps = copy.deepcopy(plan['steps'])
                memory.setdefault('initial_steps', copy.deepcopy(steps))
                for step in steps:
                    step.update(status='pending', message='等待执行')
                self._publish(goals=memory['goals'], steps=steps, current_index=None,
                              max_plans=plan_limit(memory['goals'], memory['initial_steps']))
                ctl._event('skill_plan', dict(message=plan['message'], steps=steps, plan_number=number))
                if self._risk(steps, memory['goals']):
                    question = ('此方案会暂放目标或移动其他物品。列出原位复位步骤的物品会要求复位；'
                                '其余调整可能保留改变。是否同意以下完整操作？')
                    if not self._approve(steps, question):
                        return self._finish(False, '未获同意，已保留现场并停止本次方案', memory)
                verified = None
                finished = False
                round_progress = False
                for index, step in enumerate(steps):
                    self._check_stop()
                    if revision != self._revision() or self._pending_interventions():
                        failure = dict(ok=False, error='scene_changed', stage='before_skill',
                                       message='人为干预改变现场，旧计划取消，重新规划')
                        break
                    self._publish(status='executing', current_index=index)
                    step.update(status='executing', message='正在执行')
                    self._publish(steps=steps)
                    ctl._set_status('verifying' if step['skill'] in ('verify', 'finish') else 'executing',
                                    '技能 %d/%d：%s' % (index+1, len(steps), self._describe(step)))
                    ctl._event('skill_step', dict(message=self._describe(step), step=step))
                    if revision != self._revision() or self._pending_interventions():
                        failure = dict(ok=False, error='scene_changed', stage='before_dispatch',
                                       message='技能开始前现场变化，旧方案已取消')
                        break
                    # Register physical obligations before a skill can move.
                    # Stop/driver exceptions must not erase an unpaid restore.
                    self._obligations(step, dict(ok=False), memory, steps)
                    held_before = bool(self.session.snapshot().get('held_object'))
                    if step['skill'] == 'ask':
                        accepted = self._approve(steps[index:], step['question'])
                        result = dict(ok=accepted, message='已获同意' if accepted else '人工拒绝，未继续执行')
                    elif step['skill'] == 'finish':
                        result = self._verify_all(executor, memory, verified)
                        finished = bool(result.get('ok'))
                    else:
                        result = executor.execute(step, memory['goals'],
                            protected_goals=memory['protected_goals'], cancel_event=ctl._cancel,
                            expected_revision=revision)
                    self._feedback(step, result, number)
                    physical = step['skill'] in ('pick', 'place', 'move')
                    if physical and not result.get('ok'):
                        memory['last_action_failure'] = copy.deepcopy(result)
                    if ((physical and result.get('ok'))
                            or (step['skill'] == 'observe' and result.get('changed') is True)
                            or held_before != bool(self.session.snapshot().get('held_object'))):
                        round_progress = True
                    self._obligations(step, result, memory, steps)
                    step.update(status='succeeded' if result.get('ok') else 'failed', message=result.get('message', ''))
                    self._publish(steps=steps)
                    self._check_stop()
                    if step['skill'] in ('pick', 'place', 'move'):
                        verified = None
                    elif step['skill'] == 'verify' and 'goal_index' not in step and not memory['protected_goals']:
                        verified = result if result.get('ok') else None
                    if not result.get('ok'):
                        failure = result
                        if step['skill'] == 'ask' or result.get('aborted') or result.get('fault'):
                            return self._finish(False, result.get('message') or '技能执行未完成，已保留现场', memory, result)
                        break
                    if finished:
                        if memory['obligations']:
                            failure = dict(ok=False, error='restoration_pending', message='尚有承诺复位的物品，不能结束任务')
                            break
                        return self._finish(True, '全部原始目标和复位义务已通过最终核验', memory, result)
                else:
                    if verified and not memory['obligations'] and not self.session.snapshot().get('held_object'):
                        result = self._verify_all(executor, memory, verified)
                        if result.get('ok'):
                            return self._finish(True, '全部原始目标已通过核验', memory, result)
                        failure = result
                    else:
                        failure = dict(ok=False, stage='partial_plan', error='goal_not_finished',
                                       message='局部技能已执行，原始目标尚需继续规划')
                for step in steps:
                    if step['status'] == 'pending':
                        step.update(status='cancelled', message='旧计划停止，等待新计划')
                self._publish(steps=steps)
                memory['stagnation'] = 0 if round_progress else memory['stagnation']+1
                self._publish(stagnation=memory['stagnation'])
                if memory['stagnation'] >= MAX_STAGNATION:
                    message = '连续三轮没有新的执行进展或观测信息，已保留持物和未完成事项'
                    return self._finish(False, message, memory,
                        dict(failure or {}, error='planning_stagnation', message=message))
                if failure.get('error') == 'scene_changed' and not self.session.snapshot().get('held_object'):
                    label = memory['goals'][0]['label'] if memory['goals'] else None
                    executor.execute(dict(id='refresh', skill='observe', **({'label': label} if label else {})), memory['goals'], cancel_event=ctl._cancel)
                ctl._event('skill_replan', dict(message='收到逐步反馈，重新规划剩余技能', failure=failure))
            return self._finish(False, '已达到本次技能或重规划预算，保留现场及未完成事项', memory, failure)
        except Exception as exc:
            stopped = ctl._cancel.is_set() or self.session.cancel_event.is_set()
            message = '任务已停止，保留当前持物和未完成事项' if stopped else (
                str(exc) if isinstance(exc, (ValueError, RuntimeError)) else '技能规划失败（%s）' % type(exc).__name__)
            return self._finish(False, message, memory, dict(aborted=stopped, error='skill_task_error'))

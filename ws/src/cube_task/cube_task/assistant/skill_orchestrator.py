"""Model-composed skills with immutable goals, explicit approval and feedback.

Only the dashboard's worker calls this module. It supplies no robot geometry and
never substitutes a model's completion message for measured final verification.
"""
from __future__ import annotations

import copy
import math
from contextlib import nullcontext

from .skill_protocol import MAX_SKILL_STEPS, placement_checks_summary, validate_skill_plan

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
                goals=[], steps=[], history=[], message='', goal_worklist=[], final_audit=None)


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

    def _reusable(self, result):
        """A revision alone is insufficient: robot moves do not increment it."""
        snap = self.session.snapshot()
        return bool(result and result.get('ok') is True
                    and result.get('camera_verified') is True
                    and result.get('truth', {}).get('success') is True
                    and result.get('verified_revision') is not None
                    and self._verification_current(result)
                    and not result.get('holding') and not snap.get('held_object')
                    and not getattr(self.session.runner, 'has_held_recovery', False)
                    and not result.get('fault') and not result.get('aborted')
                    and not self.ctl._cancel.is_set()
                    and not self.session.cancel_event.is_set())

    @staticmethod
    def _goal_contract(goal):
        if goal.get('destination') is not None:
            return dict(label=goal['label'], zone_name=None,
                        destination=copy.deepcopy(goal['destination']))
        return dict(label=goal['label'], zone_name=goal.get('zone_name'))

    def _proof(self, result, goals):
        if not self._reusable(result):
            return None
        contracts = [self._goal_contract(g) for g in goals]
        items = result.get('items', ())
        if len(items) != len(contracts):
            return None
        if any(item.get('ok') is not True
               or item.get('truth', {}).get('success') is not True
               or item.get('verified_revision', result['verified_revision']) != result['verified_revision']
               for item in items):
            return None
        measured = [self._goal_contract(i.get('goal', {})) for i in items]
        # Camera verification groups shared destinations, so its item order
        # can differ from the user's goal order. Require one proof per contract.
        if any(measured.count(goal) != 1 for goal in contracts):
            return None
        ordered = [copy.deepcopy(items[measured.index(goal)]) for goal in contracts]
        return dict(goals=contracts, result=dict(copy.deepcopy(result), items=ordered))

    def _action_proof(self, step, result, goals):
        if (step['skill'] not in ('move', 'place') or not self._reusable(result)
                or result.get('controller_ok') is not True
                or result.get('label') != step.get('label')
                or not self._final_destination(step, goals)):
            return None
        goal = next(g for g in goals if g['label'] == step['label'])
        contract = self._goal_contract(goal)
        destination = (goal.get('destination') or
                       dict(type='named_region', name=goal.get('zone_name')))
        if result.get('resolved_destination') != destination:
            return None
        item = dict(goal=contract, label=step['label'], ok=True, camera_verified=True,
                    truth={'success': True}, verified_revision=result['verified_revision'])
        return self._proof(dict(result, items=[item]), [goal])

    def _stack_support(self, goal):
        from .stacking import audit
        objects = [obj for obj in getattr(getattr(self.session, 'world', None), 'objects', ())
                   if obj.label == goal['label']]
        if len(objects) != 1:
            return dict(success=False, reason='support_identity_unconfirmed')
        return audit(self.session, objects[0], goal['destination'])

    def _cached_verification(self, proof, goals):
        # Reuse unchanged camera evidence, but recheck physical support:
        # contact or stability can fail below the pose-revision threshold.
        if not proof or not self._reusable(proof.get('result')):
            return None
        contracts = [self._goal_contract(g) for g in goals]
        if not contracts or any(g not in proof['goals'] for g in contracts):
            return None
        items = [copy.deepcopy(proof['result']['items'][proof['goals'].index(g)])
                 for g in contracts]
        for goal, item in zip(contracts, items):
            destination = goal.get('destination') or {}
            if destination.get('type') != 'on_object':
                continue
            support = self._stack_support(goal)
            if support.get('success') is not True:
                return None
            item['truth']['support'] = support
        result = dict(copy.deepcopy(proof['result']), items=items, skill='verify',
                      verification_reused=True, message='现场未变化，复用已通过的相机与落位核验')
        if len(items) == 1 and 'support' in items[0]['truth']:
            result['truth']['support'] = copy.deepcopy(items[0]['truth']['support'])
        return result

    def _paired_place(self, steps, index, goals):
        """Only an explicit final-place postcondition can make a pick redundant."""
        for offset in range(index + 1, len(steps)):
            following = steps[offset]
            if following['skill'] == 'observe':
                continue
            if (following['skill'] == 'place'
                    and following.get('label') == steps[index].get('label')
                    and self._final_destination(following, goals)):
                return offset
            break
        return None

    def _held_label(self):
        if not self.session.snapshot().get('held_object'):
            return None
        context = getattr(self.session.runner, '_held_recovery_context', None)
        if context and getattr(context.get('det'), 'label', None):
            return context['det'].label
        return None

    def _bind_task_contract(self, memory):
        """Freeze the requested scope independently of camera actionability."""
        from .task_contract import bind_task_contract
        inventory = memory.get('task_inventory')
        if inventory is None:
            reader = getattr(self.session, 'task_inventory', None)
            if callable(reader):
                inventory = reader()
            else:
                # Compatibility sessions have observations but no scene editor.
                # Deduplicate semantic fragments; never use all configured classes.
                labels = dict.fromkeys(o.get('label') for o in self.session.scene.get('objects', [])
                                       if isinstance(o.get('label'), str))
                inventory = [dict(label=label, count=1) for label in labels]
            # Old task memory predates manifest snapshots. A task-start camera
            # identity remains requested even if the user later removes it.
            inventory = copy.deepcopy(inventory)
            known = {item if isinstance(item, str) else item.get('label') for item in inventory}
            for original in memory.get('originals', []):
                label = original.get('label')
                if original.get('valid') and isinstance(label, str) and label not in known:
                    inventory.append(dict(label=label, count=1))
                    known.add(label)
        contract = bind_task_contract(memory.get('instruction', ''), self.session.scene,
                                      inventory, memory['goals'])
        if contract is None and memory.get('task_contract') is not None:
            raise ValueError('已保存的全体任务范围无法与原指令对应，不能缩减目标。')
        if contract is not None:
            memory['task_inventory'] = [dict(label=label, count=1)
                                        for label in contract['required_labels']]
            memory['task_contract'] = contract
            memory['goals'] = copy.deepcopy(contract['goals'])
        return contract

    def _context(self, memory, failure, executor=None):
        from .goal_readiness import goal_worklist
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
        context = dict(scene=scene, goals=copy.deepcopy(memory['goals']),
                    conversation=copy.deepcopy(memory.get('conversation', [])),
                    originals=copy.deepcopy(memory['originals']),
                    protected_goals=copy.deepcopy(memory['protected_goals']),
                    holding=bool(snap.get('held_object')), held_label=self._held_label(),
                    history=copy.deepcopy(self.ctl._skill_task['history'][-MAX_EXECUTED:]),
                    pending_obligations=list(memory['obligations']),
                    capabilities=copy.deepcopy(capabilities), planning_budget=budget,
                    last_action_failure=copy.deepcopy(memory.get('last_action_failure', {})),
                    failure=copy.deepcopy(failure), max_steps=MAX_SKILL_STEPS,
                    remaining_steps=min(MAX_SKILL_STEPS, remaining))
        if memory.get('task_contract') is not None:
            context.update(task_contract=copy.deepcopy(memory['task_contract']),
                           task_inventory=copy.deepcopy(memory['task_inventory']))
        context['goal_worklist'] = goal_worklist(scene, memory['goals'])
        audit = copy.deepcopy(memory.get('final_audit')) if self._audit_current(memory) else None
        if audit is not None:
            context['final_audit'] = audit
        self._publish(goal_worklist=context['goal_worklist'], final_audit=audit)
        return context

    def _audit_current(self, memory):
        return bool(memory.get('final_audit')
                    and memory.get('_audit_stamp') == (self._revision(), memory.get('_audit_epoch', 0))
                    and not self._pending_interventions())

    def _proof_updates_audit(self, memory, proof):
        """A new confirmed goal can supersede an earlier unconfirmed review."""
        if not proof or not self._audit_current(memory):
            return False
        if any(item.get('camera_verified') is not True for item in proof['result']['items']):
            return False
        goals = [self._goal_contract(goal) for goal in memory['goals']]
        for contract in proof['goals']:
            if contract not in goals:
                continue
            index = goals.index(contract)
            previous = next((item for item in memory['final_audit']['items']
                             if item['goal_index'] == index and item['label'] == contract['label']), {})
            if previous.get('ok') is not True or previous.get('camera_verified') is not True:
                return True
        return False

    def _verification_updates_audit(self, memory, result, selected):
        """A fresh explicit recheck can confirm or revoke a reviewed item."""
        if (not self._audit_current(memory) or not self._verification_current(result)
                or result.get('fault') or result.get('aborted') or result.get('holding')):
            return False
        contracts = [self._goal_contract(goal) for goal in memory['goals']]
        for goal in selected:
            contract = self._goal_contract(goal)
            if contract not in contracts:
                continue
            index = contracts.index(contract)
            previous = next((item for item in memory['final_audit']['items']
                             if item['goal_index'] == index and item['label'] == goal['label']), {})
            matches = [item for item in result.get('items', ()) if isinstance(item.get('goal'), dict)
                       and item['goal'].get('label') and self._goal_contract(item['goal']) == contract]
            item = matches[0] if len(matches) == 1 else {}
            evidence = dict(result, items=[item], ok=item.get('ok') is True,
                            camera_verified=item.get('camera_verified') is True,
                            truth=copy.deepcopy(item.get('truth', {})))
            confirmed = self._proof(evidence, [goal]) is not None
            # A missing/cropped recheck revokes old positive evidence; it does
            # not assert that an unseen object physically failed its goal.
            if (previous.get('ok') is True) != confirmed:
                return True
        return False

    @staticmethod
    def _invalidate_audit(memory):
        memory['_audit_epoch'] = memory.get('_audit_epoch', 0)+1
        memory.pop('_audit_stamp', None)
        memory.pop('_audit_result', None)

    def _can_audit(self, memory, failure):
        snap = self.session.snapshot()
        if (not memory['goals'] or self.ctl._cancel.is_set() or self.session.cancel_event.is_set()
                or snap.get('held_object') or snap.get('status') == 'stopped'
                or getattr(self.session.runner, 'has_held_recovery', False)
                or self._pending_interventions()
                or any(r.get('fault') or r.get('aborted') for r in
                       (failure or {}, memory.get('last_action_failure', {})))):
            return False
        health = getattr(getattr(self.session, 'arm', None), 'health', None)
        return not callable(health) or bool(health().ok)

    def _end_audit(self, executor, memory, failure, result=None, expected_revision=None):
        """One measured review per unchanged scene, with no model or grasp call."""
        if not self._can_audit(memory, failure):
            return None, False
        if self._audit_current(memory):
            return copy.deepcopy(memory['_audit_result']), False
        attempt_revision = self._revision() if expected_revision is None else expected_revision
        self.ctl._set_status('verifying', '正在审核全部原始目标，暂缓项仍保留在任务中')
        result = self._verify_all(executor, memory) if result is None else result
        self._check_stop()
        if result.get('fault') or result.get('aborted'):
            memory['last_action_failure'] = copy.deepcopy(result)
            raise RuntimeError(result.get('message') or '审核未完成，已保留现场')
        if result.get('ok') and self._proof(result, self._final_goals(memory)) is None:
            result = dict(result, ok=False, error='task_scope_incomplete', message='完整审核缺少部分原目标的证据')
        if not result.get('ok') and not result.get('error'):
            result = dict(result, error='goal_not_finished')
        current = self._verification_current(result)
        items = []
        for index, goal in enumerate(memory['goals']):
            matches = [item for item in result.get('items', ())
                       if isinstance(item.get('goal'), dict) and item['goal'].get('label')
                       and self._goal_contract(item['goal']) == self._goal_contract(goal)]
            item = matches[0] if len(matches) == 1 else {}
            camera = item.get('camera_verified') is True
            ok = bool(current and camera and item.get('ok') is True
                      and item.get('truth', {}).get('success') is True)
            items.append(dict(goal_index=index, label=goal['label'], ok=ok,
                camera_verified=bool(current and camera is True),
                error='' if ok else str(item.get('error') or result.get('error') or 'goal_unconfirmed')[:200],
                message=str(item.get('message') or ('目标已核验' if ok else '该目标尚未通过完整核验'))[:500]))
        audit = dict(phase='reviewed', scene_version=self.session.snapshot().get('scene_version'), items=items)
        memory['final_audit'] = audit
        memory['_audit_result'] = copy.deepcopy(result)
        unchanged = attempt_revision == self._revision() and not self._pending_interventions()
        # An inconclusive attempt is still one review, never a success proof.
        memory['_audit_stamp'] = (attempt_revision, memory.get('_audit_epoch', 0)) if unchanged else None
        from .goal_readiness import goal_worklist
        self._publish(final_audit=audit, goal_worklist=goal_worklist(self.session.scene, memory['goals']))
        self.ctl._event('final_audit', dict(message='全部原始目标已审核，未确认项交由模型继续决策',
                                           final_audit=audit))
        return result, True

    def _is_deferred(self, step, result, goals):
        return bool(result and result.get('deferred') is True
                    and result.get('manipulation_started') is False
                    and result.get('holding') is False
                    and not result.get('fault') and not result.get('aborted')
                    and step['skill'] == 'move' and self._final_destination(step, goals)
                    and not self.session.snapshot().get('held_object'))

    def _defer_step(self, step, result, memory, steps, number):
        if not self._is_deferred(step, result, memory['goals']):
            return False
        self._feedback(step, result, number)
        step.update(status='deferred', message=result.get('message', '暂缓未可靠测量的目标'))
        self._publish(steps=steps)
        self._check_stop()
        return True

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

        try:
            plan = ctl._client.plan_skills(instruction, context, on_delta,
                                           cancel_event=ctl._cancel)
            self._check_stop()
            plan = validate_skill_plan(plan, context)
        except ValueError as exc:
            with ctl._lock:
                reply.update(status='error', error=str(exc)[:1000])
                ctl._last_usage = copy.deepcopy(getattr(ctl._client, 'last_usage', {}))
            raise
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
        if dest.get('type') in ('region', 'on_object'):
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
        elif dest.get('type') == 'on_object':
            text += ' → '+NAMES.get(dest['label'], dest['label'])+'顶面'
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
        for name in ('detail', 'diagnostics', 'observation_mode', 'reason'):
            if isinstance(result.get(name), str):
                entry[name] = result[name][:1000]
        for name in ('changed', 'progress', 'skipped', 'verification_reused', 'deferred', 'manipulation_started'):
            if type(result.get(name)) is bool:
                entry[name] = result[name]
        checks = placement_checks_summary(result.get('placement_checks'))
        if checks is not None:
            entry['placement_checks'] = checks
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

    def _final_goals(self, memory):
        """One contract list for both measurement and success publication."""
        goals = copy.deepcopy(memory['goals']+memory['protected_goals'])
        for label in memory['restorations']:
            originals = [o for o in memory['originals'] if o['label'] == label]
            if len(originals) != 1:
                raise ValueError('缺少承诺复位物品的原始测量')
            goals.append(dict(label=label, zone_name=None,
                destination=dict(type='point', xy=list(originals[0]['center_base'][:2]))))
        return goals

    def _verify_all(self, executor, memory, cached=None):
        try:
            contract = self._bind_task_contract(memory)
        except ValueError as exc:
            return dict(ok=False, error='task_scope_unconfirmed', message=str(exc))
        try:
            goals = self._final_goals(memory)
        except ValueError as exc:
            return dict(ok=False, error='original_not_measured', message=str(exc))
        result = (self._cached_verification(cached, goals)
                  if not memory['protected_goals'] and not memory['restorations'] else None)
        if result is None:
            result = executor.execute(dict(id='final-check', skill='verify'), goals, cancel_event=self.ctl._cancel)
        if not self._verification_current(result):
            return dict(result, ok=False, error='scene_changed', message='最终核验期间现场变化，结果已过期')
        if contract is not None and result.get('ok') and self._proof(result, goals) is None:
            return dict(result, ok=False, error='task_scope_incomplete',
                        message='最终核验没有覆盖全部请求物品，不能以部分目标宣告完成')
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
                if ok:
                    try:
                        contract = self._bind_task_contract(memory)
                        expected = self._final_goals(memory)
                    except ValueError as exc:
                        ok, message = False, str(exc)
                        result = dict(result or {}, error='task_scope_unconfirmed')
                    else:
                        if contract is not None and self._proof(result, expected) is None:
                            ok, message = False, '仍有请求物品未通过最终核验，不能宣告全部完成'
                            result = dict(result or {}, error='task_scope_incomplete')
                    with self.ctl._lock:
                        self.ctl._skill_task['goals'] = copy.deepcopy(memory['goals'])
                if ok and not self._verification_current(result):
                    ok, message = False, '最终核验后现场发生变化，旧结果不能计作完成'
                if ok:
                    for goal in memory['goals'] + memory['protected_goals']:
                        if (goal.get('destination') or {}).get('type') != 'on_object':
                            continue
                        support = self._stack_support(goal)
                        if support.get('success') is True:
                            continue
                        # Idle physics may run between verification and this
                        # publication lock without crossing the pose threshold.
                        ok, message = False, '最终核验后叠放支撑或稳定性变化，不能计作完成'
                        result = copy.deepcopy(result or {})
                        result.update(error='stack_support_changed',
                            truth=dict(result.get('truth') or {}, success=False, support=support))
                        for item in result.get('items', []):
                            if self._goal_contract(item['goal']) == self._goal_contract(goal):
                                item.update(ok=False, truth=dict(item.get('truth') or {},
                                    success=False, support=copy.deepcopy(support)))
                        break
                return self._finish_locked(ok, message, memory, result)

    def _finish_locked(self, ok, message, memory, result=None):
        from .goal_readiness import goal_worklist
        ctl, snap = self.ctl, self.session.snapshot()
        worklist = goal_worklist(self.session.scene, memory['goals'])
        audit = copy.deepcopy(memory.get('final_audit')) if self._audit_current(memory) else None
        pending = dict(memory.get('obligations', {}))
        held = bool(snap.get('held_object'))
        ok = bool(ok and not held and not pending and not ctl._cancel.is_set()
                  and result and result.get('camera_verified') is True
                  and result.get('truth', {}).get('success') is True)
        if not ok and message.startswith('全部原始目标'):
            message = '最终相机、几何或复位核验未全部通过，已保留现场'
        previous_failure = memory.get('last_action_failure', {})
        if (not ok and previous_failure.get('fault')
                and not previous_failure.get('aborted')):
            # A model error or declined plan cannot clear an earlier arm fault.
            result = dict(result or {}, fault=True)
        if ok:
            memory['last_action_failure'] = {}
        result = dict(result or {}, ok=ok, holding=held, skill_task=True,
                      goals=copy.deepcopy(memory['goals']),
                      pending_obligations=pending,
                      goal_worklist=worklist, final_audit=audit,
                      message=message if not pending else message+'；仍有物品尚未按承诺复位')
        with ctl._lock:
            ctl._last_result = result
            ctl._scene = self.session.scene
            # An unmet goal is not a controller fault. Keep failed task memory
            # resumable, but allow the user to adjust an unheld, healthy scene.
            ctl._needs_reset = bool(not ok and (
                held or result.get('fault') or result.get('aborted')
                or snap.get('status') == 'stopped'))
            ctl._status = 'complete' if ok else 'stopped' if ctl._cancel.is_set() else 'error'
            ctl._stage = result['message']
            ctl._error = '' if ok else result['message']
            ctl._skill_task.update(status='complete' if ok else 'stopped' if ctl._cancel.is_set() else 'failed',
                                   message=result['message'], current_index=None,
                                   goal_worklist=worklist, final_audit=audit)
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
            selected_destination=None, resume=False, conversation=()):
        ctl = self.ctl
        previous = getattr(ctl, '_skill_memory', None)
        resuming = bool(resume and previous and previous.get('goals'))
        if resuming:
            memory = previous
        else:
            scene = self.session.scene
            originals = [copy.deepcopy(o) for o in scene.get('objects', []) if o.get('valid')]
            memory = dict(instruction=instruction, goals=copy.deepcopy(goals or []),
                          conversation=copy.deepcopy(list(conversation)),
                          protected_goals=copy.deepcopy(list(protected_goals)),
                          originals=originals, obligations={}, restorations=[],
                          selected_destination=copy.deepcopy(selected_destination))
        memory.setdefault('restorations', [])
        memory.setdefault('_audit_epoch', 0)
        if resuming:
            self._invalidate_audit(memory)
        initial_failure = failure if isinstance(failure, dict) else {}
        action_failed = initial_failure.get('ok') is not True and (
            initial_failure.get('fault')
            or initial_failure.get('skill') in ('pick', 'place', 'move')
            or initial_failure.get('stage') in ('pick', 'place', 'move', 'inspect_place',
                                               'release', 'transit')
            or initial_failure.get('error') in ('placement_changed', 'placement_infeasible',
                                               'placement_planning_budget_exceeded')
            or (initial_failure.get('holding') is True and bool(initial_failure.get('diagnostics'))))
        memory.setdefault('last_action_failure', copy.deepcopy(initial_failure) if action_failed else {})
        if initial_failure.get('fault') and not initial_failure.get('aborted'):
            memory['last_action_failure'] = copy.deepcopy(initial_failure)
        memory['stagnation'] = 0
        ctl._skill_memory = memory
        ctl._skill_task = idle_skill_task()
        self._publish(goals=memory['goals'],
                      max_plans=plan_limit(memory['goals'], memory.get('initial_steps', ())))
        executor = self.executor_factory(self.session, originals=memory['originals'])
        try:
            self._check_stop()
            self._bind_task_contract(memory)
            self._publish(goals=memory['goals'],
                          max_plans=plan_limit(memory['goals'], memory.get('initial_steps', ())))
            if not self.session.scene.get('objects') and not self.session.snapshot().get('held_object'):
                self.session.observe()
                if not resuming:
                    memory['originals'] = [copy.deepcopy(o) for o in self.session.scene.get('objects', []) if o.get('valid')]
                    executor.originals = copy.deepcopy(memory['originals'])
            number = 0
            while number < ctl._skill_task['max_plans']:
                # Reserve the last model request for a measured final review.
                if number == ctl._skill_task['max_plans']-1 and not self._audit_current(memory):
                    reviewed, _new = self._end_audit(executor, memory, failure)
                    if reviewed is not None:
                        if reviewed.get('ok'):
                            return self._finish(True, '全部原始目标和复位义务已通过最终核验', memory, reviewed)
                        failure = reviewed
                number += 1
                self._check_stop()
                remaining = MAX_EXECUTED-ctl._skill_task['executed']
                if remaining <= 0:
                    reviewed, _new = self._end_audit(executor, memory, failure)
                    if reviewed is None:
                        break
                    if reviewed.get('ok'):
                        return self._finish(True, '全部原始目标和复位义务已通过最终核验', memory, reviewed)
                    failure = reviewed
                self._publish(status='thinking' if number == 1 else 'replanning', plan_number=number)
                post_audit_decision = self._audit_current(memory)
                context, revision = self._context(memory, failure, executor), self._revision()
                try:
                    plan = self._request(instruction, context, number)
                except ValueError as exc:
                    self._check_stop()
                    # A rejected model plan has sent no robot commands. Keep
                    # the physical failure and immutable goals, and let the
                    # model correct its plan within the existing finite budget.
                    failure = dict(ok=False, stage='plan_validation', error='invalid_skill_plan',
                                   message='方案未通过执行前检查：'+str(exc)[:1000],
                                   holding=bool(self.session.snapshot().get('held_object')))
                    memory['stagnation'] += 1
                    self._publish(stagnation=memory['stagnation'])
                    ctl._event('skill_replan', dict(message=failure['message'], failure=failure))
                    if self._audit_current(memory):
                        return self._finish(False, failure['message'], memory, failure)
                    if memory['stagnation'] >= MAX_STAGNATION:
                        reviewed, new = self._end_audit(executor, memory, failure)
                        if reviewed is not None:
                            if reviewed.get('ok'):
                                return self._finish(True, '全部原始目标和复位义务已通过最终核验', memory, reviewed)
                            if new:
                                failure = reviewed
                                continue
                        return self._finish(False, '连续三轮方案未产生进展，已保留现场和未完成事项',
                                            memory, dict(failure, error='planning_stagnation'))
                    continue
                if revision != self._revision() or self._pending_interventions():
                    failure = dict(ok=False, stage='plan', error='scene_changed', message='模型规划期间现场发生变化，旧计划未执行')
                    executor.execute(dict(id='refresh', skill='observe'), memory['goals'], cancel_event=ctl._cancel)
                    continue
                if plan['action'] != 'plan':
                    if not memory['goals'] and plan['action'] in ('answer', 'clarify'):
                        self._publish(status='idle', message=plan['message'])
                        ctl._set_status('ready', plan['message'])
                        return dict(ok=True, readonly=True, message=plan['message'])
                    reviewed, new = self._end_audit(executor, memory, failure)
                    if reviewed is not None:
                        if reviewed.get('ok'):
                            return self._finish(True, '全部原始目标和复位义务已通过最终核验', memory, reviewed)
                        failure = reviewed
                        if new:
                            continue
                    return self._finish(False, plan['message'], memory, failure)
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
                skipped_pairs = set()
                finished = False
                round_progress = False
                queue_completed = False
                finish_failed = False
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
                    deferred = getattr(executor, 'defer_unobserved', None)
                    candidate = deferred(step, memory['goals']) if callable(deferred) else None
                    if self._defer_step(step, candidate, memory, steps, number):
                        continue
                    # Register physical obligations before a skill can move.
                    # Stop/driver exceptions must not erase an unpaid restore.
                    self._obligations(step, dict(ok=False), memory, steps)
                    held_before = bool(self.session.snapshot().get('held_object'))
                    proof_step = step
                    physical = step['skill'] in ('pick', 'place', 'move')
                    result = None
                    checker = getattr(executor, 'check_completed', None)
                    paired = (self._paired_place(steps, index, memory['goals'])
                              if step['skill'] == 'pick' else None)
                    if callable(checker) and (paired is not None or index in skipped_pairs):
                        placement = steps[paired] if paired is not None else step
                        result = checker(placement, memory['goals'], cancel_event=ctl._cancel,
                                         expected_revision=revision)
                        if result is not None and result.get('ok') and result.get('skipped'):
                            if paired is not None:
                                skipped_pairs.add(paired)
                            proof_step = placement
                            result = dict(result, skill=step['skill'])
                        elif index in skipped_pairs and result is None:
                            result = dict(ok=False, error='completion_changed',
                                message='已完成状态不再成立，未执行空爪放置，等待重新规划')
                    # Every actual manipulation invalidates all earlier proofs,
                    # even when no human edit (and hence no revision) occurred.
                    prior_verified = verified
                    if physical and not (result and result.get('skipped') is True):
                        verified = None
                    if step['skill'] == 'ask':
                        accepted = self._approve(steps[index:], step['question'])
                        result = dict(ok=accepted, changed=False,
                            reason='approval_only_no_scene_change',
                            message='已获操作授权；不代表人工已调整现场或目标已完成'
                            if accepted else '人工拒绝，未继续执行')
                    elif step['skill'] == 'finish':
                        result = (copy.deepcopy(memory['_audit_result']) if self._audit_current(memory)
                                  else self._verify_all(executor, memory, verified))
                        self._end_audit(executor, memory, result, result=result, expected_revision=revision)
                        finished = bool(result.get('ok'))
                        finish_failed = not finished
                    elif step['skill'] == 'verify':
                        selected = ([memory['goals'][step['goal_index']]]
                                    if 'goal_index' in step else memory['goals'])
                        result = self._cached_verification(verified, selected)
                    if result is None:
                        result = executor.execute(step, memory['goals'],
                            protected_goals=memory['protected_goals'], cancel_event=ctl._cancel,
                            expected_revision=revision)
                    if self._defer_step(step, result, memory, steps, number):
                        verified = prior_verified
                        continue
                    if result.get('ok') and (result.get('fault') or result.get('aborted')):
                        # Opening can succeed before a retreat/driver failure.
                        # Physical release success cannot authorize another step.
                        result = dict(result, ok=False, message=(
                            '执行报告包含机械臂故障，已停止后续动作并保留现场'
                            if result.get('fault') else '任务已停止，保留当前现场与未完成事项'))
                    if (step['skill'] == 'verify' and result.get('ok') is True
                            and result.get('camera_verified') is True
                            and 'verified_revision' in result
                            and not self._verification_current(result)):
                        result = dict(result, ok=False, error='scene_changed',
                                      message='核验后现场发生变化，旧结果不能计作步骤完成')
                    self._feedback(step, result, number)
                    if ((physical and result.get('skipped') is not True)
                            or (step['skill'] == 'observe' and result.get('changed') is True)
                            or held_before != bool(self.session.snapshot().get('held_object'))):
                        self._invalidate_audit(memory)
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
                    if physical:
                        proof = self._action_proof(proof_step, result, memory['goals'])
                        if self._proof_updates_audit(memory, proof):
                            self._invalidate_audit(memory)
                        if proof is not None and not memory['protected_goals'] and not memory['restorations']:
                            verified = proof
                    elif step['skill'] == 'verify':
                        if self._verification_updates_audit(memory, result, selected):
                            self._invalidate_audit(memory)
                        if not memory['protected_goals'] and not memory['restorations']:
                            verified = self._proof(result, selected)
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
                    queue_completed = True
                    from .goal_readiness import goal_worklist
                    worklist = goal_worklist(self.session.scene, memory['goals'])
                    ready = any(item['status'] == 'observed_pending' for item in worklist)
                    full_proof = self._cached_verification(verified, self._final_goals(memory))
                    if (verified and not memory['obligations'] and not self.session.snapshot().get('held_object')
                            and (not ready or full_proof is not None)):
                        result = (copy.deepcopy(memory['_audit_result']) if self._audit_current(memory)
                                  else self._verify_all(executor, memory, verified))
                        self._end_audit(executor, memory, result, result=result, expected_revision=revision)
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
                from .goal_readiness import goal_worklist
                worklist = goal_worklist(self.session.scene, memory['goals'])
                self._publish(goal_worklist=worklist)
                work_done = queue_completed and not any(item['status'] == 'observed_pending' for item in worklist)
                if work_done or finish_failed or memory['stagnation'] >= MAX_STAGNATION:
                    reviewed, new = self._end_audit(executor, memory, failure)
                    if reviewed is not None:
                        if reviewed.get('ok'):
                            return self._finish(True, '全部原始目标和复位义务已通过最终核验', memory, reviewed)
                        failure = reviewed
                        if not new and post_audit_decision:
                            return self._finish(False, '全部目标审核后仍有未完成项，已保留原目标与暂缓项', memory, reviewed)
                if memory['stagnation'] >= MAX_STAGNATION:
                    if self._audit_current(memory):
                        continue
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

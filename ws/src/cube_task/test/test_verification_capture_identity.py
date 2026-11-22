"""A later cropped view must not discard a different goal's completed check."""
import copy
import threading
from types import MethodType, SimpleNamespace

import numpy as np
import pytest

from cube_task.assistant.simulation import SimulationSession
from cube_task.assistant.skill_execution import SkillExecutor
from cube_task.core.types import Detection, Zone


def verification_session(*, changed=False, first_physical=True):
    red = Detection('red', .99, center_base=np.array([.2, -.2, .05]),
                    top_z=.05, height=.05, length=.05, width=.05, yaw=0.)
    green = copy.deepcopy(red)
    green.label, green.center_base = 'green', np.array([.2, .2, .05])
    cropped = copy.deepcopy(red)
    cropped.valid, cropped.reject_reason = False, 'touches_image_border'
    zones = {name: Zone(name, np.array([.2, y, 0.]), (.2, .2))
             for name, y in (('zone_red', -.2), ('zone_green', .2))}
    scene = dict(objects=[], zones=[dict(name=z.name, center=z.center.tolist(),
                                         size=list(z.size), yaw=0.) for z in zones.values()])
    session = SimpleNamespace(scene=scene, _physics=None,
        _environment_revision=0, _observation_revision=0, _interventions=[],
        runner=SimpleNamespace(zones=zones, has_held_recovery=False),
        world=SimpleNamespace(held=None), _operation_lock=threading.RLock(),
        cancel_event=threading.Event(), _check_open=lambda: None,
        snapshot=lambda: dict(held_object=None, editor={}))
    objects = {label: SimpleNamespace(label=label) for label in ('red', 'green')}
    session._evaluation_object = lambda det: objects.get(det.label) if det.valid else None
    session._verify_zone = lambda obj, name: (obj is not None and name=='zone_'+obj.label
        and (first_physical or obj.label!='red'))
    captures = []
    def observe(target, **kwargs):
        captures.append(target)
        session._last_fresh = [red] if len(captures)==1 else [green, cropped]
        session._observed_detections = list(session._last_fresh)
        if changed and len(captures)==2:
            session._environment_revision += 1
        session._observation_revision = session._environment_revision
        return scene
    session.observe = observe
    session.verify_goals = MethodType(SimulationSession.verify_goals, session)
    goals = [dict(label='red', zone_name='zone_red'), dict(label='green', zone_name='zone_green')]
    return session, goals, captures


def test_each_goal_keeps_identity_from_its_own_fresh_capture():
    session, goals, captures = verification_session()
    result = SkillExecutor(session).execute(dict(id='check', skill='verify'), goals)
    assert len(captures)==2
    assert result['ok'] and all(item['truth']['success'] for item in result['items'])
    assert result['camera_verified']


@pytest.mark.parametrize('changed,first_physical', [(True, True), (False, False)])
def test_per_capture_identity_cannot_override_scene_changes_or_physical_failure(changed, first_physical):
    session, goals, _captures = verification_session(changed=changed, first_physical=first_physical)
    result = SkillExecutor(session).execute(dict(id='check', skill='verify'), goals)
    assert not result['ok']

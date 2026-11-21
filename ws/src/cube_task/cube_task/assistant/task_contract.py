"""Bind clear task-wide requests to identity metadata, never motion geometry.

The inventory is an operator/task manifest, not the configured class catalog
or the currently graspable camera detections. This deliberately small intent
recognizer binds only unqualified requests for every object's default zone;
other meanings remain with the normal planner.
"""
from __future__ import annotations

import copy
import re


_OBJECT_ZH = r'(?:物品|物体|物件|东西|目标)'
_ALL_ZH = re.compile(
    r'(?:所有|全部|全体)(?:的)?' + _OBJECT_ZH
    + r'|每(?:一)?(?:件|个)(?:' + _OBJECT_ZH
    + r'|(?=都|各自|分别|放|移|送|归|回|复位|整理|还))'
    + r'|' + _OBJECT_ZH + r'(?:全都|全部|都)'
    + r'|^(?:(?:请|请你|麻烦|帮我|将|把|让)\s*)*(?:全部|全都)'
    + r'(?=按|各自|分别|放|移|送|归|回|复位|整理|还)')
_ALL_EN = re.compile(
    r'\b(?:all(?:\s+of)?(?:\s+(?:the|these|those))?|every|each)'
    r'\s+(?:objects?|items?|things?)\b')
_DEFAULT = re.compile(
    r'默认(?:的)?(?:放置)?(?:区域|区|位置|地点)'
    r'|\bdefault\s+(?:zones?|areas?|regions?|positions?|locations?)\b')
_ACTION = re.compile(r'放|移|送|归|回|复位|整理|还|\b(?:put|place|move|return|restore|reset|send|sort)\b')
_SCOPE_EXCLUSION = re.compile(
    # A subset, an exclusion, or a measurement qualification is not "all".
    r'仅|只|其中|中的|里的|除了|除去|除外|以外|不包括|排除|剩下|剩余|其他|其余'
    r'|可见|看见|看到|已观察|检测到|识别到|有效|可执行|可抓取|能抓'
    r'|\b(?:only|except|excluding|exclude|other|remaining|visible|observed|detected|valid|'
    r'selectable|graspable|but|leave)\b|\b(?:can\s+see|currently\s+seen|apart\s+from)\b'
    r'|\b(?:objects?|items?|things?)\s+(?:from|near|with|that|which)\b'
    r'|\b(?:objects?|items?|things?)\s+(?:in|on)\s+'
    r'(?!(?:their|its|the)\s+default\b|default\b)'
    # Task-start pose is a different destination contract.
    r'|原位|原处|原来|原先|原始|最初|起始|开始时'
    r'|\b(?:original|initial|starting|previous)\b|\bback\s+where\b'
    r'|不动|不移动|保留原状')
_NEGATED = re.compile(
    r'不要|别|不用|无需|不必|(?<!能)不能|(?<!可)不可|禁止|暂不|停止|取消'
    r'|\b(?:not|never|without|cancel|stop)\b|\bdon[’\x27]t\b'
)
_READONLY = re.compile(
    r'是否|有没有|哪些|什么|哪个|多少|如何|怎么|在哪|告诉|回答|说明|描述|列出|查询'
    r'|查看|检查|确认|核实|统计|介绍'
    r'|\b(?:tell|show|list|check|verify|confirm|describe|explain)\b'
    r'|^(?:should|is|are|do|does|did|how|what|where|which|why|have|has)\b')
_COMPLETED_QUESTION = re.compile(r'了[吗么]|(?:已经|已).*[?？]')


def _all_default_request(instruction):
    if not isinstance(instruction, str):
        return False
    text = ' '.join(instruction.casefold().split())
    action = _ACTION.search(text)
    if (not action or not (_ALL_ZH.search(text) or _ALL_EN.search(text))
            or not _DEFAULT.search(text) or _SCOPE_EXCLUSION.search(text)
            or _NEGATED.search(text) or _COMPLETED_QUESTION.search(text)):
        return False
    readonly = _READONLY.search(text)
    # Polite questions can request execution. A later check is part of that
    # execution, while "check/tell me whether to put ..." is a read-only query.
    return readonly is None or readonly.start() > action.start()


def _inventory_labels(inventory):
    if not isinstance(inventory, (list, tuple)) or not inventory:
        raise ValueError('全体任务缺少实际物品清单，不能按可见子集宣告完成。')
    labels = []
    for item in inventory:
        if isinstance(item, str):
            label, count = item, 1
        elif isinstance(item, dict):
            label, count = item.get('label'), item.get('count', 1)
        else:
            raise ValueError('实际物品清单格式无效。')
        if (not isinstance(label, str) or not label or label != label.strip()
                or len(label) > 100 or type(count) is not int or count != 1
                or label in labels):
            raise ValueError('实际物品清单必须能逐件唯一引用，不能忽略缺失或重复物品。')
        labels.append(label)
    return labels


def bind_task_contract(instruction, scene, inventory, existing_goals=()):
    """Return an ``all_default`` contract, or None for other task semantics.

    ``inventory`` contains labels or ``{label, count: 1}`` records. It must come
    from the actual task manifest, independently of camera validity. Existing
    normalized goals retain their indices and destinations; conflicting goals
    raise ValueError instead of changing an earlier promise during recovery.
    No caller-owned input is changed and no coordinates enter the contract.
    """
    if not _all_default_request(instruction):
        return None
    labels = _inventory_labels(inventory)
    mapping = scene.get('class_to_zone') if isinstance(scene, dict) else None
    zones = scene.get('zones') if isinstance(scene, dict) else None
    if not isinstance(mapping, dict) or not isinstance(zones, list):
        raise ValueError('全体归位任务缺少默认区域配置。')
    defaults = {}
    for label in labels:
        zone = mapping.get(label)
        matches = [z for z in zones if isinstance(z, dict) and z.get('name') == zone]
        if (not isinstance(zone, str) or not zone or len(matches) != 1
                or matches[0].get('valid', True) is not True):
            raise ValueError('物品 %s 缺少唯一有效的默认区域，不能省略该目标。' % label)
        defaults[label] = zone
    if not isinstance(existing_goals, (list, tuple)):
        raise ValueError('已有最终目标清单无效。')
    goals, seen = [], set()
    for goal in existing_goals:
        if not isinstance(goal, dict):
            raise ValueError('已有最终目标清单无效。')
        label = goal.get('label')
        if (not isinstance(label, str) or label not in defaults or label in seen
                or goal.get('zone_name') != defaults[label]
                or goal.get('destination') is not None
                or set(goal) - {'label', 'zone_name', 'destination'}):
            raise ValueError('已有最终目标与全体默认归位要求冲突，不能改写已承诺任务。')
        goals.append(copy.deepcopy(goal))
        seen.add(label)
    goals.extend(dict(label=label, zone_name=defaults[label])
                 for label in labels if label not in seen)
    return dict(kind='all_default', required_labels=list(labels), goals=goals)

"""Small semantic dialogue references, independent of scene or action state."""
from __future__ import annotations

import re


MAX_MESSAGES = 8
MAX_CONTENT = 400
_ACTIONS = {'plan', 'answer', 'clarify', 'stop'}
_SECRETS = re.compile(
    r'\bBearer\s+[^\s,;，；]+'
    r'|\bsk-[A-Za-z0-9_-]{8,}'
    r'|(?:api[_ -]?key|access[_ -]?token|token|secret|密钥)\s*[=:：]\s*["\']?[^\s,;，；]+',
    re.IGNORECASE)


def _safe_text(value):
    if not isinstance(value, str):
        return ''
    value = _SECRETS.sub('[敏感内容已隐藏]', value)
    return ' '.join(value.split())[:MAX_CONTENT]


def model_conversation(value):
    """A separate allowlist: history cannot smuggle actions or approvals."""
    if not isinstance(value, list) or len(value) > MAX_MESSAGES:
        raise ValueError('对话上下文必须是至多八条语义摘要。')
    result = []
    for record in value:
        if not isinstance(record, dict):
            raise ValueError('对话摘要记录无效。')
        role = record.get('role')
        fields = {'role', 'content'} | ({'action'} if role == 'assistant' else set())
        if (role not in ('user', 'assistant') or set(record) != fields
                or not isinstance(record.get('content'), str)):
            raise ValueError('对话摘要字段不符合约定。')
        content = _safe_text(record['content'])
        if not content or (role == 'assistant' and (
                not isinstance(record['action'], str) or record['action'] not in _ACTIONS)):
            raise ValueError('对话摘要内容或动作类型无效。')
        item = dict(role=role, content=content)
        if role == 'assistant':
            item['action'] = record['action']
        result.append(item)
    return result


def conversation_summary(messages, *, after_id=0):
    """Keep four completed turns; assistant raw streams are never forwarded.

    Replanning can yield several assistant plans for one user message. Only its
    last validated public explanation is useful as the next turn's antecedent.
    A completed plan here records dialogue, not proof that its actions succeeded.
    """
    turns, user, assistant = [], None, None

    def finish_turn():
        if user is not None and assistant is not None:
            turns.append([user, assistant])

    for message in messages:
        if (message.get('id', 0) <= after_id or message.get('status') != 'complete'):
            continue
        if message.get('role') == 'user':
            finish_turn()
            content = _safe_text(message.get('content'))
            user = dict(role='user', content=content) if content else None
            assistant = None
        elif message.get('role') == 'assistant' and user is not None:
            decision = message.get('decision')
            if (not isinstance(decision, dict) or not isinstance(decision.get('action'), str)
                    or decision['action'] not in _ACTIONS):
                continue
            content = _safe_text(decision.get('message'))
            if content:
                assistant = dict(role='assistant', content=content, action=decision['action'])
    finish_turn()
    return [record for turn in turns[-MAX_MESSAGES // 2:] for record in turn]

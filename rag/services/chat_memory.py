from collections import OrderedDict, deque
import threading
import time

_lock = threading.Lock()
_sessions = OrderedDict()
MAX_TURNS = 10
MAX_SESSIONS = 5000
SESSION_TTL_SECONDS = 3600


def _expire(now):
    while _sessions:
        session_id, item = next(iter(_sessions.items()))
        if now - item["updated"] <= SESSION_TTL_SECONDS:
            break
        _sessions.pop(session_id, None)

def get(session_id):
    now = time.monotonic()
    with _lock:
        _expire(now)
        item = _sessions.get(session_id)
        if item is None or now - item["updated"] > SESSION_TTL_SECONDS:
            _sessions.pop(session_id, None)
            return []
        item["updated"] = now
        _sessions.move_to_end(session_id)
        return list(item["turns"])

def add(session_id, question, answer):
    now = time.monotonic()
    with _lock:
        _expire(now)
        item = _sessions.setdefault(
            session_id,
            {"turns": deque(maxlen=MAX_TURNS), "updated": now, "title": question},
        )
        if not item.get("title"):
            item["title"] = question
        item["turns"].append({"question": question, "answer": answer})
        item["updated"] = now
        _sessions.move_to_end(session_id)
        while len(_sessions) > MAX_SESSIONS:
            _sessions.popitem(last=False)

def clear(session_id):
    with _lock:
        _sessions.pop(session_id, None)

def list_sessions():
    now = time.monotonic()
    with _lock:
        _expire(now)
        sessions = []
        for sid, item in reversed(list(_sessions.items())):
            turns = item["turns"]
            title = item.get("title") or (turns[0]["question"] if turns else "New chat")
            sessions.append({
                "session_id": sid,
                "title": title,
                "updated": item["updated"],
                "turns_count": len(turns)
            })
        return sessions

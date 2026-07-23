from collections import OrderedDict
from datetime import datetime, timezone
from threading import Lock


class IntakeActivityStore:
    def __init__(self, max_sessions: int = 200):
        self.max_sessions = max_sessions
        self._items: OrderedDict[str, dict] = OrderedDict()
        self._lock = Lock()

    def update(
        self,
        session_id: str,
        phase: str,
        detail: str,
        *,
        active: bool = True,
        tool_name: str | None = None,
    ) -> dict:
        with self._lock:
            previous = self._items.get(session_id, {})
            activity = {
                "session_id": session_id,
                "phase": phase,
                "detail": detail,
                "active": active,
                "tool_name": tool_name,
                "sequence": int(previous.get("sequence", 0)) + 1,
                "updated_at": datetime.now(timezone.utc),
            }
            self._items[session_id] = activity
            self._items.move_to_end(session_id)
            while len(self._items) > self.max_sessions:
                self._items.popitem(last=False)
            return dict(activity)

    def get(self, session_id: str) -> dict:
        with self._lock:
            activity = self._items.get(session_id)
            if activity is not None:
                return dict(activity)
        return {
            "session_id": session_id,
            "phase": "IDLE",
            "detail": "大模型待命",
            "active": False,
            "tool_name": None,
            "sequence": 0,
            "updated_at": None,
        }


intake_activity = IntakeActivityStore()

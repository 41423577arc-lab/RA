from typing import Generic, TypeVar


ActionT = TypeVar("ActionT")


class RepeatedActionGuard(Generic[ActionT]):
    def __init__(self, max_repeated_actions: int):
        if max_repeated_actions < 1:
            raise ValueError("max_repeated_actions must be at least 1")
        self.max_repeated_actions = max_repeated_actions
        self._last_action: ActionT | None = None
        self._count = 0

    def observe(self, action: ActionT) -> bool:
        if action == self._last_action:
            self._count += 1
        else:
            self._last_action = action
            self._count = 1
        return self._count > self.max_repeated_actions

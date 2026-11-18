from typing import Literal
from common.base import Modulator
import numpy as np


class Reward_Modulator(Modulator):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def signal(self, locals):
        reward = locals.get("reward")
        return reward


class QTableModulator(Modulator):
    def __init__(self, num_states: int, num_actions: int, *,
                 return_type: Literal["td_error", "prev_reward"] = "prev_reward"):
        super().__init__()
        self._num_states = num_states
        self._num_actions = num_actions
        assert return_type in ("td_error", "prev_reward")
        self._return_type = return_type
        self.q_lookup = np.zeros((num_states, num_actions))

    def reset(self):
        self.q_lookup.fill(0.0)

    def signal(self, locals: dict) -> float:
        state = locals.get("state")
        action = locals.get("action")
        assert 0 <= state < self._num_states
        assert 0 <= action < self._num_actions
        reward = locals.get("reward")
        # Lookup recent reward from previous state-action pair
        prev_reward = self.q_lookup[state, action].copy()
        # Update lookup table
        self.q_lookup[state, action] = reward
        # Return signal based on return_type
        if self._return_type == "prev_reward":
            return prev_reward
        elif self._return_type == "td_error":
            return reward - prev_reward
    

class TD_Error_Modulator(Modulator):
    def __init__(self, *, gamma=0.9):
        super().__init__()
        self.gamma = gamma

    def signal(self, locals):
        state = locals.get("state")
        next_state = locals.get("next_state")
        action = locals.get("action")
        reward = locals.get("reward")
        q_table = locals.get("self").network.weights[0]

        if action < 0:
            # No action selected
            q_now = np.max(q_table[state, :])
        else:
            q_now = q_table[state, action]
        q_next = np.max(q_table[next_state, :])
        td_error = reward + self.gamma * q_next - q_now
        return td_error


MODULATOR_DICT = {
    "q_lookup": QTableModulator,
    "reward": Reward_Modulator,
    "td_error": TD_Error_Modulator,
}

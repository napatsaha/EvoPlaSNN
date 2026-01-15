import abc
import numpy as np


class Modulator(abc.ABC):
    def signal(self, locals: dict) -> float:
        pass


class Reward_Modulator(Modulator):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def signal(self, locals):
        reward = locals.get("reward")
        return reward
    

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
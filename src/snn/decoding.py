import numpy as np


class RewardManager:
    def __init__(self):
        self.memory = {"t": [], "label": [], "prediction": [], "reward": []}

    def add(self, t: int, label: int, prediction: int, reward: float):
        self.memory["t"].append(t)
        self.memory["label"].append(label)
        self.memory["prediction"].append(prediction)
        self.memory["reward"].append(reward)

    def calculate_reward(self, label: int, spike_out: np.ndarray, timestep: int) -> float:
        """
        Calculate the reward based on the label and prediction.
        """
        # If no spikes
        if np.all(spike_out == 0):
            pred = None
            reward = 0.0
        else:
            pred = np.argmax(spike_out).item()
            reward = float(np.equal(label, pred).item())

        self.add(timestep, label, pred, reward)
        return reward

    def accuracy(self) -> float:
        """
        Calculate the accuracy of the predictions.
        """
        if len(self.memory["label"]) == 0:
            return 0.0
        accuracy = np.mean(np.equal(self.memory["label"], self.memory["prediction"]))
        return accuracy

    def reset(self):
        """
        Reset the reward manager.
        """
        self.memory = {"t": [], "label": [], "prediction": [], "reward": []}
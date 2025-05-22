from .base import LearningRule


class ANN_Rule(LearningRule):
    """
    A LearningRule that is approximated by a fully-connected ANN.
    """
    def __init__(self, input_size, hidden_size=None):
        super().__init__()
        self.input_size = input_size

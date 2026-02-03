from typing import Literal
from common.base import LearningRule
class Empty_Rule(LearningRule):
    """
    A dummy learning rule that does nothing.
    """
    def __init__(self):
        super().__init__()

    def update(self, *args, **kwargs) -> float:
        # No update
        return 0.0
    

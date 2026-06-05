from abc import ABC, abstractmethod
import numpy as np
import gymnasium as gym

class Controller(ABC):
    @abstractmethod
    def get_action(self, state, env) -> np.ndarray:
        pass
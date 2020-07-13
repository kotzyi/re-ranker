import numpy as np


class ENV:
    """
    This is a guide for making environment class
    """
    def __init__(self, state_dim: int):
        # must have the state_dim parameter
        pass

    def reset(self) -> np.array:
        # return state
        pass

    def step(self, action: np.array, debug: bool = False) -> (float, np.array, int):
        # return reward, state, done
        pass

    def print_status(self) -> None:
        pass

    def get_reward(self, state: list) -> float:
        pass

    def sample(self) -> np.array:
        pass

    def softmax(self, x: np.array) -> np.array:
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

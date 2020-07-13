import numpy as np
import logging
from envs.env import ENV


logger = logging.getLogger(__name__)


class CardTypeV1(ENV):
    def __init__(self, state_dim: int):
        self.num_category = state_dim
        self.user_personality = np.random.dirichlet(np.ones(self.num_category), size=1)[0]
        self.done = 0
        self.satisfy = 1
        self.views = 0
        self.disappoint_factor = 0.9
        self.state = np.random.dirichlet(np.ones(self.num_category), size=1).tolist()
        self.priv_state = None
        self.action = None
        self.reward = None

    def reset(self) -> np.array:
        self.user_personality = np.random.dirichlet(np.ones(self.num_category), size=1)[0]
        self.done = np.array([[0]])
        self.satisfy = 1

        return np.reshape(np.array(self.state), (1, self.num_category))  # np.array(self.state)

    def step(self, action: np.array, debug: bool = False) -> (np.array, np.array, np.array):
        self.action = action[0]
        self.priv_state = self.state
        self.state = [1 - abs(a - p) for a, p in zip(self.action, self.user_personality)]
        self.satisfy = sum(self.state)
        self.reward = np.array([self.get_reward(self.state)])
        if debug:
            self.print_status()

        return self.reward, np.reshape(self.state, (1, self.num_category)), self.done

    def print_status(self):
        logger.debug(f'--------------------------------------------------------------------------')
        logger.debug(f'USER PERSONALITY : {self.user_personality}')
        logger.debug(f'USER SATISFY     : {self.satisfy}')
        logger.debug(f'CURR STATE       : {self.priv_state}')
        logger.debug(f'NEXT STATE       : {self.state}')
        logger.debug(f'ACTION           : {self.action}')
        logger.debug(f'REWARD           : {self.reward}')
        logger.debug(f'DONE             : {self.done}')
        logger.debug(f'--------------------------------------------------------------------------')

    def get_reward(self, state: list) -> float:
        return (sum(state) - self.num_category) / 10

    def sample(self) -> np.array:
        return np.array(np.random.dirichlet(np.ones(self.num_category), size=1))

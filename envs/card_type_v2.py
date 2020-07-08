import numpy as np
import logging
import math
from envs.env import ENV
from rank.config import CardTypeV2Config


logger = logging.getLogger(__name__)


class CardTypeV2(ENV):
    """
    N users
    M card types
    Observations
    user personality
    """
    def __init__(self, state_dim: int = 21):
        if state_dim % 3:
            raise ValueError("a state dimension of CARD TYPE V2 must be multiples of three")
        self.state_dim = state_dim
        self.num_card_types: int = int(self.state_dim / 3)
        self.num_users = CardTypeV2Config.users
        self.num_cards = CardTypeV2Config.cards
        self.user_personalities = np.random.dirichlet(np.random.random(self.num_card_types), size=self.num_users)[0]
        # self.trendy_factors = np.random.random(self.num_users)
        self.done = 0
        self.long_term_observation = np.zeros(self.num_card_types)  # np.random.dirichlet(np.random.random(self.num_card_types), size=self.num_users)
        self.short_term_observation = self.long_term_observation
        self.feedback = np.zeros(self.num_card_types)
        self.state = np.concatenate([self.long_term_observation, self.short_term_observation, self.feedback])
        self.previous_state = self.state
        self.satisfy = np.random.random(self.num_users)
        self.action = None
        self.reward = None

    def reset(self) -> np.array:
        # TO-DO: MUST consider user personality should be reset or not !!!!
        self.long_term_observation = np.zeros(self.num_card_types)
        self.short_term_observation = self.long_term_observation
        self.feedback = np.zeros(self.num_card_types)
        self.state = np.concatenate([self.long_term_observation, self.short_term_observation, self.feedback])
        self.done = 0

        return np.reshape(self.state, (1, self.state_dim))

    def step(self, action: np.array, debug: bool = False) -> (float, np.array, int):
        self.action = self.softmax(action[0])
        self.previous_state = self.state
        self.feedback = np.array([1 - abs(a - p) for a, p in zip(self.action, self.user_personalities)])
        recommended_cards = self.recommend_card(self.action)
        self.short_term_observation = np.array([
            round(min(s, p)) for s, p in zip(recommended_cards, self.user_personalities * self.num_cards)])

        self.long_term_observation = self.long_term_observation + self.short_term_observation
        num_short_term_view = max(1, sum(self.short_term_observation))
        num_long_term_view = max(1, sum(self.long_term_observation))

        self.state = np.concatenate([self.long_term_observation / num_long_term_view,
                                     self.short_term_observation / num_short_term_view, self.feedback])
        self.reward = self.get_reward()
        if debug:
            self.print_status()

        return self.reward, np.reshape(self.state, (1, self.state_dim)), self.done

    def print_status(self):
        logger.debug(f'--------------------------------------------------------------------------')
        logger.debug(f'USER PERSONALITY : {self.user_personalities}')
        logger.debug(f'USER SATISFY     : {self.feedback}')
        logger.debug(f'CURR STATE       : {self.previous_state}')
        logger.debug(f'NEXT STATE       : {self.state}')
        logger.debug(f'ACTION           : {self.action}')
        logger.debug(f'REWARD           : {self.reward}')
        logger.debug(f'DONE             : {self.done}')
        logger.debug(f'--------------------------------------------------------------------------')

    def get_reward(self) -> float:
        return (sum(self.feedback) - self.num_card_types) / self.num_card_types

    def sample(self) -> np.array:
        return np.array(np.random.dirichlet(np.ones(self.num_card_types), size=1))

    def recommend_card(self, action):
        recommended_cards = np.array([])
        seed = np.random.choice(a=list(range(self.num_card_types)), size=self.num_cards, p=action)

        for category in range(self.num_card_types):
            recommended_cards = np.append(recommended_cards, np.count_nonzero(seed == category))
        return recommended_cards

    def softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)
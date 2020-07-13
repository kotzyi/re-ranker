import copy
import numpy as np
import torch


class Policy(object):
    """
    This is a guide for making policy class
    """
    def __init__(self, conf: dict):
        self.actor = None
        self.actor_target = None
        self.actor_optimizer = None
        self.critic = None
        self.critic_target = None
        self.critic_optimizer = None
        self.device = None

    def train(self, args, replay_buffer) -> (float, float):
        """
        Please follow the below steps
        1. Sample replay buffer
        2. Compute the target Q value
        3. Get current Q estimate
        4. Compute critic loss
        5. Optimize the critic
        6. Compute actor loss
        7. Optimize the actor
        8. Update the frozen target models

        :param args: arguments from console command
        :param replay_buffer: replay buffer that keeps (state, next_state, reward, done) transitions
        :return: actor and critic loss
        """
        pass

    def select_action(self, states: np.array) -> np.array:
        num_users = states.shape[0]
        actions = []
        states = torch.FloatTensor(states).to(self.device)
        for state in states:
            actions.append(self.actor(state))
        actions = torch.cat(actions)
        actions = actions.cpu().data.numpy().reshape(num_users, -1)
        return actions

    def save(self, filename) -> None:
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer.pt")

        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer.pt")

    def load(self, filename) -> None:
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer.pt"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer.pt"))
        self.actor_target = copy.deepcopy(self.actor)

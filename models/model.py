import copy
import numpy as np
import torch


class Policy(object):
    def __init__(self, conf: dict):
        self.actor = None
        self.actor_target = None
        self.actor_optimizer = None
        self.critic = None
        self.critic_target = None
        self.critic_optimizer = None

    def train(self, args, replay_buffer) -> (float, float):
        # Sample replay buffer
        # Compute the target Q value
        # Get current Q estimate
        # Compute critic loss
        # Optimize the critic
        # Compute actor loss
        # Optimize the actor
        # Update the frozen target models
        pass

    def select_action(self, state: np.array) -> np.array:
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        x = self.actor(state)
        return np.array([x.cpu().data.numpy().flatten()])

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

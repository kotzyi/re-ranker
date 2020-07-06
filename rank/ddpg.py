import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from apex import amp

# Implementation of Deep Deterministic Policy Gradients (DDPG)
# Paper: https://arxiv.org/abs/1509.02971


class DDPGActor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, model):
        super(DDPGActor, self).__init__()

        self.fc1 = nn.Linear(state_dim, model.fc1)
        self.fc2 = nn.Linear(model.fc1, model.fc2)
        self.fc3 = nn.Linear(model.fc2, model.fc3)
        self.fc_mu = nn.Linear(model.fc3, action_dim)
        self.LayerNorm = nn.LayerNorm(action_dim, eps=model.layer_norm)
        self.max_action = max_action

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        mu = (torch.tanh(self.LayerNorm(self.fc_mu(x))) + self.max_action) / 2
        return mu


class DDPGCritic(nn.Module):
    def __init__(self, state_dim, action_dim, model):
        super(DDPGCritic, self).__init__()

        self.fc1 = nn.Linear(state_dim, model.fc1)
        self.fc2 = nn.Linear(model.fc1 + action_dim, model.fc2)
        self.fc3 = nn.Linear(model.fc2, model.fc3)

    def forward(self, state, action):
        q = F.relu(self.fc1(state))
        q = F.relu(self.fc2(torch.cat([q, action], 1)))
        return self.fc3(q)


class DDPG(object):
    def __init__(self, conf):
        self.discount = conf.discount
        self.tau = conf.tau
        self.device = conf.device
        self.actor = DDPGActor(conf.state_dim, conf.action_dim, conf.max_action, conf.actor).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=conf.actor_lr)

        self.critic = DDPGCritic(conf.state_dim, conf.action_dim, conf.critic).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), weight_decay=1e-2, lr=conf.critic_lr)

        if conf.fp16:
            self.actor, self.actor_optimizer = amp.initialize(self.actor, self.actor_optimizer,
                                                              opt_level=conf.fp16_opt_level)
            self.critic, self.critic_optimizer = amp.initialize(self.critic, self.critic_optimizer,
                                                              opt_level=conf.fp16_opt_level)

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        return np.array([self.actor(state).cpu().data.numpy().flatten()])

    def train(self, args, replay_buffer):
        # Sample replay buffer
        state, action, reward, next_state, not_done = replay_buffer.sample(args.batch_size, self.device)

        # Compute the target Q value
        target_Q = self.critic_target(next_state, self.actor_target(next_state))
        target_Q = reward + (not_done * self.discount * target_Q).detach()

        # Get current Q estimate
        current_Q = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q, target_Q)

        # Optimize the critic
        self.critic.zero_grad()
        if args.fp16:
            with amp.scale_loss(critic_loss, self.critic_optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            critic_loss.backward()
        if args.fp16:
            torch.nn.utils.clip_grad_norm_(amp.master_params(self.critic_optimizer), args.max_grad_norm)
        else:
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), args.max_grad_norm)
        self.critic_optimizer.step()

        # Compute actor loss
        actor_loss = -self.critic(state, self.actor(state)).mean()

        # Optimize the actor
        self.actor.zero_grad()
        if args.fp16:
            with amp.scale_loss(actor_loss, self.actor_optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            actor_loss.backward()
        if args.fp16:
            torch.nn.utils.clip_grad_norm_(amp.master_params(self.actor_optimizer), args.max_grad_norm)
        else:
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), args.max_grad_norm)
        self.actor_optimizer.step()

        # Update the frozen target models
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return critic_loss.clone().detach().cpu().mean().item(), actor_loss.clone().detach().cpu().mean().item()

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")

        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)

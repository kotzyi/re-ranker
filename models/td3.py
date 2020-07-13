import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.policy import Policy


# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477


class TD3Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, model):
        super(TD3Actor, self).__init__()

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
        mu = (torch.tanh(self.fc_mu(x)) + self.max_action) / 2
        return mu


class TD3Critic(nn.Module):
    def __init__(self, state_dim, action_dim, model):
        super(TD3Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, model.fc1)
        self.l2 = nn.Linear(model.fc1, model.fc2)
        self.l3 = nn.Linear(model.fc2, model.fc3)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, model.fc1)
        self.l5 = nn.Linear(model.fc1, model.fc2)
        self.l6 = nn.Linear(model.fc2, model.fc3)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1


class TD3(Policy):
    def __init__(self, conf):
        self.max_action = conf.max_action
        self.discount = conf.discount
        self.tau = conf.tau
        self.policy_noise = conf.policy_noise
        self.noise_clip = conf.noise_clip
        self.policy_freq = conf.policy_freq
        self.device = conf.device
        self.total_it = 0

        self.actor = TD3Actor(conf.state_dim, conf.action_dim, conf.max_action, conf.actor).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=conf.actor_lr)

        self.critic = TD3Critic(conf.state_dim, conf.action_dim, conf.critic).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=conf.critic_lr)

        if conf.fp16:
            try:
                from apex import amp
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

            self.actor, self.actor_optimizer = amp.initialize(self.actor, self.actor_optimizer,
                                                              opt_level=conf.fp16_opt_level)
            self.critic, self.critic_optimizer = amp.initialize(self.critic, self.critic_optimizer,
                                                                opt_level=conf.fp16_opt_level)

    def train(self, args, replay_buffer):
        self.total_it += 1

        # Sample replay buffer
        state, action, reward, next_state, not_done = replay_buffer.sample(args.batch_size, self.device)
        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (
                    torch.randn_like(action) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)

            next_action = (
                    self.actor_target(next_state) + noise
            ).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + self.discount * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic.zero_grad()
        if args.fp16:
            try:
                from apex import amp
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

            with amp.scale_loss(critic_loss, self.critic_optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            critic_loss.backward()

        if args.fp16:
            torch.nn.utils.clip_grad_norm_(amp.master_params(self.critic_optimizer), args.max_grad_norm)
        else:
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), args.max_grad_norm)
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:

            # Compute actor lose
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

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
        return None, None

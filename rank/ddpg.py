import torch
import torch.nn as nn
from torch.nn.functional import gelu, relu
import torch.nn.functional as F

ACT2FN = {"relu": F.relu, "mish": lambda x: x * torch.tanh(nn.functional.softplus(x))}


class MuNet(nn.Module):
    def __init__(self):
        super(MuNet, self).__init__()
        self.fc1 = nn.Linear(7, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 32)
        self.fc_mu = nn.Linear(32, 7)
        self.LayerNorm = nn.LayerNorm(7, eps=1e-12)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        mu = (torch.tanh(self.LayerNorm(self.fc_mu(x))) + 1) / 2  # Multipled by 2 because the action space of the Pendulum-v0 is [-2,2]
        return mu


class QNet(nn.Module):
    def __init__(self):
        super(QNet, self).__init__()

        self.fc_s = nn.Linear(7, 64)
        self.fc_a = nn.Linear(7, 64)
        self.fc_q = nn.Linear(128, 32)
        self.fc_3 = nn.Linear(32, 1)
        self.LayerNorm = nn.LayerNorm(7, eps=1e-12)

    def forward(self, x, a):
        h1 = F.relu(self.fc_s(x))
        h2 = F.relu(self.fc_a(a))
        cat = torch.cat([h1, h2], dim=1)
        q = F.relu(self.fc_q(cat))
        q = self.fc_3(q)
        return q


class DDPGCritic(nn.Module):
    def __init__(self, config):
        super(DDPGCritic, self).__init__()
        # self.fc_s = nn.Linear(7, 64)
        # self.fc_a = nn.Linear(7, 64)
        # self.fc_q = nn.Linear(128, 32)
        # self.fc_3 = nn.Linear(32, 1)
        self.action_layer = DDPGActionMLP(config.action)
        # self.trend_layer = DDPGTrendMLP(config.trend)
        self.short_term_personality_layer = DDPGShortTermPersonalityMLP(config.long_term_personality)
        # self.long_term_personality_layer = DDPGLongTermPersonalityMLP(config.short_term_personality)
        self.critic_head = DDPGCriticHead(config.critic_head)
        self.critic_hidden = DDPGCriticHidden(config.critic_hidden)

    def forward(self, state, action):
        # trend = state[:, :6]
        # long = state[:, 6:12]
        # short = state[:, 12:]
        # long_term_personality = self.long_term_personality_layer(long)
        short_term_personality = self.short_term_personality_layer(state)
        # trend = self.trend_layer(trend)
        hidden_action = self.action_layer(action)
        hidden_states = self.critic_hidden(short_term_personality, hidden_action)
        value = self.critic_head(hidden_states)

        # value = torch.tanh(value) + 1 / 2
        return value
        # h1 = F.relu(self.fc_s(state))
        # h2 = F.relu(self.fc_a(action))
        # cat = torch.cat([h1, h2], dim=1)
        # q = F.relu(self.fc_q(cat))
        # q = self.fc_3(q)
        # return q


class DDPGActor(nn.Module):
    def __init__(self, config):
        super(DDPGActor, self).__init__()
        # self.trend_layer = DDPGTrendMLP(config.trend)
        # self.short_term_personality_layer = DDPGShortTermPersonalityMLP(config.long_term_personality)
        self.long_term_personality_layer = DDPGLongTermPersonalityMLP(config.short_term_personality)
        self.actor_head = DDPGActorHead(config.actor_head)
        self.softmax = nn.Softmax(dim=1)
        # self.fc1 = nn.Linear(7, 256)
        # self.fc2 = nn.Linear(256, 128)
        # self.fc_mu = nn.Linear(128, 7)
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, state):
        # trend = state[:, :6]
        # long = state[:, 6:12]
        # short = state[:, 12:]
        long_term_personality = self.long_term_personality_layer(state)
        # short_term_personality = self.short_term_personality_layer(state)
        # trend = self.trend_layer(trend)
        action = self.softmax(self.actor_head(long_term_personality))

        return action
        # x = F.relu(self.fc1(state))
        # x = F.relu(self.fc2(x))
        # mu = self.softmax(self.fc_mu(x))  # Multipled by 2 because the action space of the Pendulum-v0 is [-2,2]
        # return mu


class DDPGActorHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = DDPGBaseMLP(config)
        # self.decoder = nn.Linear(config.output_size, config.output_size, bias=False)
        # self.bias = nn.Parameter(torch.zeros(config.output_size))
        # self.decoder.bias = self.bias

    def forward(self, short_term_personality):
        # hidden_states = torch.cat((trend, long_term_personality, short_term_personality), 1)
        hidden_states = self.dense(short_term_personality)
        # hidden_states = self.decoder(hidden_states)
        return hidden_states


class DDPGCriticHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = DDPGBaseMLP(config)
        # self.decoder = nn.Linear(config.output_size, config.output_size, bias=False)
        # self.bias = nn.Parameter(torch.zeros(config.output_size))
        # self.decoder.bias = self.bias

    def forward(self, hidden_states):
        # hidden_states = torch.cat([short_term_personality, action], dim=1)
        hidden_states = self.dense(hidden_states)
        # hidden_states = self.decoder(hidden_states)
        return hidden_states


class DDPGCriticHidden(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = DDPGBaseMLP(config)
        # self.decoder = nn.Linear(config.output_size, config.output_size, bias=False)
        # self.bias = nn.Parameter(torch.zeros(config.output_size))
        # self.decoder.bias = self.bias

    def forward(self, short_term_personality, action):
        hidden_states = torch.cat([short_term_personality, action], dim=1)
        hidden_states = self.dense(hidden_states)
        # hidden_states = self.decoder(hidden_states)
        return hidden_states


class DDPGTrendMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.base_layer = DDPGBaseMLP(config)

    def forward(self, env_states):
        return self.base_layer(env_states)


class DDPGLongTermPersonalityMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.base_layer = DDPGBaseMLP(config)

    def forward(self, user_states):
        return self.base_layer(user_states)


class DDPGShortTermPersonalityMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.base_layer = DDPGBaseMLP(config)

    def forward(self, user_states):
        return self.base_layer(user_states)


class DDPGActionMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.base_layer = DDPGBaseMLP(config)

    def forward(self, user_states):
        return self.base_layer(user_states)


class DDPGBaseMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.input_size, config.output_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = nn.ReLU()
        else:
            self.transform_act_fn = nn.ReLU()
        self.LayerNorm = nn.LayerNorm(config.output_size, eps=config.layer_norm_eps)

    def forward(self, user_states):
        user_states = F.relu(self.dense(user_states))
        # user_states = self.transform_act_fn(user_states)
        user_states = self.LayerNorm(user_states)
        return user_states

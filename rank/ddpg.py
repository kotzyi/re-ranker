import torch
import torch.nn as nn
from torch.nn.functional import gelu, relu


ACT2FN = {"gelu": gelu, "relu": relu, "mish": lambda x: x * torch.tanh(nn.functional.softplus(x))}


class DDPGCritic(nn.Module):
    def __init__(self, config):
        super(DDPGCritic, self).__init__()
        self.action_layer = DDPGActionMLP(config.action)
        # self.trend_layer = DDPGTrendMLP(config.trend)
        self.short_term_personality_layer = DDPGShortTermPersonalityMLP(config.long_term_personality)
        # self.long_term_personality_layer = DDPGLongTermPersonalityMLP(config.short_term_personality)
        self.critic_head = DDPGCriticHead(config.critic_head)

    def forward(self, state, action):
        # trend = state[:, :6]
        # long = state[:, 6:12]
        # short = state[:, 12:]
        # long_term_personality = self.long_term_personality_layer(long)
        short_term_personality = self.short_term_personality_layer(state)
        # trend = self.trend_layer(trend)
        hidden_action = self.action_layer(action)
        value = self.critic_head(short_term_personality, hidden_action)
        # value = torch.tanh(value) + 1 / 2
        return value


class DDPGActor(nn.Module):
    def __init__(self, config):
        super(DDPGActor, self).__init__()
        # self.trend_layer = DDPGTrendMLP(config.trend)
        self.short_term_personality_layer = DDPGShortTermPersonalityMLP(config.long_term_personality)
        # self.long_term_personality_layer = DDPGLongTermPersonalityMLP(config.short_term_personality)
        self.actor_head = DDPGActorHead(config.actor_head)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, state):
        # trend = state[:, :6]
        # long = state[:, 6:12]
        # short = state[:, 12:]
        # long_term_personality = self.long_term_personality_layer(long)
        short_term_personality = self.short_term_personality_layer(state)
        # trend = self.trend_layer(trend)
        action = self.actor_head(short_term_personality)
        action = (torch.tanh(action) + 1) / 2
        return action


class DDPGActorHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = DDPGBaseMLP(config)
        self.decoder = nn.Linear(config.output_size, config.output_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.output_size))
        self.decoder.bias = self.bias

    def forward(self, short_term_personality):
        # hidden_states = torch.cat((trend, long_term_personality, short_term_personality), 1)
        hidden_states = self.dense(short_term_personality)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class DDPGCriticHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = DDPGBaseMLP(config)
        self.decoder = nn.Linear(config.output_size, config.output_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.output_size))
        self.decoder.bias = self.bias

    def forward(self, short_term_personality, action):
        hidden_states = torch.cat((short_term_personality, action), 1)
        hidden_states = self.dense(hidden_states)
        hidden_states = self.decoder(hidden_states)
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
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(config.output_size, eps=config.layer_norm_eps)

    def forward(self, user_states):
        user_states = self.dense(user_states)
        user_states = self.transform_act_fn(user_states)
        user_states = self.LayerNorm(user_states)
        return user_states

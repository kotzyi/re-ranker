import torch
import torch.nn as nn
import torch.nn.functional as F


ACT2FN = {"gelu": F.gelu, "relu": F.relu, "mish": lambda x: x * torch.tanh(nn.functional.softplus(x))}


class DQN(nn.Module):
    def __init__(self, config):
        super(DQN, self).__init__()
        self.card_type_layer = DQNCardTypeMLP(config.card_type)
        self.category_layer = DQNCategoryMLP(config.category)
        self.personality_layer = DQNPersonalityMLP(config.personality)
        self.action_head = DQNActionHead(config.head)
        self.value_head = DQNValueHead(config.head)

    def forward(self, state):
        personality = self.personality_layer(state.personality)
        category = self.category_layer(state.category)
        card_type = self.card_type_layer(state.card_type)
        action = self.action_head(personality, category, card_type)
        value = self.value_head(personality, category, card_type)
        return action, value


class DQNActionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = DQNBaseMLP(config)
        self.decoder = nn.Linear(config.output_size, config.output_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.output_size))
        self.decoder.bias = self.bias

    def forward(self, personality, category, card_type):
        hidden_states = torch.cat((personality, category, card_type), 1)
        hidden_states = self.dense(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class DQNValueHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = DQNBaseMLP(config)
        self.decoder = nn.Linear(config.output_size, config.output_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.output_size))
        self.decoder.bias = self.bias

    def forward(self, personality, category, card_type):
        hidden_states = torch.cat((personality, category, card_type), 1)
        hidden_states = self.dense(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class DQNCardTypeMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.base_layer = DQNBaseMLP(config)

    def forward(self, env_states):
        return self.base_layer(env_states)


class DQNCategoryMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.base_layer = DQNBaseMLP(config)

    def forward(self, env_states):
        return self.base_layer(env_states)


class DQNPersonalityMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.base_layer = DQNBaseMLP(config)

    def forward(self, user_states):
        return self.base_layer(user_states)


class DQNBaseMLP(nn.Module):
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





import torch
import torch.optim as optim
import numpy as np
from collections import namedtuple
import torch.nn.functional as F
from rank.replay_buffer import ReplayBuffer
from rank.config import ModelConfig
from rank.ddpg import DDPGActor, DDPGCritic, MuNet, QNet
from rank.env import ENV
try:
    from torch.utils.tensorboard import SummaryWriter, FileWriter
except ImportError:
    from tensorboardX import SummaryWriter, FileWriter


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
State = namedtuple('State', ('personality', 'category', 'card_type'))

MEMORY_SIZE = 20000
BATCH_SIZE = 128
GAMMA = 0.99
TARGET_UPDATE = 100
NUM_EPISODES = 10000
ACTOR_LEARNING_RATE = 0.00005
CRITIC_LEARNING_RATE = 0.0001
TAU = 0.005
PRINT_INTERVAL = 10
MINIMUM_MEMORY_SIZE = 1000
NUM_CATEGORY = 7
DEBUG = False
EPISODE_DEPTH = 20


def add_noise(action, mu, sigma):
    noise = (np.random.normal(mu, sigma, NUM_CATEGORY) + action)[0]
    noise = softmax(noise)
    return np.reshape(noise, (1, NUM_CATEGORY))


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def train(actor, actor_target, critic, critic_target, memory, critic_optimizer, actor_optimizer):
    state, action, reward, next_state, done = memory.sample(BATCH_SIZE)

    target = reward + GAMMA * critic_target(next_state, actor_target(next_state))
    critic_loss = F.smooth_l1_loss(critic(state, action), target.detach())
    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()

    actor_loss = -critic(state, actor(state)).mean()  # That's all for the policy loss.
    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()
    return critic_loss.mean().item(), actor_loss.mean().item()


def soft_update(model, model_target):
    for param_target, param in zip(model_target.parameters(), model.parameters()):
        param_target.data.copy_(param_target.data * (1.0 - TAU) + param.data * TAU)


def main():
    tb_writer = SummaryWriter("./log")
    score = 0.0
    memory = ReplayBuffer(MEMORY_SIZE)
    env = ENV(num_recommend=10, num_category=NUM_CATEGORY)

    actor = MuNet() # DDPGActor(ModelConfig)
    actor_target = MuNet() #DDPGActor(ModelConfig)
    actor_target.load_state_dict(actor.state_dict())

    critic = QNet() # DDPGCritic(ModelConfig)
    critic_target = QNet() # DDPGCritic(ModelConfig)
    critic_target.load_state_dict(critic.state_dict())

    actor_optimizer = optim.Adam(actor.parameters(), lr=ACTOR_LEARNING_RATE)
    critic_optimizer = optim.Adam(critic.parameters(), lr=CRITIC_LEARNING_RATE)
    c_loss, a_loss = 0.0, 0.0
    action = []

    for i_episode in range(NUM_EPISODES):
        state = env.reset()

        for t in range(EPISODE_DEPTH):
            action = actor(torch.from_numpy(state).float())

            action = action.detach().numpy()
            # action = add_noise(action.detach().numpy(), 0, 0.1)
            reward, next_state, done = env.step(action, debug=DEBUG)
            memory.push(state, action, reward/10, next_state, done)
            score += reward
            state = next_state

            if done:
                break
        if len(memory) > MINIMUM_MEMORY_SIZE:
            for i in range(TARGET_UPDATE):
                c_loss, a_loss = train(actor, actor_target, critic, critic_target, memory, critic_optimizer, actor_optimizer)
                soft_update(critic, critic_target)
                soft_update(actor, actor_target)

        if i_episode % PRINT_INTERVAL == 0 and i_episode != 0:
            tb_writer.add_scalar('RETURN', score/(EPISODE_DEPTH * PRINT_INTERVAL), i_episode)
            tb_writer.add_scalar('CRITIC_LOSS', c_loss, i_episode)
            tb_writer.add_scalar('ACTOR_LOSS', a_loss, i_episode)
            # for name, param in actor.named_parameters():
            #     tb_writer.add_histogram(name, param.clone().cpu().data.numpy(), i_episode)
            # for name, param in critic.named_parameters():
            #     tb_writer.add_histogram(name, param.clone().cpu().data.numpy(), i_episode)

            print(f"# of episode :{i_episode}, avg score : {score/(EPISODE_DEPTH * PRINT_INTERVAL):.2f}, action: {action[0]}")
            score = 0.0


if __name__ == "__main__":
    main()

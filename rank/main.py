import torch
import torch.optim as optim
from collections import namedtuple
import torch.nn.functional as F
from rank.replay_buffer import ReplayBuffer
from rank.config import ModelConfig
from rank.ddpg import DDPGActor, DDPGCritic
from rank.env import ENV


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
State = namedtuple('State', ('personality', 'category', 'card_type'))

MEMORY_SIZE = 10000
BATCH_SIZE = 64
GAMMA = 0.99
TARGET_UPDATE = 10
NUM_EPISODES = 10000
ACTOR_LEARNING_RATE = 0.0001
CRITIC_LEARNING_RATE = 0.0001
TAU = 0.005
PRINT_INTERVAL = 20
MINIMUM_MEMORY_SIZE = 2000
NUM_CATEGORY = 1


def train(actor, actor_target, critic, critic_target, memory, critic_optimizer, actor_optimizer):
    state, action, reward, next_state, done = memory.sample(BATCH_SIZE)

    target = reward + (1 - done) * GAMMA * critic_target(next_state, actor_target(next_state))
    critic_loss = F.smooth_l1_loss(critic(state, action), target.detach())
    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()
    # print(f'critic loss: {critic_loss.item()}')

    print(-critic(state, actor(state)).mean())
    actor_loss = -critic(state, actor(state)).mean()  # That's all for the policy loss.
    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()
    # print(f'actor loss: {actor_loss.item()}')


def soft_update(model, model_target):
    for param_target, param in zip(model_target.parameters(), model.parameters()):
        param_target.data.copy_(param_target.data * (1.0 - TAU) + param.data * TAU)


def main():
    score = 0.0
    memory = ReplayBuffer(MEMORY_SIZE)
    env = ENV(num_recommend=10, num_category=NUM_CATEGORY)

    actor = DDPGActor(ModelConfig).to(device)
    actor_target = DDPGActor(ModelConfig).to(device)
    actor_target.load_state_dict(actor.state_dict())

    critic = DDPGCritic(ModelConfig).to(device)
    critic_target = DDPGCritic(ModelConfig).to(device)
    critic_target.load_state_dict(critic.state_dict())

    actor_optimizer = optim.Adam(actor.parameters(), lr=ACTOR_LEARNING_RATE)
    critic_optimizer = optim.Adam(critic.parameters(), lr=CRITIC_LEARNING_RATE)

    for i_episode in range(NUM_EPISODES):
        state = env.reset()

        for t in range(30):  # maximum length of episode is 200 for Pendulum-v0
            action = actor(torch.from_numpy(state).float())
            # a = a.item() + ou_noise()[0]
            action = action.detach().numpy()
            reward, next_state, done = env.step(action)
            memory.push(state, action, reward/100, next_state, done)
            score += reward
            state = next_state

            if done:
                break
        if len(memory) > MINIMUM_MEMORY_SIZE:
            for i in range(TARGET_UPDATE):
                train(actor, actor_target, critic, critic_target, memory, critic_optimizer, actor_optimizer)
                soft_update(critic, critic_target)
                soft_update(actor, actor_target)

        if i_episode % PRINT_INTERVAL == 0 and i_episode != 0:
            print(f"# of episode :{i_episode}, avg score : {score/PRINT_INTERVAL:.2f}")
            score = 0.0


if __name__ == "__main__":
    main()

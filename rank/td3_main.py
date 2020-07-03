import numpy as np
import torch
import argparse
import os

from rank.td3 import TD3
from rank.env import ENV
from rank.replay_buffer import ReplayBuffer
try:
    from torch.utils.tensorboard import SummaryWriter, FileWriter
except ImportError:
    from tensorboardX import SummaryWriter, FileWriter


def eval_policy(policy, eval_env, eval_episodes=100):
    avg_reward = 0.
    state, done = eval_env.reset(), False
    for _ in range(eval_episodes):
        action = policy.select_action(np.array(state))
        reward, next_state, done = env.step(action, debug=False)
        avg_reward += reward

    avg_reward /= eval_episodes

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    return avg_reward


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="TD3")  # Policy name (TD3, DDPG or OurDDPG)
    parser.add_argument("--env", default="HalfCheetah-v2")  # OpenAI gym environment name
    parser.add_argument("--seed", default=0, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_timesteps", default=25e3, type=int)  # Time steps initial random policy is used
    parser.add_argument("--eval_freq", default=1000, type=int)  # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e6, type=int)  # Max time steps to run environment
    parser.add_argument("--expl_noise", default=0.1)  # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=256, type=int)  # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99)  # Discount factor
    parser.add_argument("--tau", default=0.005)  # Target network update rate
    parser.add_argument("--policy_noise", default=0.2)  # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5)  # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)  # Frequency of delayed policy updates
    parser.add_argument("--save_model", action="store_true")  # Save model and optimizer parameters
    parser.add_argument("--load_model", default="")  # Model load file name, "" doesn't load, "default" uses file_name
    parser.add_argument("--print_interval", default=1000, type=int)
    args = parser.parse_args()

    file_name = f"{args.policy}_{args.env}_{args.seed}"
    print("---------------------------------------")
    print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
    print("---------------------------------------")

    if not os.path.exists("./results"):
        os.makedirs("./results")

    if args.save_model and not os.path.exists("./models"):
        os.makedirs("./models")

    env = ENV(num_recommend=10, num_category=7)

    tb_writer = SummaryWriter("./log")
    state_dim = 7
    action_dim = 7
    max_action = 1.0

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": args.discount,
        "tau": args.tau,
    }

    # Initialize policy
    # Target policy smoothing is scaled wrt the action scale
    kwargs["policy_noise"] = args.policy_noise * max_action
    kwargs["noise_clip"] = args.noise_clip * max_action
    kwargs["policy_freq"] = args.policy_freq
    policy = TD3(**kwargs)


    if args.load_model != "":
        policy_file = file_name if args.load_model == "default" else args.load_model
        policy.load(f"./models/{policy_file}")

    replay_buffer = ReplayBuffer(20000)

    # Evaluate untrained policy
    evaluations = [eval_policy(policy, env)]

    state, done = env.reset(), False
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0
    c_loss, a_loss = None, None

    for t in range(int(args.max_timesteps)):
        episode_timesteps += 1

        # Select action randomly or according to policy
        if t < args.start_timesteps:
            action = env.sample()
        else:
            action = np.array((
                    policy.select_action(np.array(state))
                    + np.random.normal(0, max_action * args.expl_noise, size=action_dim)
            ).clip(-max_action, max_action))

        reward, next_state, done = env.step(action, debug=False)
        replay_buffer.push(state, action, reward / 10, next_state, done)
        state = next_state
        episode_reward += reward

        # Train agent after collecting sufficient data
        if t >= args.start_timesteps:
            c_loss, a_loss = policy.train(replay_buffer, args.batch_size)

        if t >= args.start_timesteps and episode_timesteps % args.print_interval == 0 and episode_timesteps != 0:
            # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
            tb_writer.add_scalar('RETURN', episode_reward / args.print_interval, episode_timesteps)
            if c_loss is not None:
                tb_writer.add_scalar('CRITIC_LOSS', c_loss, episode_timesteps)
                tb_writer.add_scalar('ACTOR_LOSS', a_loss, episode_timesteps)

            print(
                f"Total T: {t + 1} Episode Num: {episode_num + 1} Episode T: {episode_timesteps}"
                f" Reward: {episode_reward/args.print_interval:.3f}")
            # Reset environment
            # state, done = env.reset(), False
            episode_reward = 0
            # episode_timesteps = 0
            episode_num += 1

        # Evaluate episode
        if (t + 1) % args.eval_freq == 0:
            evaluations.append(eval_policy(policy, env))
            #np.save(f"./results/{file_name}", evaluations)
            #if args.save_model: policy.save(f"./models/{file_name}")

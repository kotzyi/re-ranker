import numpy as np
import logging
import argparse
import torch
import os
from rank.td3 import TD3
from rank.ddpg import DDPG
from rank.config import DDPGConfig, TD3Config
from rank.env import ENV
from rank.replay_buffer import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter


BASE_MODEL_CLASSES = {
    "DDPG": (DDPGConfig, DDPG),
    "TD3": (TD3Config, TD3),
}
logger = logging.getLogger(__name__)


def eval_policy(policy, eval_env, args):
    avg_reward = 0.
    state, done = eval_env.reset(), False
    for _ in range(args.eval_episodes):
        action = policy.select_action(np.array(state))
        reward, next_state, done = eval_env.step(action, debug=args.eval_debug)
        avg_reward += reward

    avg_reward /= args.eval_episodes
    logger.info(f"Evaluation over {args.eval_episodes} episodes: {avg_reward:.3f}")
    return avg_reward


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="DDPG", help="Policy name (TD3, DDPG)")
    parser.add_argument("--env", default="CardType-v1", help="OpenAI gym environment name")
    parser.add_argument("--start_timesteps", default=25e3, type=int, help="Time steps initial random policy is used")
    parser.add_argument("--eval_freq", default=1000, type=int, help="How often (time steps) we evaluate")
    parser.add_argument("--max_timesteps", default=1e6, type=int, help="Max time steps to run environment")
    parser.add_argument("--expl_noise", default=0.1, help="Std of Gaussian exploration noise")
    parser.add_argument("--batch_size", default=256, type=int, help="Batch size for both actor and critic")
    parser.add_argument("--save_model", action="store_true", help="Save model and optimizer parameters")
    parser.add_argument("--load_model", default="",
                        help="Model load file name, '' doesn't load, 'default' uses file_name")
    parser.add_argument("--print_interval", default=1000, type=int)
    parser.add_argument("--buffer_size", default=20000, type=int)
    parser.add_argument("--eval_episodes", default=100, type=int)
    parser.add_argument("--eval_debug", default=False, type=bool)
    parser.add_argument("--model_path", default="./models")
    parser.add_argument("--log_path", default="./logs")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--fp16", action="store_true",
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument("--fp16_opt_level", type=str, default="O1",
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']. "
                             "See details at https://nvidia.github.io/apex/amp.html")


    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S",
        level=logging.INFO,
    )

    args = parser.parse_args()
    model_config, model = BASE_MODEL_CLASSES[args.policy]
    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1

    model_config["fp16"] = args.fp16
    model_config["fp16_opt_level"] = args.fp16_opt_level
    model_config["device"] = device


    file_name = f"{args.policy}_{args.env}"
    logger.info(f"Policy: {args.policy}, Env: {args.env}")

    if args.save_model and not os.path.exists("./models"):
        os.makedirs(args.model_path)

    env = ENV(num_recommend=10, num_category=model_config.state_dim)

    tb_writer = SummaryWriter(args.log_path)
    policy = model(model_config)

    if args.load_model != "":
        policy_file = file_name if args.load_model == "default" else args.load_model
        policy.load(f"./models/{policy_file}")

    replay_buffer = ReplayBuffer(args.buffer_size)

    # Evaluate untrained policy
    evaluations = [eval_policy(policy, env, args)]

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
                    + np.random.normal(0, model_config.max_action * model_config.expl_noise,
                                       size=model_config.action_dim)
            ).clip(-model_config.max_action, model_config.max_action))

        reward, next_state, done = env.step(action, debug=args.eval_debug)
        replay_buffer.push(state, action, reward, next_state, done)
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

            logger.info(
                f"Total T: {t + 1} Episode Num: {episode_num + 1} Episode T: {episode_timesteps}"
                f" Reward: {episode_reward/args.print_interval:.3f}")
            # Reset environment
            episode_reward = 0
            episode_num += 1

        # Evaluate episode
        if (t + 1) % args.eval_freq == 0:
            evaluations.append(eval_policy(policy, env, args))
            if args.save_model:
                policy.save(f"{args.model_path}/{file_name}")


if __name__ == "__main__":
    main()
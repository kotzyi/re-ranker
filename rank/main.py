import numpy as np
import logging
import argparse
import torch
import os
from models.td3 import TD3
from models.ddpg import DDPG
from rank.config import DDPGConfig, TD3Config
from envs.card_type_v1 import CardTypeV1
from envs.card_type_v2 import CardTypeV2
from envs.card_type_v3 import CardTypeV3
from envs.env import ENV
from rank.replay_buffer import ReplayBuffer
try:
    from torch.utils.tensorboard import SummaryWriter, FileWriter
except ImportError:
    from tensorboardX import SummaryWriter, FileWriter


MODEL_CLASSES = {
    "DDPG": (DDPGConfig, DDPG),
    "TD3": (TD3Config, TD3),
}
ENVS = {
    "CARDTYPE-V1": CardTypeV1,
    "CARDTYPE-V2": CardTypeV2,
    "CARDTYPE-V3": CardTypeV3,
}
logger = logging.getLogger(__name__)


def eval_policy(policy, env: ENV, args) -> float:
    avg_reward = 0.
    state = env.reset()
    logger.info(f"personality: {env.user_personalities} first_states: {state} ")
    for _ in range(args.eval_episodes):
        action = policy.select_action(state)
        reward, state, done = env.step(action, debug=False)
        avg_reward += np.mean(reward)

    avg_reward /= args.eval_episodes
    logger.info(f"1 - Evaluation over {args.eval_episodes} episodes: {avg_reward:.3f} last_states: {state[-1][-2:]} last action: {action}")
    return avg_reward


# def eval_policy_2(policy, env: ENV, replay_buffer, args) -> float:
#     avg_reward = 0.
#
#     logger.info(f"personality: {env.user_personalities}")
#     for _ in range(args.eval_episodes):
#         state, _, _, _, _ = replay_buffer.sample(1, args.device)
#         action = policy.select_action(state.cpu().data.numpy())
#         reward, state, done = env.step(action, debug=False)
#         avg_reward += np.mean(reward)
#
#     avg_reward /= args.eval_episodes
#     logger.info(f"2 - Evaluation over {args.eval_episodes} episodes: {avg_reward:.3f} last_states: {state[-1][-2:]} last action: {action}")
#     return avg_reward


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="DDPG", help="Policy name (TD3, DDPG)")
    parser.add_argument("--env", default="CARDTYPE-V2", help="OpenAI gym environment name")
    parser.add_argument("--start_timesteps", default=25e3, type=int, help="Time steps initial random policy is used")
    parser.add_argument("--eval_freq", default=1000, type=int, help="How often (time steps) we evaluate")
    parser.add_argument("--max_timesteps", default=1e5, type=int, help="Max time steps to run environment")
    parser.add_argument("--expl_noise", default=0.1, help="Std of Gaussian exploration noise")
    parser.add_argument("--batch_size", default=256, type=int, help="Batch size for both actor and critic")
    parser.add_argument("--save_model", action="store_true", help="Save model and optimizer parameters")
    parser.add_argument("--load_model", default="",
                        help="Model load file name, '' doesn't load, 'default' uses file_name")
    parser.add_argument("--print_interval", default=1000, type=int)
    parser.add_argument("--buffer_size", default=50000, type=int)
    parser.add_argument("--eval_episodes", default=200, type=int)
    parser.add_argument("--eval_debug", default=False, type=bool)
    parser.add_argument("--model_path", default="./pretrained_models")
    parser.add_argument("--log_path", default="./logs")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--fp16", action="store_true",
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument("--fp16_opt_level", type=str, default="O1",
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']. "
                             "See details at https://nvidia.github.io/apex/amp.html")

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S",
        level=logging.DEBUG,
    )
    args = parser.parse_args()

    # Load policy and model configuration
    model_config, policy = MODEL_CLASSES[args.policy]

    # Load environment
    env = ENVS[args.env](state_dim=model_config.state_dim)

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
    args.device = device

    save_file_name = f"{args.policy}_{args.env}"
    logger.info(f"Policy: {args.policy}, Env: {args.env}")

    if args.save_model and not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    tb_writer = SummaryWriter(f"{args.log_path}/{args.policy}-{args.env}")

    # Load policy object
    policy = policy(model_config)

    if args.load_model != "":
        policy_file = save_file_name if args.load_model == "default" else args.load_model
        policy.load(f"{args.model_path}/{policy_file}")

    # Load replay buffer
    replay_buffer = ReplayBuffer(args.buffer_size)

    # Evaluate untrained policy
    evaluations = [eval_policy(policy, env, args)]

    states = env.reset()
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0
    c_loss, a_loss = None, None

    for t in range(int(args.max_timesteps)):
        episode_timesteps += 1

        # Select action randomly or according to policy
        if t < args.start_timesteps:  # before start-timesteps, it uses random action.
            actions = env.sample()
            rewards, next_states, dones = env.step(actions, debug=False)
        else:  # after start-timesteps, it uses action from policy.
            actions = (policy.select_action(states)
                      + np.random.normal(0, model_config.max_action * model_config.expl_noise,
                                         size=model_config.action_dim)
                      ).clip(0, model_config.max_action)
            rewards, next_states, dones = env.step(actions, debug=args.eval_debug)

        for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
            replay_buffer.push(state, action, reward, next_state, done)
        states = next_states
        episode_reward += np.mean(rewards)

        # Train agent after collecting sufficient data
        if t >= args.start_timesteps:
            c_loss, a_loss = policy.train(args, replay_buffer)

        if t >= args.start_timesteps and episode_timesteps % args.print_interval == 0 and episode_timesteps != 0:
            # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
            tb_writer.add_scalar('RETURN', episode_reward / args.print_interval, episode_timesteps)
            if c_loss is not None:
                tb_writer.add_scalar(f'{args.policy}-{args.env}-CRITIC_LOSS', c_loss, episode_timesteps)
                tb_writer.add_scalar(f'{args.policy}-{args.env}-ACTOR_LOSS', a_loss, episode_timesteps)

            logger.info(
                f"Total T: {t + 1} Episode Num: {episode_num + 1} Episode T: {episode_timesteps}"
                f" Reward: {episode_reward/args.print_interval:.3f}")
            # Reset environment
            episode_reward = 0
            episode_num += 1

        # Evaluate episode
        if (t + 1) % args.eval_freq == 0:
            evaluations.append(eval_policy(policy, env, replay_buffer, args))
            if args.save_model:
                policy.save(f"{args.model_path}/{save_file_name}")


if __name__ == "__main__":
    main()

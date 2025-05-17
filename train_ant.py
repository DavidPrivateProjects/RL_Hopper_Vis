import gymnasium as gym
from stable_baselines3 import SAC, TD3, A2C
import os
import argparse
import numpy as np

# -----------------------------------------------
# Command-line usage examples:
#
# Train from scratch:
#   python train_ant.py Ant-v5 SAC all_legs --reward-threshold 1000 -t
#
# Continue training from a checkpoint:
#   python train_ant.py Ant-v5 SAC 3_legs -t --pretrained ./models/All_Legs/all_legs_SAC_125000.zip --reward-threshold 1000
#
# Test a trained model:
#   python train_ant.py Ant-v5 SAC -s ./models/SAC_125000.zip
# -----------------------------------------------

# Wrapper to selectively zero out action components
class ZeroOutActionsWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = env.action_space  # Keep action space unchanged

    def action(self, action):
        action = np.array(action, dtype=np.float32)
        # Disable specific joint actions (example: partial leg control)
        action[6:] = 0.0
        action[2:4] = 0.0
        return action

# Directory setup for model and logging output
model_dir = "models"
log_dir = "logs"
os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

# Select and initialize the RL model
def get_model(algo, env):
    if algo == 'SAC':
        return SAC('MlpPolicy', env, verbose=1, device='cuda', tensorboard_log=log_dir)
    elif algo == 'TD3':
        return TD3('MlpPolicy', env, verbose=1, device='cuda', tensorboard_log=log_dir)
    elif algo == 'A2C':
        return A2C('MlpPolicy', env, verbose=1, device='cuda', tensorboard_log=log_dir)
    else:
        print('Algorithm not supported.')
        return None

# Training loop with optional checkpoint loading and early stopping
def train(env, algo, path_to_pretrained_model=None, reward_threshold=None, name=None):
    if path_to_pretrained_model and os.path.isfile(path_to_pretrained_model):
        print(f"Loading pretrained model from {path_to_pretrained_model}")
        if algo == 'SAC':
            model = SAC.load(path_to_pretrained_model, env=env)
        elif algo == 'TD3':
            model = TD3.load(path_to_pretrained_model, env=env)
        elif algo == 'A2C':
            model = A2C.load(path_to_pretrained_model, env=env)
        else:
            print("Unsupported algorithm.")
            return
    else:
        model = get_model(algo, env)
        if not model:
            return

    TIMESTEPS = 5000
    iters = 0
    eval_env = ZeroOutActionsWrapper(gym.make(env.spec.id))  # Separate evaluation env

    while True:
        iters += 1
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)
        model.save(f"{model_dir}/{name}_{algo}_{TIMESTEPS * iters}")

        # Evaluate policy
        rewards = []
        for _ in range(5):
            obs = eval_env.reset()[0]
            done = False
            ep_reward = 0
            steps = 0

            while not done and steps < 1000:
                steps += 1
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, _, _ = eval_env.step(action)
                ep_reward += reward

            rewards.append(ep_reward)
            print(rewards)

        avg_reward = np.mean(rewards)
        print(f"[Iter {iters}] Average reward: {avg_reward:.2f}")

        if reward_threshold and avg_reward >= reward_threshold:
            print(f"Stopping early: reward {avg_reward:.2f} >= threshold {reward_threshold}")
            break

        if iters > 40:  # Optional cap to prevent endless training
            print("Training was stopped due to max iteration cap.")
            break

# Test loop to run a trained model in the environment
def test(env, algo, path_to_model):
    if algo == 'SAC':
        model = SAC.load(path_to_model, env=env)
    elif algo == 'TD3':
        model = TD3.load(path_to_model, env=env)
    elif algo == 'A2C':
        model = A2C.load(path_to_model, env=env)
    else:
        print('Algorithm not supported.')
        return

    obs = env.reset()[0]
    extra_steps = 50
    steps = 400

    while True:
        steps -= 1
        if steps == 0:
            steps = 400
            obs = env.reset()[0]
            extra_steps = 50

        action, _ = model.predict(obs)
        obs, _, terminated, _, _ = env.step(action)

        if terminated:
            extra_steps -= 1

        if extra_steps == 0:
            obs = env.reset()[0]
            extra_steps = 50


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train or test a reinforcement learning model.')
    parser.add_argument('gymenv', help='Gymnasium environment (e.g., Ant-v5)')
    parser.add_argument('algo', help='Algorithm (SAC, TD3, A2C)')
    parser.add_argument('name', help='Model/log save name')
    parser.add_argument('-t', '--train', action='store_true', help='Train the model')
    parser.add_argument('-s', '--test', metavar='path_to_model', help='Test a trained model (path to .zip)')
    parser.add_argument('--pretrained', metavar='path_to_model', help='Continue training from this checkpoint')
    parser.add_argument('--reward-threshold', type=float, help='Stop training when avg reward exceeds this')

    args = parser.parse_args()

    if args.train:
        env = gym.make(args.gymenv, render_mode=None)
        env = ZeroOutActionsWrapper(env)
        train(env, args.algo, path_to_pretrained_model=args.pretrained,
              reward_threshold=args.reward_threshold, name=args.name)

    if args.test:
        if os.path.isfile(args.test):
            env = gym.make(args.gymenv, render_mode='human')
            env = ZeroOutActionsWrapper(env)
            test(env, args.algo, path_to_model=args.test)
        else:
            print(f"Model file not found: {args.test}")

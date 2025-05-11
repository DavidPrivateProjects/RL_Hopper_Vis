import gymnasium as gym 
from stable_baselines3 import SAC, TD3, A2C
import os
import argparse
import numpy as np

# Train from scratch:
# python train_ant_new.py Ant-v5 SAC --reward-threshold 2000 -t

# Retrain from checkpoint, stop when reward > 3000:
# python train_ant_new.py Ant-v5 SAC -t --pretrained ./models/SAC_125000.zip --reward-threshold 2000

# Test a trained model:
# python train_ant_new.py Ant-v5 SAC -s ./models/SAC_125000.zip


class ZeroOutActionsWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = env.action_space  # Keep the action space unchanged

    def action(self, action):
        action = np.array(action, dtype=np.float32)
        #action[11:17] = 0.0  # Set actions 11 to 16 to zero
        return action

# Create directories to hold models and logs
model_dir = "models"
log_dir = "logs"
os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

def get_model(algo, env):
    """Returns the appropriate model based on the algorithm."""
    if algo == 'SAC':
        return SAC('MlpPolicy', env, verbose=1, device='cuda', tensorboard_log=log_dir)
    elif algo == 'TD3':
        return TD3('MlpPolicy', env, verbose=1, device='cuda', tensorboard_log=log_dir)
    elif algo == 'A2C':
        return A2C('MlpPolicy', env, verbose=1, device='cuda', tensorboard_log=log_dir)
    else:
        print('Algorithm not found')
        return None

def train(env, algo, path_to_pretrained_model=None, reward_threshold=None, eval_freq=5000):
    """Train the model, optionally continuing from a pretrained model and stopping based on average reward."""
    
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
    eval_env = ZeroOutActionsWrapper(gym.make(env.spec.id))  # Evaluation environment

    while True:
        iters += 1
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)
        model.save(f"{model_dir}/{algo}_{TIMESTEPS*iters}")
        
        # Evaluate the model
        rewards = []
        for _ in range(5):
            obs = eval_env.reset()[0]
            done = False
            ep_reward = 0
            time_steps = 0
            while not done and time_steps < 1000:
                time_steps += 1
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, _, _ = eval_env.step(action)
                ep_reward += reward
            rewards.append(ep_reward)
            print(rewards)

        avg_reward = np.mean(rewards)
        print(f"[Iter {iters}] Average reward: {avg_reward:.2f}")

        if reward_threshold and avg_reward >= reward_threshold:
            print(f"Stopping early: average reward {avg_reward:.2f} >= threshold {reward_threshold}")
            break

        if iters > 20:  # Optional max cap
            break

def test(env, algo, path_to_model):
    """Test the trained model."""
    model = None

    if algo == 'SAC':
        model = SAC.load(path_to_model, env=env)
    elif algo == 'TD3':
        model = TD3.load(path_to_model, env=env)
    elif algo == 'A2C':
        model = A2C.load(path_to_model, env=env)
    else:
        print('Algorithm not found')
        return

    obs = env.reset()[0]
    extra_steps = 50
    while True:
        action, _ = model.predict(obs)
        obs, _, terminated, _, _ = env.step(action)
        
        if terminated:
            extra_steps -= 1
        
        if extra_steps == 0:
            obs = env.reset()[0]
            extra_steps = 50

if __name__ == '__main__':
    # Parse command line inputs
    parser = argparse.ArgumentParser(description='Train or test model.')
    parser.add_argument('gymenv', help='Gymnasium environment i.e. Humanoid-v4')
    parser.add_argument('algo', help='StableBaseline3 RL algorithm i.e. SAC, TD3')
    parser.add_argument('-t', '--train', action='store_true', help='Flag to train the model')
    parser.add_argument('-s', '--test', metavar='path_to_model', help='Path to model for testing')
    parser.add_argument('--pretrained', metavar='path_to_model', help='Path to pretrained model for continued training')
    parser.add_argument('--reward-threshold', type=float, help='Stop training when avg reward exceeds this')

    args = parser.parse_args()

    if args.train:
        gymenv = gym.make(args.gymenv, render_mode=None)
        gymenv = ZeroOutActionsWrapper(gymenv)
        
        train(
            gymenv,
            args.algo,
            path_to_pretrained_model=args.pretrained,
            reward_threshold=args.reward_threshold
        )

    if args.test:
        if os.path.isfile(args.test):
            gymenv = gym.make(args.gymenv, render_mode='human')
            gymenv = ZeroOutActionsWrapper(gymenv)
            
            test(gymenv, args.algo, path_to_model=args.test)
        else:
            print(f'{args.test} not found.')

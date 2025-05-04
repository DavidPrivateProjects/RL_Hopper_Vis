import gymnasium as gym
from stable_baselines3 import SAC, TD3, A2C
import os
import argparse
import numpy as np
from collections import OrderedDict

class ZeroOutActionsWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = env.action_space  # Keep the action space unchanged

    def action(self, action):
        action = np.array(action, dtype=np.float32)
        action[11:17] = 0.0  # Set actions 11 to 16 to zero
        return action


# HumanoidStandup-v5
# python main.py HumanoidStandup-v5 SAC -s ./models/SAC_125000.zip
# python main.py HumanoidStandup-v5 SAC -t
# 
# https://www.youtube.com/watch?v=OqvXHi_QtT0

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


def train(env, algo):
    """Train the model."""
    model = get_model(algo, env)
    if not model:
        return

    TIMESTEPS = 50000
    iters = 0
    while True:
        iters += 1
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)
        model.save(f"{model_dir}/{algo}_{TIMESTEPS*iters}")
        if iters > 5:
            break


def test(env, algo, path_to_model):
    """Test the trained model."""
    model = None

    # Load the model based on the algorithm
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
        
        if terminated==True:
            extra_steps -= 1
        
        if extra_steps == 0:
            obs = env.reset()[0]
            extra_steps = 50


if __name__ == '__main__':
    # Parse command line inputs
    parser = argparse.ArgumentParser(description='Train or test model.')
    parser.add_argument('gymenv', help='Gymnasium environment i.e. Humanoid-v4')
    parser.add_argument('algo', help='StableBaseline3 RL algorithm i.e. SAC, TD3')
    parser.add_argument('-t', '--train', action='store_true')
    parser.add_argument('-s', '--test', metavar='path_to_model')
    args = parser.parse_args()

    if args.train:
        gymenv = gym.make(args.gymenv, render_mode=None)
        gymenv = ZeroOutActionsWrapper(gymenv)
        
        train(gymenv, args.algo)

    if args.test:
        if os.path.isfile(args.test):
            gymenv = gym.make(args.gymenv, render_mode='human')
            gymenv = ZeroOutActionsWrapper(gymenv)
            
            test(gymenv, args.algo, path_to_model=args.test)
        else:
            print(f'{args.test} not found.')
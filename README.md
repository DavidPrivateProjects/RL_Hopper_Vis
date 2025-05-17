# Mujoco Ant Reinforcement Learning with Injury
This project explores how MuJoCo agents trained with the SAC algorithm adapt to injuries: modeled as changes in the action space—and how quickly they can regain a target level of performance through retraining.

## Demo
### Ant Walker with no Injuries

<p align="center">
  <img src="https://github.com/user-attachments/assets/ae20329e-23f4-4113-8164-44c2c535c739" alt="Rot_All_Legs" width="20%" />
  <img src="https://github.com/user-attachments/assets/e80fb345-3fbf-41b8-a31e-ad53f70b21b0" alt="All_Legs" width="75%" />
</p>


### Ant Walker with only 3 functional Legs

<p align="center">
  <img src="https://github.com/user-attachments/assets/924d9931-f7da-4d69-ab98-5dff9a128769" alt="Rot_All_Legs" width="20%" />
  <img src="https://github.com/user-attachments/assets/fdf0c27a-e3e2-4dde-b34d-f3eaebab7cd9" alt="All_Legs" width="75%" />
</p>

### Ant Walker with 2 functional Legs (symmetrical)

<p align="center">
  <img src="https://github.com/user-attachments/assets/9ea10625-80e9-44f6-884a-a756bad052a4" alt="Rot_All_Legs" width="20%" />
  <img src="https://github.com/user-attachments/assets/df32a77f-998b-4ad1-8179-a827d60fcd51" alt="All_Legs" width="75%" />
</p>

### Ant Walker with functional 2 Legs (asymmetrical)

<p align="center">
  <img src="https://github.com/user-attachments/assets/dcdf8f31-834c-436a-b17e-be568d5fba79" alt="Rot_All_Legs" width="20%" />
  <img src="https://github.com/user-attachments/assets/61865376-028a-424f-af45-76bfeffdf7db" alt="All_Legs" width="75%" />
</p>

## Results
<p align="center">
<img src="https://github.com/user-attachments/assets/ac817306-02f9-4fe6-8caf-59755b7a40ac" alt="Rot_All_Legs" width="25%" />
</p>
<p align="center">
  <img src="https://github.com/user-attachments/assets/5f41314c-86da-4521-9626-a5678bf61b85" alt="All_Legs" width="45%" />
  <img src="https://github.com/user-attachments/assets/d2d93fba-f28c-4821-98ea-8e5f1518037d" alt="All_Legs" width="47%" />
</p>



## Requirements
To run this code, you need Python 3.7 or later and the following Python packages:
- gymnasium
- stable-baselines3
- torch
- numpy

## Soft Actor-Critic (SAC) Overview
Soft Actor-Critic (SAC) is a model-free, off-policy reinforcement learning algorithm designed for continuous action spaces. Unlike traditional actor-critic methods, SAC introduces an entropy-regularized objective that promotes exploration by encouraging policies to remain stochastic.

The SAC agent maintains two Q-value networks (to minimize overestimation bias), a value target network, and a stochastic policy network. The actor aims to maximize both the expected return and the entropy of its policy. This is formalized as:

J(π) = E [ Q(s, a) - α * log π(a | s) ]

Here, α is the temperature parameter that determines the trade-off between reward maximization and entropy. A higher entropy leads to more exploration, which helps the policy avoid premature convergence to suboptimal behaviors.

SAC learns from a replay buffer and updates its networks using sampled experiences, making it more sample-efficient than on-policy methods. Its stability and efficiency make it well-suited for complex robotic environments like Ant-v5.

## Usage
Training from Scratch
To train an agent from scratch using the SAC algorithm:

python train_ant.py Ant-v5 SAC all_legs --reward-threshold 1000 -t
This command launches training with the specified reward threshold. Models and logs will be saved under a folder prefix named all_legs.

Continue Training from a Checkpoint
You can continue training from a previously saved model as follows:

python train_ant.py Ant-v5 SAC three_legs -t --pretrained ./models/all_legs_SAC_125000.zip --reward-threshold 2000

This will load the checkpoint and resume training until the average reward exceeds the given threshold or a maximum iteration cap is reached.

Testing a Trained Model
To test a trained model and visualize its behavior:

python train_ant.py Ant-v5 SAC -s ./models/all_legs_SAC_125000.zip
This runs the agent in a rendering-enabled environment using the specified checkpoint.

## Command Line Interface
The script accepts the following arguments:

- gymenv: Name of the Gymnasium environment (e.g., Ant-v5)
- algo: Algorithm to use (SAC, TD3, or A2C)
- name: Prefix name used for saving model checkpoints and logs
- t, --train: Flag to train the model
- s, --test PATH: Test a trained model using the given path
- pretrained PATH: Path to a pretrained model for continued training
-reward-threshold FLOAT: Average reward threshold to trigger early stopping

## License
This project is licensed under the MIT License. You are free to use, modify, and distribute the code as long as the original license is included in any copies or substantial portions of the software.

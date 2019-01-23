# Pytorch_Multi_Agent_DDPG_Unity_Continuous_Control
Two deep reinforcement learning agents that solve a continuous control task with separate actors and a shared critic.

Uses Unity-ML Reacher environment: https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis

Written using Python 3 and PyTorch.

## Deep Reinforcement Learning
Uses Deep Deterministic Policy Gradients written in Pytorch. Two agents have a shared critic and experience replay, but keep separate actor networks which are trained independently. 

## The Environment
#### State Space
Each agent is provided a state with 24 different values. 8 values for that agent's paddle location, 8 values for the opponent's paddle location, and 8 values for the ball location. 

#### Action Space
Each agent takes 2 continuous actions at every time step which each have a value between -1 and 1. These features represent the force applied in the x and y directions (the environment has gravity, so applying a force in the positive y direction represents a jump). 

#### Scoring
An agent is rewarded with 0.1 when it hits the ball over the net to the other side. If the ball is hit out of bounds, it receives an additional reward of -0.01.

#### Termination
An episode terminates when the ball is hit out of bounds, if the ball hits the ground, or if the ball hits the net. There are several other conditionals which trigger termination, including a maximum number of time steps. More information can be read about the scoring on the environment page which is linked above.

## Dependencies
```
copy
math
matplotlib
pickle
random
sys
torch
unityagents
warnings
```

## Solve criteria
The agents have "solved" the environment when the agents achieve an average score of 0.5 in 100 consecutive games. The better agent's score is the one included in calculating the 100 game score after each game has terminated.

## Usage
All code is included in report.ipynb. Run each cell to train the agent from scratch. Weights are included for all neural networks in the network_weights folder. Metric data for graphing purposes is saved in the data folder.

# Goal Conditioned Inverse Reinforcement Learning (GC-IRL)

This repository presents the implementation and exploration of **Goal Conditioned Inverse Reinforcement Learning (GC-IRL)**. The project builds on the foundation of **Goal Conditioned Reinforcement Learning (GC-RL)** using Proximal Policy Optimization (PPO) and extends it into the inverse setting to infer reward structures from expert trajectories conditioned on specific goals.

## Features

- **Custom Environment**: RocketMinion, a 2D environment designed specifically for goal-conditioned learning frameworks with inertia-based dynamics and dynamic goals.
- **Goal-Conditioned RL**: Implementation of PPO for training agents to perform dynamic goal-directed tasks.
- **Goal-Conditioned IRL**: Conceptualized framework for inferring reward functions that generalize across diverse and dynamic goals.
- **Empirical Insights**: Preliminary results and challenges in implementing GC-RL and GC-IRL.

## Overview

### Background
Reinforcement Learning (RL) is a robust paradigm for training agents to perform tasks by maximizing a reward signal. However, designing reward functions for complex tasks is often infeasible. Inverse Reinforcement Learning (IRL) addresses this by extracting reward functions from expert demonstrations. Traditional IRL struggles with scalability and dynamic objectives, motivating this project’s development of GC-IRL.

### Goal Conditioned Learning
By conditioning rewards and policies on explicit goals, GC-RL and GC-IRL enhance adaptability to dynamic and multi-objective environments, providing a robust framework for real-world applications.

### RocketMinion Environment
RocketMinion is a custom simulation environment featuring:
- Dynamic goals that respawn upon completion.
- Inertia-based movement requiring sophisticated policy learning.
- A discrete action space for computational efficiency.

CLICK ON THE IMAGE FOR A YOUTUBE VIDEO:
[![YouTube](http://i.ytimg.com/vi/F9vgCncis1g/hqdefault.jpg)](https://www.youtube.com/watch?v=F9vgCncis1g)

## Methodology

1. **Goal Conditioned Reinforcement Learning (GC-RL)**:
   - Training with PPO in the RocketMinion environment.
   - Shaped reward function to guide agents towards goals while penalizing meandering.

2. **Goal Conditioned Inverse Reinforcement Learning (GC-IRL)**:
   - Augments expert trajectories with goal information.
   - Leverages Maximum Entropy IRL principles for goal-conditioned reward inference.
   - Optimization objective maximizes the likelihood of expert trajectories under a goal-conditioned policy.

3. **Challenges**:
   - Scalability of computing goal-conditioned rewards.
   - Representation of diverse and dynamic goals.
   - Algorithmic adjustments for goal-conditioned data.

## Preliminary Results

- **GC-RL**: Achieved 86% success rate in reaching goals within a 10-second time limit using PPO.
- **GC-IRL**: Conceptual framework developed, with promising potential to generalize reward functions across dynamic goals. Future work required to refine expert policy and scale implementation.

## Future Directions

- Complete the implementation and testing of GC-IRL.
- Optimize computational efficiency for goal-conditioned reward extraction.
- Explore deep learning-based representations for dynamic goal inference.
- Extend applications to real-world, multi-objective environments.

## Setup and Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/gc-irl.git
   cd gc-irl

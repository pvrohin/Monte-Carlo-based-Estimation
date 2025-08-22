# Monte Carlo-Based Estimation

Implementation of Monte Carlo methods to solve reinforcement learning problems based on averaging sample returns. This repository contains implementations of both on-policy Monte Carlo control and exploring starts Monte Carlo methods for a simple robot navigation environment.

## Project Overview

This project demonstrates Monte Carlo reinforcement learning algorithms applied to a robot navigation problem. The environment consists of a 6-state system (states 0-5) where the robot can move left (-1) or right (+1) from any position. The goal is to learn optimal policies that maximize expected rewards.

### Environment Description

- **States**: 6 discrete states [0, 1, 2, 3, 4, 5]
- **Actions**: Binary actions [-1, 1] representing left and right movement
- **Rewards**: 
  - +1 for reaching state 0 with action -1
  - +5 for reaching state 5 with action 1
  - 0 for all other state-action combinations
- **Terminal States**: States 0 and 5 are terminal states
- **Transition Model**: Stochastic transitions with 80% probability of intended movement, 15% probability of staying in place, and 5% probability of opposite movement

## Implementation Details

### 1. On-Policy Monte Carlo (`On_policy_MC.py`)

This implementation uses an ε-greedy policy with the following characteristics:

- **Policy**: ε-greedy with ε = 0.1
- **Starting State**: Fixed starting state (state 3)
- **Exploration**: Balances exploitation of learned Q-values with exploration of random actions
- **Value Estimation**: Weighted average of Q-values with 90% weight on the greedy action

### 2. Exploring Starts Monte Carlo (`Exploring_starts_MC.py`)

This implementation uses exploring starts to ensure sufficient exploration:

- **Policy**: Greedy policy based on learned Q-values
- **Starting State**: Random selection from non-terminal states [1, 2, 3, 4]
- **Exploration**: Initial random action followed by greedy policy
- **Value Estimation**: Direct selection of maximum Q-value for each state

## Key Features

- **Monte Carlo Control**: Uses sample returns to estimate action-value functions
- **Policy Improvement**: Automatically updates policies based on learned Q-values
- **Stochastic Environment**: Handles probabilistic state transitions
- **Visualization**: Plots value function convergence over episodes
- **Flexible Episode Length**: Configurable number of episodes for learning

## Algorithm Details

### Monte Carlo Control Loop

1. **Episode Generation**: Generate episodes using current policy
2. **Return Calculation**: Compute discounted returns for each state-action pair
3. **Q-Value Update**: Update Q-values using sample averages
4. **Policy Update**: Improve policy based on updated Q-values
5. **Convergence**: Repeat until policy stabilizes

### Key Parameters

- **Discount Factor (γ)**: 0.95
- **Episode Length**: 6000 episodes (configurable)
- **Epsilon (ε)**: 0.1 for ε-greedy exploration
- **Learning Rate**: Implicit through sample averaging

## Usage

### Prerequisites

```bash
pip install matplotlib
```

### Running the Algorithms

#### On-Policy Monte Carlo
```bash
python On_policy_MC.py
```

#### Exploring Starts Monte Carlo
```bash
python Exploring_starts_MC.py
```

### Output

Both implementations provide:
- Final Q-values for all state-action pairs
- Optimal policy for each non-terminal state
- Convergence plots showing value function evolution
- Performance metrics and analysis

## Code Structure

### Core Functions

- `model_distribution(from_state, action)`: Implements stochastic transition model
- `reward(current_robot_state, action_taken)`: Defines reward function
- `policy(state, Q)`: Implements action selection policy
- `Value_estimate(state, Q)`: Estimates state values from Q-values

### Data Structures

- **Q**: Dictionary storing action-value functions
- **Returns**: Dictionary storing sample returns for each state-action pair
- **V**: List of value function estimates over episodes

## Results and Analysis

The algorithms learn to:
- Navigate efficiently from any starting position to terminal states
- Maximize expected rewards through optimal action selection
- Handle stochastic environment dynamics
- Converge to stable policies over sufficient episodes

## Technical Notes

- **First-Visit Monte Carlo**: Only first occurrence of state-action pairs in episodes are used for updates
- **Sample Averaging**: Q-values are updated using arithmetic mean of all returns
- **Policy Convergence**: Policies converge to optimal deterministic policies
- **Exploration vs Exploitation**: Balanced through ε-greedy or exploring starts

## Future Enhancements

- Support for continuous state/action spaces
- Implementation of other Monte Carlo variants (every-visit, off-policy)
- Performance comparison with other RL algorithms
- Parameter tuning and optimization
- Extended environment scenarios

## License

This project is licensed under the terms specified in the LICENSE file.

## Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests to improve the implementation or documentation.

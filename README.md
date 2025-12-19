# GridWorld 4×4 – Value Iteration (Reinforcement Learning)

This project demonstrates Value Iteration, a fundamental algorithm in Reinforcement Learning (RL), applied to a simple 4×4 GridWorld environment. The goal is to compute the optimal value of each state using the Bellman Equation, allowing an agent to reason about long-term rewards before taking actions.

The agent starts at the top-left corner (state 0) and aims to reach the bottom-right corner (state 15), which is the terminal state.


Reinforcement Learning is a branch of machine learning where an agent learns how to behave in an environment by interacting with it. At each step:

- The agent is in a state
- It chooses an action
- The environment returns:
  - A reward
  - A new state

The agent’s objective is to maximize cumulative (long-term) reward, not just immediate reward.
Reinforcement Learning problems are typically modeled as a Markov Decision Process (MDP).

### Markov Decision Process (MDP)

An MDP is defined by five components:

- States (S)
All possible positions the agent can be in.
In this project: 16 states (0 to 15).

- Actions (A)
The choices available to the agent.
In this project: Up, Down, Left, Right.

- Transition Probability (P)
The probability of reaching the next state given the current state and action.

- Reward Function (R)
The immediate reward received after taking an action.

- Discount Factor (γ)
Determines how much future rewards are valued compared to immediate rewards.


### Environment

- Grid Size: 4×4 (16 states total)
- State Numbering: Row-major order (0 → 15)
- Starting State: Top-left corner (state 0)
- Terminal State: Bottom-right corner (state 15)
- Actions: Up, Down, Left, Right
- Policy: Uniform random policy (each action has probability 0.25)
- Discount Factor (γ): 1.0 (future rewards are fully valued)
- Convergence Threshold (θ): 1e-4

#### Reward Structure:
- -1 for each move
- 0 at terminal state (state 15)

This reward setup encourages the agent to reach the terminal state in as few steps as possible.

### Algorithm

The Bellman Equation provides a recursive definition of the value function. It expresses the value of a state as the expected reward of all possible next states.

**What is the Value Function?**

The value function, denoted as V(s), represents the expected cumulative reward an agent can obtain starting from state s and following a given policy.

In simple terms: “How good is it to be in this state?”

Higher values mean the state is closer (or more favorable) to the goal.

The value function V(s) is computed iteratively using the Bellman Equation:

    V(s) = Σ π(a|s) * Σ P(s'|s,a) * [R(s,a,s') + γ * V(s')]

Where:
- V(s): Value of current state
- π(a|s): Policy - Probability of taking action a in state s (uniform distribution: 0.25 for each action)
- P(s'|s,a): Transition probability - Probability of transitioning to next state s' (deterministic: 1.0 for valid moves)
- R(s,a,s'): Reward function (-1 per step, 0 at terminal) - Reward received after the transition
- γ: Discount factor (1.0 in this case)
- V(s'): Value of next state

#### What is Value Iteration?

Value Iteration is a dynamic programming algorithm used to compute the optimal value function.

- Start with arbitrary values (usually zeros)
- Repeatedly apply the Bellman Equation
- Update values until changes are very small (convergence)
Once the values converge, the agent implicitly knows which states are better and can derive an optimal policy.

### Flow

1. Initialize V(s) = 0 for all states
2. Set γ = 1.0, θ = 1e-4
3. Repeat until convergence:
   a. ΔV = 0 (track maximum change)
   b. For each non-terminal state s:
      i. Store old value: v = V(s)
      ii. For each action a (up, down, left, right):
          - Calculate next state s'
          - Handle boundary conditions
          - Compute expected value
      iii. Update V(s) using Bellman equation
      iv. ΔV = max(ΔV, |v - V(s)|)
   c. If ΔV < θ, break (converged)
5. Output final value function V(s)


      <img width="644" height="622" alt="image" src="https://github.com/user-attachments/assets/e184106d-92a8-4975-b895-207662fb8aff" />

The value function shows the expected cumulative reward (negative steps) from each state to the goal:

- State 15 (bottom-right): V(15) = 0 ✅ Terminal state
- State 14 (one step from goal): V(14) ≈ -29.99 (approximately 30 steps)
- State 0 (top-left, start): V(0) ≈ -59.42 (approximately 59 steps)

The values increase (become less negative) as we move closer to the terminal state, indicating shorter paths to the goal.

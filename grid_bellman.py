import numpy as np
import matplotlib.pyplot as plt

# Grid size
N = 4

# Rewards
MOVE_REWARD = -1
TERMINAL_REWARD = 0

# Initialize value function V(s) = 0 for all states
V = np.zeros((N, N))

# Discount factor (no discounting)
gamma = 1.0

# Convergence threshold
theta = 1e-4

# Terminal state (bottom-right corner)
terminal_state = (N-1, N-1)

print(f"Grid Size: {N}x{N}")
print(f"Discount Factor (gamma): {gamma}")
print(f"Convergence Threshold (theta): {theta}")
print(f"Terminal State: {terminal_state}")

# Define possible actions: up, down, left, right
actions = {
    'up': (-1, 0),
    'down': (1, 0),
    'left': (0, -1),
    'right': (0, 1)
}

# Probability of each action (equal probability)
action_prob = 1.0 / len(actions)

print(f"\nActions: {list(actions.keys())}")
print(f"Action Probability: {action_prob}")

def get_next_state(state, action, grid_size):
    """
    Calculate next state given current state and action.
    Handle grid boundaries (agent stays in place if trying to move out of bounds).
    """
    row, col = state
    delta_row, delta_col = action
    
    # Calculate next position
    next_row = row + delta_row
    next_col = col + delta_col
    
    # Check boundaries and stay in place if out of bounds
    if next_row < 0 or next_row >= grid_size or next_col < 0 or next_col >= grid_size:
        next_row, next_col = row, col
    
    return (next_row, next_col)

def value_iteration(V, gamma, theta, terminal_state, actions, grid_size):
    """
    Perform value iteration using the Bellman equation.
    """
    iteration = 0
    action_prob = 1.0 / len(actions)
    
    while True:
        delta = 0  # Track maximum change in values
        V_new = V.copy()  # Create a copy of current value function
        
        # Iterate over all states
        for row in range(grid_size):
            for col in range(grid_size):
                state = (row, col)
                
                # Skip terminal state
                if state == terminal_state:
                    continue
                
                # Compute new value using Bellman equation
                v = 0
                
                for action_name, action_delta in actions.items():
                    # Get next state
                    next_state = get_next_state(state, action_delta, grid_size)
                    next_row, next_col = next_state
                    
                    # Reward for taking this action (moving)
                    reward = MOVE_REWARD
                    
                    # Expected value for this action
                    v += action_prob * (reward + gamma * V[next_row, next_col])
                
                # Update value function
                V_new[row, col] = v
                
                # Track maximum change
                delta = max(delta, abs(V_new[row, col] - V[row, col]))
        
        # Update value function
        V = V_new.copy()
        iteration += 1
        
        # Check for convergence
        if delta < theta:
            print(f"\nConverged after {iteration} iterations")
            print(f"Final delta: {delta:.6f}")
            break
        
        # Print progress every 100 iterations
        if iteration % 100 == 0:
            print(f"Iteration {iteration}: delta = {delta:.6f}")
    
    return V, iteration

# Run value iteration
print("\n" + "="*50)
print("RUNNING VALUE ITERATION")
print("="*50)
V_final, num_iterations = value_iteration(V, gamma, theta, terminal_state, actions, N)

# Display results
print("\n" + "="*50)
print("FINAL VALUE FUNCTION")
print("="*50)
print(V_final)
print("\n" + "="*50)

# Analysis
print("\nANALYSIS")
print("="*50)
print(f"Grid Size: {N}x{N}")
print(f"Total States: {N*N}")
print(f"Iterations to Converge: {num_iterations}")
print(f"Discount Factor (gamma): {gamma}")
print(f"Convergence Threshold (theta): {theta}")
print(f"\nValue at Start State (0,0): {V_final[0, 0]:.2f}")
print(f"Value at Terminal State ({N-1},{N-1}): {V_final[N-1, N-1]:.2f}")
print(f"\nMinimum Value: {np.min(V_final):.2f}")
print(f"Maximum Value: {np.max(V_final):.2f}")

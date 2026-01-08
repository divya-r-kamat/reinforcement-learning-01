
# Deep Q-Network (DQN) for Autonomous Navigation

This project implements an autonomous navigation system using **Deep Q-Learning (DQN)** with **Replay Buffer** to train an agent car to navigate through a complex urban city environment and reach multiple sequential targets efficiently.

The agent learns directly from sensor data and target-relative information, enabling intelligent decision-making in a realistic, obstacle-rich city map inspired by Paris.

- **Deep Q-Network (DQN)** with 5 fully connected layers  
- **Multi-Target Navigation:** Sequential goal reaching (A1 → A2 → A3)  
- **Custom City Map:** Paris-inspired urban layout  
- **Prioritized Experience Replay:** Intelligent memory sampling  
- **Real-Time Visualization:** Live sensor data, rewards, and agent movement  
- **Professional UI:** Nordic-themed PyQt6 interface  


---

## What is Deep Reinforcement Learning?
Reinforcement Learning (RL) is a machine learning paradigm where an agent learns to make decisions by interacting with an environment. At each time step:
1. The agent observes the current state  
2. Selects an action  
3. Receives a reward  
4. Transitions to a new state  

The goal is to learn a policy that maximizes cumulative long-term reward.

---

## Why Deep Q-Learning?
Traditional Q-Learning becomes infeasible in large or continuous state spaces. **Deep Q-Networks (DQN)** solve this problem by using a neural network to approximate the Q-value function:

```
Q(s, a) ≈ Qθ(s, a)
```

This allows learning directly from raw or high-dimensional inputs, such as sensor readings.

---

## System Overview
The agent represents a vehicle navigating a city map using distance sensors and target-relative features. It must:
- Avoid obstacles (buildings, river)  
- Stay on the road  
- Reach three targets sequentially  
- Learn optimal navigation strategies through trial and error  

---

## Neural Network Architecture
The DQN consists of **five fully connected (FC) layers**, designed to balance expressiveness and computational efficiency.

**Network Design:**
```
Input Layer (9 neurons)
        ↓
Hidden Layer 1 (128 neurons) + ReLU
        ↓
Hidden Layer 2 (256 neurons) + ReLU
        ↓
Hidden Layer 3 (256 neurons) + ReLU
        ↓
Hidden Layer 4 (128 neurons) + ReLU
        ↓
Output Layer (5 neurons)
```

---

### Input Features (9 Dimensions)
1–7. Seven distance sensors (angles: −45°, −30°, −15°, 0°, 15°, 30°, 45°)  
8. Normalized angle to the current target  
9. Normalized distance to the current target  

---

### Output Actions (5 Discrete Actions)
| Action Index | Description            |
|-------------:|------------------------|
| 0            | Turn Left (5°)        |
| 1            | Go Straight           |
| 2            | Turn Right (5°)       |
| 3            | Sharp Left Turn (20°) |
| 4            | Sharp Right Turn (20°)|

---

## Prioritized Experience Replay
Instead of sampling experiences uniformly, the system uses **Prioritized Experience Replay** to improve learning efficiency.

**Memory Buffers:**
- **Priority Memory (3,000 capacity):** Stores successful experiences  
- **Regular Memory (10,000 capacity):** Stores all other experiences  

**Dynamic Sampling Strategy:**
```python
success_rate = len(priority_memory) / total_memory
priority_ratio = 0.3 + (success_rate * 0.4)

priority_samples = BATCH_SIZE * priority_ratio
regular_samples = BATCH_SIZE * (1 - priority_ratio)
```
Between **30% and 70%** of each training batch comes from successful experiences.

---

## Training Hyperparameters
| Parameter       | Value    | Description                      |
|-----------------|---------:|---------------------------------|
| BATCH_SIZE      | 256      | Training batch size            |
| GAMMA (γ)       | 0.9      | Discount factor                |
| LEARNING_RATE   | 0.0001   | Adam optimizer learning rate   |
| TAU (τ)         | 0.001    | Soft target network update     |
| EPSILON_DECAY   | 0.9998   | Exploration decay rate         |
| MIN_EPSILON     | 0.01     | Minimum exploration            |
| MAX_CRASHES     | 2        | Reset threshold                |
| SENSOR_DIST     | 15       | Sensor range (pixels)          |
| SPEED           | 5        | Forward speed                  |
| TURN_SPEED      | 5°       | Regular turn angle             |
| SHARP_TURN      | 20°      | Sharp turn angle               |

---

## 🗺 City Map Design
**Map Specifications:**
- Dimensions: 1000 × 800 pixels  
- Theme: Paris-inspired urban environment
  
<img width="607" height="610" alt="Screenshot 2025-12-17 193704" src="https://github.com/user-attachments/assets/25223ba0-e79f-43ad-b4cd-4429e21f0040" />

**Environment Features:**
- Dense street networks  
- River (Seine representation)  
- Parks and green spaces  
- Urban blocks  
- Roundabouts and intersections  

**Target Configuration:**
- 3 Targets - A1 , A2 , A3
Upon reaching a target:
- The next target becomes active  
- A **+100 reward bonus** is given  

---

### Visual Semantics of the Map
| Color / Feature | Meaning            |
|-----------------|--------------------|
| White roads     | Navigable areas   |
| Dark buildings  | Obstacles         |
| Blue/Green      | Parks (navigable) |
| Navy blue       | River (obstacle)  |

---

## Reward System
The reward function balances goal completion, safety, and efficiency.

| Event           | Reward  | Description             |
|-----------------|--------:|-------------------------|
| Target reached  | +100    | Successful checkpoint  |
| Wall collision  | −100    | Crash                  |
| Forward progress| +20     | Staying on road        |
| Moving closer   | 0       | Reduced distance       |
| Moving away     | −10     | Increased distance     |
| Time step       | −0.1    | Encourages efficiency  |

**Collision Detection Logic:**
```python
if brightness < 0.4:
    reward = -100
    episode_done = True
```


---

## Running the Application
```bash
python citymap.py
```

---

## Usage Instructions
1. **Launch Application:** Run `python citymap.py`  
2. **Set Start Position:** Click on map to place car
3. **Add Targets:** Click map to add Target A1, A2, A3  
4. **Finalize Setup:** Right-click to confirm targets  
5. **Start Training:** Press `SPACE` or click **START** button  
6. **Monitor Progress:** Watch reward chart and console logs  
7. **Pause/Resume:** Press `SPACE` anytime during training  
---

## Video Demonstration

Youtube Link : https://youtu.be/DWugeb-6NTE

---

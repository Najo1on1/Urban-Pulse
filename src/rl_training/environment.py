import gymnasium as gym
from gymnasium import spaces
import numpy as np
import logging

class UrbanTrafficEnv(gym.Env):
    """
    A custom RL environment simulating urban traffic intersections.
    Goal: Maximize throughput and minimize congestion using PPO.
    """
    def __init__(self, num_intersections=4, max_capacity=100):
        super(UrbanTrafficEnv, self).__init__()
        
        self.num_intersections = num_intersections
        self.max_capacity = max_capacity
        self.current_step = 0
        self.max_steps = 200 # An episode represents a rush hour period
        
        # ACTION SPACE: 
        # For each intersection, the agent chooses a light phase.
        # 0 = North/South Green, 1 = East/West Green
        # Using MultiBinary allows the agent to flip switches independently
        self.action_space = spaces.MultiBinary(self.num_intersections)
        
        # OBSERVATION SPACE:
        # What the agent "sees": The number of cars waiting at each of the 4 approaches 
        # (North, South, East, West) for every intersection.
        # Shape: (num_intersections, 4)
        self.observation_space = spaces.Box(
            low=0, 
            high=self.max_capacity, 
            shape=(self.num_intersections * 4,), 
            dtype=np.float32
        )
        
        # Initialize the state (empty roads)
        self.state = np.zeros((self.num_intersections, 4), dtype=np.float32)

    def reset(self, seed=None, options=None):
        """Resets the environment for a new training episode."""
        super().reset(seed=seed)
        self.current_step = 0
        # Start with a random amount of traffic to prevent the agent from memorizing one scenario
        self.state = np.random.randint(5, 20, size=(self.num_intersections, 4)).astype(np.float32)
        return self.state.flatten(), {}

    def step(self, action): 
        """
        Executes one time-step within the environment based on the Agent's action.
        action array: e.g., [1, 0, 0, 1] (Intersections 1 & 4 are E/W green, 2 & 3 are N/S green)
        """
        self.current_step += 1
        reward = 0
        
        # 1. Process Traffic based on the Light Phases
        for i in range(self.num_intersections):
            light_phase = action[i]
            
            if light_phase == 0: # North/South Green
                # N/S cars move (decrease queue), E/W cars stop (queue stays/grows)
                cars_moving = np.sum(self.state[i, 0:2]) # N and S approaches
                self.state[i, 0:2] = np.maximum(0, self.state[i, 0:2] - 5) # 5 cars pass per step
            else: # East/West Green
                cars_moving = np.sum(self.state[i, 2:4]) # E and W approaches
                self.state[i, 2:4] = np.maximum(0, self.state[i, 2:4] - 5)
                
            # 2. Reward Function
            # +1 for moving cars, -0.5 for cars stuck at red lights
            stuck_cars = np.sum(self.state[i]) 
            reward += (cars_moving * 1.0) - (stuck_cars * 0.5)

        # 3. Inject new traffic (Simulating incoming demand)
        incoming_traffic = np.random.randint(0, 4, size=(self.num_intersections, 4))
        self.state += incoming_traffic
        self.state = np.clip(self.state, 0, self.max_capacity) # Cap at physical road limit

        # 4. Check if the episode is over
        terminated = False
        truncated = self.current_step >= self.max_steps
        
        # Catastrophic failure condition: Gridlock
        if np.mean(self.state) > (self.max_capacity * 0.9):
            reward -= 1000 # Massive penalty for causing gridlock
            terminated = True
            
        return self.state.flatten(), float(reward), terminated, truncated, {}

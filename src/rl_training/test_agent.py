import time
import logging
from stable_baselines3 import PPO
from src.rl_training.environment import UrbanTrafficEnv

logging.basicConfig(level=logging.INFO, format='%(message)s')

def print_intersection(step, state, action, reward):
    print(f"\n--- TIME STEP {step} ---")
    print(f"Reward this step: {reward:.2f}")
    
    for i in range(len(action)):
        # Because the state is flattened (16 items), Intersection 'i' is at index i*4
        idx = i * 4
        n, s, e, w = state[idx:idx+4]
        
        # 0 = N/S Green, 1 = E/W Green
        light = "🟢 N/S Green | 🔴 E/W Red" if action[i] == 0 else "🔴 N/S Red  | 🟢 E/W Green"
        
        print(f"Intersection {i+1}: {light}")
        print(f"  Queues -> North: {int(n):02d}, South: {int(s):02d}, East: {int(e):02d}, West: {int(w):02d}")

def main():
    # 1. Load the Environment
    env = UrbanTrafficEnv()
    
    # 2. Load the Trained Agent
    model_path = "models/ppo_traffic/ppo_agent_v1"
    print(f"Loading trained agent from {model_path}.zip...\n")
    model = PPO.load(model_path)

    # 3. Start the Live Test
    obs, _ = env.reset()
    print("🚦 STARTING LIVE TRAFFIC INFERENCE 🚦")
    
    total_reward = 0
    # We will just run 10 steps so it's easy to read in the terminal
    for step in range(1, 11):
        # deterministic=True forces the agent to use its optimal learned policy
        action, _states = model.predict(obs, deterministic=True)
        
        # Take the action in the environment
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        print_intersection(step, obs, action, reward)
        
        # Pause for 1.5 seconds so you can watch the AI "think" in real-time
        time.sleep(1.5)
        
        if terminated or truncated:
            print("\n🚨 Gridlock reached or episode ended!")
            break
            
    print(f"\n🏁 TEST COMPLETE. Total Reward for 10 steps: {total_reward:.2f}")

if __name__ == "__main__":
    main()

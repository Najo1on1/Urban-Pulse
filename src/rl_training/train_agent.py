import os
import logging
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env

# Import your custom environment
from src.rl_training.environment import UrbanTrafficEnv

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def main():
    # 1. Directory Setup
    models_dir = "models/ppo_traffic"
    log_dir = "lightning_logs/rl_tensorboard"
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # 2. Environment Initialization (Check once, then vectorize)
    logging.info("Verifying Urban Traffic Environment...")
    check_env(UrbanTrafficEnv(num_intersections=20), warn=True)
    
    # Spawn exactly 10 workers to match your .wslconfig limit
    # make_vec_env automatically adds the Monitor wrapper to all 10 clones
    logging.info("Spawning 10 Parallel Environments...")
    vec_env = make_vec_env(
        lambda: UrbanTrafficEnv(num_intersections=20), 
        n_envs=10, 
        vec_env_cls=SubprocVecEnv
    )

    # 3. Initialize the PPO Agent
    logging.info("Spawning PPO Agent on CPU for maximum MLP performance...")
    model = PPO(
        "MlpPolicy", 
        vec_env, 
        verbose=1, 
        learning_rate=0.0003,
        n_steps=2048, 
        batch_size=64,
        tensorboard_log=log_dir,
        device="cpu" 
    )

    # 4. The Training Loop
    TIMESTEPS = 250000 
    logging.info(f"Starting Training for {TIMESTEPS} timesteps...")
    
    model.learn(total_timesteps=TIMESTEPS, tb_log_name="PPO_Phase3")

    # 5. Save the Agent
    model_path = f"{models_dir}/ppo_agent_20_nodes"
    model.save(model_path)
    logging.info(f"Training Complete. Agent saved to {model_path}.zip")

if __name__ == "__main__":
    main()
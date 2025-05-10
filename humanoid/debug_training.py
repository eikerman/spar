import os
import time
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

from humanoid_env import HumanoidEnv


def make_env():
    """
    Create a function that returns a new instance of the environment
    """

    def _init():
        env = HumanoidEnv(render=True)  # Use GUI for debugging
        return env

    return _init


def train_single_env(total_timesteps=10000):
    """
    Train on a single environment for debugging
    """
    # Create and wrap environment with DummyVecEnv directly
    env = DummyVecEnv([make_env()])

    # Create log directory
    log_dir = "logs/debug/"
    os.makedirs(log_dir, exist_ok=True)

    # Create model with simpler parameters
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=log_dir,
        learning_rate=3e-4,
        n_steps=1024,  # Smaller batch for debugging
        batch_size=64,
        n_epochs=5,
    )

    # Create a simple callback to save the model
    class SimpleCallback(BaseCallback):
        def __init__(self, verbose=0):
            super(SimpleCallback, self).__init__(verbose)

        def _on_step(self):
            # Just log progress occasionally
            if self.n_calls % 1000 == 0 and self.verbose > 0:
                print(f"Training progress: {self.n_calls}/{total_timesteps} steps")
            return True

    callback = SimpleCallback(verbose=1)

    print("Starting training...")
    # Train for a few steps
    model.learn(total_timesteps=total_timesteps, callback=callback)

    # Save model
    model.save(f"{log_dir}/debug_model")

    print("Debug training completed!")

    return model, env


def test_debug_model(model, env, n_steps=1000):
    """
    Test the model for a few steps
    """
    obs = env.reset()

    for i in range(n_steps):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = env.step(action)

        if i % 100 == 0:
            print(f"Step {i}, reward: {rewards[0]}")

        if dones[0]:
            print("Episode done, resetting")
            obs = env.reset()

        time.sleep(0.01)  # Slow down for visualization

    env.close()


if __name__ == "__main__":
    try:
        # Train for just a few steps to debug
        print("Initializing training...")
        model, env = train_single_env(total_timesteps=10000)

        # Test the trained model
        print("Testing trained model...")
        test_debug_model(model, env)

    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback

        traceback.print_exc()
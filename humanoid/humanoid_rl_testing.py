import time
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from humanoid_env import HumanoidEnv


def make_env(render=True):
    """Create environment creation function"""

    def _init():
        return HumanoidEnv(render=render)

    return _init


def test_trained_model(model_path, render=True, n_episodes=5):
    """
    Test a trained model
    """
    # Create environment wrapped in DummyVecEnv
    env = DummyVecEnv([make_env(render=render)])

    try:
        # Load the trained model
        model = PPO.load(model_path)
        print(f"Successfully loaded model from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Run test episodes
    for episode in range(n_episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        step = 0

        print(f"Episode {episode + 1}")

        while not done:
            # Get model's action
            action, _ = model.predict(obs, deterministic=True)

            # Apply action
            obs, reward, done, info = env.step(action)

            total_reward += reward[0]
            step += 1

            # Slow down visualization if rendering
            if render:
                time.sleep(0.01)

            # Print progress every 100 steps
            if step % 100 == 0:
                # Get the base position from the first environment
                if hasattr(env.envs[0], 'get_humanoid_position'):
                    pos, _ = env.envs[0].get_humanoid_position()
                    print(f"Step {step}: Position = {pos}, Reward = {reward[0]:.2f}, Total = {total_reward:.2f}")

            # End after max steps or if done
            if done[0] or step >= 1000:
                break

        # Episode summary
        if hasattr(env.envs[0], 'get_humanoid_position'):
            pos, _ = env.envs[0].get_humanoid_position()
            print(f"Episode {episode + 1} finished after {step} steps with total reward {total_reward:.2f}")
            print(f"Final position: x={pos[0]:.2f}, y={pos[1]:.2f}, z={pos[2]:.2f}")
        print("-" * 50)

    # Close environment
    env.close()


def manual_control_test():
    """
    Test the environment with manual control to verify it works
    """
    # Create environment directly without vectorization
    env = HumanoidEnv(render=True)
    env.reset()

    print("Manual control test. Press Ctrl+C to exit.")

    try:
        while True:
            # Generate random actions
            action = np.random.uniform(-1, 1, size=env.action_space.shape)

            # Take step
            obs, reward, done, info = env.step(action)

            # Print info
            if done:
                print("Episode finished, resetting...")
                env.reset()

            time.sleep(0.01)
    except KeyboardInterrupt:
        print("Manual test interrupted.")

    env.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["test", "manual"], default="test",
                        help="Test mode: 'test' for testing trained model, 'manual' for manual control")
    parser.add_argument("--model", type=str, default="logs/humanoid_ppo/humanoid_final.zip",
                        help="Path to the trained model")
    parser.add_argument("--episodes", type=int, default=5,
                        help="Number of test episodes")

    args = parser.parse_args()

    try:
        if args.mode == "test":
            test_trained_model(args.model, render=True, n_episodes=args.episodes)
        else:
            manual_control_test()
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
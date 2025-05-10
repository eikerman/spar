import time
import numpy as np
import pybullet as p

from humanoid_env import HumanoidEnv


def test_environment_only():
    """
    Test just the environment functionality without RL
    """
    print("Testing basic environment functionality...")

    try:
        # Create the environment
        env = HumanoidEnv(render=True)
        print("Environment created successfully")

        # Test reset
        obs = env.reset()
        print(f"Reset successful. Observation shape: {obs.shape}")

        # Test random actions
        for i in range(500):  # Run 500 steps with random actions
            action = np.random.uniform(-1, 1, size=env.action_space.shape)
            obs, reward, done, info = env.step(action)

            if i % 50 == 0:  # Print info every 50 steps
                pos, orn = env.get_humanoid_position()
                print(f"Step {i}:")
                print(f"  Position: {pos}")
                print(f"  Reward: {reward}")
                print(f"  Done: {done}")

            if done:
                print("Episode terminated, resetting...")
                obs = env.reset()

            time.sleep(0.01)  # Slow down for visualization

        # Close the environment
        env.close()
        print("Environment test completed successfully")

    except Exception as e:
        print(f"Error testing environment: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_environment_only()
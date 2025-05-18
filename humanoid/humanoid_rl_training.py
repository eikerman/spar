import os
import time
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed

from humanoid_env import HumanoidEnv


# Custom callback for logging during training
class TensorboardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)

    def _on_step(self):
        try:
            # Log additional info every 1000 steps
            if self.n_calls % 1000 == 0:
                # For vectorized environments
                if hasattr(self.training_env.venv, 'envs'):
                    # Get the first environment
                    env = self.training_env.venv.envs[0]
                    if hasattr(env, 'get_humanoid_position'):
                        pos, _ = env.get_humanoid_position()
                        self.logger.record('humanoid/x_position', pos[0])
        except Exception as e:
            if self.verbose > 0:
                print(f"Warning in callback: {e}")
        return True


def make_env(rank, seed=0):
    """
    Utility function for multiprocessed env.
    """

    def _init():
        try:
            env = HumanoidEnv(render=False)  # Only render first environment
            env.seed(seed + rank)
            return env
        except Exception as e:
            print(f"Error initializing environment {rank}: {e}")
            raise

    set_random_seed(seed)
    return _init


def train_humanoid(total_timesteps=2000000, n_envs=4):
    """
    Train the humanoid using PPO algorithm
    """
    # Create log directory
    log_dir = "logs/humanoid_ppo/"
    os.makedirs(log_dir, exist_ok=True)

    # Create vectorized environments
    env_fns = [make_env(i) for i in range(n_envs)]

    if n_envs == 1:
        env = DummyVecEnv(env_fns)
    else:
        try:
            env = SubprocVecEnv(env_fns)
        except Exception as e:
            print(f"Failed to create SubprocVecEnv: {e}. Falling back to DummyVecEnv.")
            env = DummyVecEnv(env_fns)

    # Initialize model
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=log_dir,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
    )

    # Create a custom checkpoint callback
    class CheckpointCallback(BaseCallback):
        def __init__(self, save_freq, save_path, name_prefix="model", verbose=0):
            super(CheckpointCallback, self).__init__(verbose)
            self.save_freq = save_freq
            self.save_path = save_path
            self.name_prefix = name_prefix
            self.last_save_step = 0

        def _on_step(self):
            if self.n_calls - self.last_save_step >= self.save_freq:
                path = f"{self.save_path}/{self.name_prefix}_{self.n_calls}"
                self.model.save(path)
                if self.verbose > 0:
                    print(f"Saved model checkpoint to {path}")
                self.last_save_step = self.n_calls
            return True

    # Create our callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=total_timesteps // 10,  # Save 10 times during training
        save_path=log_dir,
        name_prefix="humanoid_checkpoint",
        verbose=1
    )

    # Custom logger callback
    tensorboard_callback = TensorboardCallback(verbose=1)

    # Train model
    start_time = time.time()

    try:
        print(f"Starting training for {total_timesteps} steps...")

        # Use both callbacks together
        model.learn(
            total_timesteps=total_timesteps,
            callback=[checkpoint_callback, tensorboard_callback]
        )

    except Exception as e:
        print(f"Error during training: {e}")
        # Save what we have so far
        model.save(f"{log_dir}/humanoid_emergency_save")
        print("Saved emergency model checkpoint")

    # Save final model
    model.save(f"{log_dir}/humanoid_final")

    print(f"Training completed in {(time.time() - start_time) / 60:.2f} minutes")

    return model, env


def simple_evaluation(model, env, n_eval_episodes=3):
    """
    Simple evaluation of a trained model
    """
    print("Running simple evaluation...")

    for episode in range(n_eval_episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        step = 0

        print(f"Episode {episode + 1}")

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)

            total_reward += reward[0]
            step += 1

            if done[0] or step >= 1000:
                break

            if step % 100 == 0:
                print(f"Step {step}: Reward = {reward[0]:.2f}, Total = {total_reward:.2f}")

        print(f"Episode {episode + 1}: Total reward: {total_reward:.2f}, Steps: {step}")

    return total_reward



if __name__ == "__main__":
    try:
        import argparse

        parser = argparse.ArgumentParser(description='Train the humanoid to walk')
        parser.add_argument('--steps', type=int, default=500000,
                            help='Total timesteps for training')
        parser.add_argument('--envs', type=int, default=1,
                            help='Number of parallel environments')
        parser.add_argument('--lr', type=float, default=3e-4,
                            help='Learning rate')
        parser.add_argument('--no-eval', action='store_true',
                            help='Skip evaluation after training')

        args = parser.parse_args()

        print(f"Starting training with {args.envs} environments for {args.steps} steps")
        print(f"Learning rate: {args.lr}")

        # Update PPO's learning rate
        from stable_baselines3.common.callbacks import ProgressBarCallback

        # Train with the specified parameters
        model, env = train_humanoid(
            total_timesteps=args.steps,
            n_envs=args.envs
        )

        # Simple evaluation if not skipped
        if not args.no_eval:
            simple_evaluation(model, env)

        # Close environment
        env.close()

    except Exception as e:
        print(f"Error in main: {e}")
        import traceback

        traceback.print_exc()
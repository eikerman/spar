import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import pybullet as p
import pybullet_data
import time
import random
import os

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Helper functions to handle arrays without NumPy
def to_list(tensor):
    """Convert PyTorch tensor to list"""
    return tensor.cpu().tolist()


def from_list(lst, dtype=torch.float32):
    """Convert list to PyTorch tensor"""
    return torch.tensor(lst, dtype=dtype)


class SwordFighterEnv:
    """Environment for sword fighting simulation using PyBullet"""

    def _orient_toward_opponent(self, body_id, current_pos, target_pos, assist_strength=1.0):
        """Apply torque to orient the fighter toward the opponent"""
        # Calculate direction to target
        direction = [target_pos[0] - current_pos[0], target_pos[1] - current_pos[1], 0]

        # Skip if too close (avoid division by zero)
        dist = (direction[0] ** 2 + direction[1] ** 2) ** 0.5
        if dist < 0.001:
            return

        # Normalize direction
        direction = [direction[0] / dist, direction[1] / dist, 0]

        # Get current orientation
        _, orn = p.getBasePositionAndOrientation(body_id)
        rot_matrix = p.getMatrixFromQuaternion(orn)

        # Extract current forward vector (x-axis from the rotation matrix)
        forward = [rot_matrix[0], rot_matrix[3], rot_matrix[6]]

        # Project to XY plane and normalize
        forward[2] = 0
        forward_norm = (forward[0] ** 2 + forward[1] ** 2) ** 0.5

        if forward_norm < 0.001:
            return

        forward = [forward[0] / forward_norm, forward[1] / forward_norm, 0]

        # Calculate the cross product to find the rotation axis
        cross_product = [
            forward[1] * direction[2] - forward[2] * direction[1],
            forward[2] * direction[0] - forward[0] * direction[2],
            forward[0] * direction[1] - forward[1] * direction[0]
        ]

        # Calculate dot product to find the angle
        dot_product = forward[0] * direction[0] + forward[1] * direction[1] + forward[2] * direction[2]

        # Only apply torque if not already aligned
        if dot_product < 0.99:
            # Apply torque proportional to the cross product and scaled by assist_strength
            torque_strength = 5.0 * assist_strength * self.orientation_stabilization_strength

            # Apply torque only around z-axis for 2D rotation
            p.applyExternalTorque(
                body_id, -1,
                torqueObj=[0, 0, cross_product[2] * torque_strength],
                flags=p.WORLD_FRAME
            )

    def __init__(self, render=True):
        # Initialize PyBullet
        self.render_mode = render
        if render:
            p.connect(p.GUI)  # For visualization
        else:
            p.connect(p.DIRECT)  # Headless mode for faster training

        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # Set gravity to zero for floating effect
        p.setGravity(0, 0, 0)

        # Load ground plane (kept for reference)
        self.plane_id = p.loadURDF("plane.urdf")

        # Create the fighters
        self.fighter1 = self._create_fighter(position=[0, 1, 1])
        self.fighter2 = self._create_fighter(position=[0, -1, 1])

        # Define target height for floating
        self.target_height = 1.0

        # Simplified action space - 3 dimensions for x, y, rotation movement
        self.action_space_size = 3

        # Actual observation size from _get_observation method
        test_obs = self._get_observation()
        self.observation_space_size = len(test_obs)
        print(f"Actual observation space size: {self.observation_space_size}")
        print(f"Action space size: {self.action_space_size}")

        # Episode parameters
        self.max_episode_steps = 500
        self.current_step = 0

        # Debug parameters for stabilization adjustment
        self.height_stabilization_strength = 100.0
        self.orientation_stabilization_strength = 200.0

    def _create_fighter(self, position):
        """Create a simplified skeleton fighter with a sword"""

        # For this example, we'll use a simplified approach with primitive shapes
        # Create main body (torso)
        torso = p.createMultiBody(
            baseMass=10.0,
            basePosition=position,
            baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.2, 0.1, 0.3]),
            baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=[0.2, 0.1, 0.3],
                                                     rgbaColor=[0.7, 0.7, 0.7, 1])
        )

        # Sword
        sword_length = 1.0
        sword = p.createMultiBody(
            baseMass=1.0,
            basePosition=[position[0] + 0.5, position[1], position[2]],
            baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=[sword_length / 2, 0.05, 0.01]),
            baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=[sword_length / 2, 0.05, 0.01],
                                                     rgbaColor=[0.8, 0.8, 0.8, 1])
        )

        # Create constraint to attach sword to torso
        p.createConstraint(
            torso, -1, sword, -1, p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=[0.3, 0, 0],
            childFramePosition=[-sword_length / 2 - 0.1, 0, 0]
        )

        return {
            "torso": torso,
            "sword": sword,
            "joints": []  # Would store joint IDs in full implementation
        }

    def reset(self):
        """Reset the environment for a new episode"""
        p.resetBasePositionAndOrientation(self.fighter1["torso"], [0, 1, self.target_height], [0, 0, 0, 1])
        p.resetBasePositionAndOrientation(self.fighter2["torso"], [0, -1, self.target_height], [0, 0, 0, 1])

        # Reset velocities to zero
        p.resetBaseVelocity(self.fighter1["torso"], [0, 0, 0], [0, 0, 0])
        p.resetBaseVelocity(self.fighter2["torso"], [0, 0, 0], [0, 0, 0])

        self.current_step = 0
        return self._get_observation()

    def _get_observation(self):
        """Get the current state observation"""
        # For fighter 1
        pos1, orn1 = p.getBasePositionAndOrientation(self.fighter1["torso"])
        vel1, ang_vel1 = p.getBaseVelocity(self.fighter1["torso"])

        # For fighter 2
        pos2, orn2 = p.getBasePositionAndOrientation(self.fighter2["torso"])
        vel2, ang_vel2 = p.getBaseVelocity(self.fighter2["torso"])

        # Combine observations (positions, orientations, velocities)
        observation = list(pos1) + list(orn1) + list(vel1) + list(ang_vel1) + \
                      list(pos2) + list(orn2) + list(vel2) + list(ang_vel2)

        return observation

    def step(self, action):
        """Execute one step in the environment"""
        # Convert action to list if it's not already
        if isinstance(action, torch.Tensor):
            action = to_list(action)

        # Make sure we have the right number of action dimensions
        if len(action) > self.action_space_size:
            action = action[:self.action_space_size]
        while len(action) < self.action_space_size:
            action.append(0.0)  # Pad with zeros if needed

        # Get current position and orientation
        pos1, orn1 = p.getBasePositionAndOrientation(self.fighter1["torso"])
        pos2, orn2 = p.getBasePositionAndOrientation(self.fighter2["torso"])

        # Simplified movement - only in x, y plane with rotation
        # Scale actions for better control
        move_scale = 20.0

        # Apply force in x and y directions
        p.applyExternalForce(
            self.fighter1["torso"], -1,
            forceObj=[action[0] * move_scale, action[1] * move_scale, 0],
            posObj=[0, 0, 0],
            flags=p.LINK_FRAME
        )

        # Apply torque for rotation around z-axis
        rotation_scale = 10.0
        p.applyExternalTorque(
            self.fighter1["torso"], -1,
            torqueObj=[0, 0, action[2] * rotation_scale],
            flags=p.LINK_FRAME
        )

        # Apply height stabilization force to maintain floating effect
        self._apply_height_stabilization(self.fighter1["torso"], self.target_height)

        # Apply torque to face opponent for player-controlled fighter (if action[2] is small)
        # Only assist with orientation when the player isn't actively rotating
        if abs(action[2]) < 0.2:
            self._orient_toward_opponent(self.fighter1["torso"], pos1, pos2, assist_strength=0.2)

        # Basic AI for opponent (fighter2) - always move toward fighter1
        self._control_opponent()

        # Step the simulation
        p.stepSimulation()
        if self.render_mode:
            time.sleep(1. / 240.)  # Adjust visualization speed

        # Get new observation
        observation = self._get_observation()

        # Calculate reward
        reward = self._calculate_reward()

        # Check if episode is done
        self.current_step += 1
        done = self.current_step >= self.max_episode_steps

        # Additional info
        info = {}

        return observation, reward, done, info

    def _apply_height_stabilization(self, body_id, target_height):
        """Apply forces to maintain a stable height and upright orientation"""
        pos, orn = p.getBasePositionAndOrientation(body_id)
        vel, ang_vel = p.getBaseVelocity(body_id)

        # PD controller for height
        height_error = target_height - pos[2]
        damping = -vel[2] * 5.0  # Damping coefficient

        # Calculate stabilization force with PD control
        k_p = self.height_stabilization_strength  # Proportional gain
        k_d = self.height_stabilization_strength * 0.2  # Derivative gain

        force_z = k_p * height_error + k_d * damping

        # Apply the force
        p.applyExternalForce(
            body_id, -1,
            forceObj=[0, 0, force_z],
            posObj=[0, 0, 0],
            flags=p.WORLD_FRAME
        )

        # Get the current rotation matrix from quaternion
        rot_matrix = p.getMatrixFromQuaternion(orn)

        # Extract up vector (z-axis) from rotation matrix
        up_x = rot_matrix[2]
        up_y = rot_matrix[5]
        up_z = rot_matrix[8]

        # Calculate error from upright position (should be pointing up)
        # For upright orientation, up vector should be [0, 0, 1]
        upright_error_x = 0 - up_x
        upright_error_y = 0 - up_y
        upright_error_z = 1 - up_z

        # Apply torque to correct orientation
        k_rot = self.orientation_stabilization_strength  # Rotational gain
        k_damp = self.orientation_stabilization_strength * 0.05  # Damping

        torque_x = k_rot * upright_error_y - k_damp * ang_vel[0]  # Cross product for rotation
        torque_y = k_rot * -upright_error_x - k_damp * ang_vel[1]  # Cross product for rotation

        # Apply stabilizing torque to keep upright
        p.applyExternalTorque(
            body_id, -1,
            torqueObj=[torque_x, torque_y, 0],  # No torque on z-axis to allow rotation in that plane
            flags=p.WORLD_FRAME
        )

    def _control_opponent(self):
        """Simple AI for opponent (fighter2)"""
        # Get positions and orientations of both fighters
        pos1, orn1 = p.getBasePositionAndOrientation(self.fighter1["torso"])
        pos2, orn2 = p.getBasePositionAndOrientation(self.fighter2["torso"])

        # Calculate direction vector to fighter1
        direction = [pos1[0] - pos2[0], pos1[1] - pos2[1], 0]

        # Normalize direction
        distance = (direction[0] ** 2 + direction[1] ** 2) ** 0.5
        if distance > 0.001:  # Avoid division by zero
            direction = [direction[0] / distance, direction[1] / distance, 0]

        # Apply force to move toward fighter1
        pursue_force = 15.0
        p.applyExternalForce(
            self.fighter2["torso"], -1,
            forceObj=[direction[0] * pursue_force, direction[1] * pursue_force, 0],
            posObj=[0, 0, 0],
            flags=p.WORLD_FRAME
        )

        # Apply height stabilization and orientation stabilization
        self._apply_height_stabilization(self.fighter2["torso"], self.target_height)

        # Apply torque to face fighter1
        self._orient_toward_opponent(self.fighter2["torso"], pos2, pos1)

    def _calculate_reward(self):
        """Calculate the reward based on the current state"""
        reward = 0

        # Get positions and orientations of both fighters
        pos1, orn1 = p.getBasePositionAndOrientation(self.fighter1["torso"])
        pos2, orn2 = p.getBasePositionAndOrientation(self.fighter2["torso"])

        # Check for contacts between sword1 and fighter2
        contact_points = p.getContactPoints(self.fighter1["sword"], self.fighter2["torso"])
        if len(contact_points) > 0:
            # Reward for hitting opponent
            reward += 10.0

        # Check for contacts between sword2 and fighter1
        contact_points = p.getContactPoints(self.fighter2["sword"], self.fighter1["torso"])
        if len(contact_points) > 0:
            # Penalty for getting hit
            reward -= 10.0

        # Distance-based reward to encourage getting closer to opponent
        distance = ((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2) ** 0.5
        proximity_reward = 0.5 * (2.0 - min(2.0, distance))  # Max 1.0 when close
        reward += proximity_reward

        # Small reward for movement (exploration)
        vel, _ = p.getBaseVelocity(self.fighter1["torso"])
        movement_reward = 0.01 * (vel[0] ** 2 + vel[1] ** 2) ** 0.5
        reward += movement_reward

        # Penalty for extreme height deviation
        height_error = abs(pos1[2] - self.target_height)
        if height_error > 0.3:
            reward -= height_error * 2.0

        # Penalty for not being upright
        rot_matrix = p.getMatrixFromQuaternion(orn1)
        up_z = rot_matrix[8]  # z-component of the up vector
        upright_error = abs(1.0 - up_z)
        reward -= upright_error * 2.0

        # Reward for facing the opponent
        # Calculate direction to opponent
        direction_to_opponent = [pos2[0] - pos1[0], pos2[1] - pos1[1], 0]
        direction_norm = (direction_to_opponent[0] ** 2 + direction_to_opponent[1] ** 2) ** 0.5

        if direction_norm > 0.001:
            # Normalize
            direction_to_opponent = [
                direction_to_opponent[0] / direction_norm,
                direction_to_opponent[1] / direction_norm,
                0
            ]

            # Extract forward vector (x-axis) from rotation matrix
            forward_x = rot_matrix[0]
            forward_y = rot_matrix[3]

            # Calculate dot product between forward vector and direction to opponent
            # 1.0 means perfectly facing opponent, -1.0 means facing away
            facing_alignment = forward_x * direction_to_opponent[0] + forward_y * direction_to_opponent[1]

            # Reward for facing opponent (transforms -1 to 1 range to 0 to 1 range)
            facing_reward = (facing_alignment + 1) * 0.5
            reward += facing_reward

        return reward

    def close(self):
        """Clean up resources"""
        p.disconnect()


# The rest of the code (Actor, Critic, PPOBuffer, PPOAgent, train, test)
# remains the same as in the original implementation

class Actor(nn.Module):
    """Policy network for PPO agent"""

    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()

        # Print dimensions for debugging
        print(f"Actor Network - Input: {state_dim}, Output: {action_dim}")

        self.network = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh()
        )

        # Mean and std deviation layers
        self.mean = nn.Linear(32, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, state):
        # Convert to tensor if needed
        if not isinstance(state, torch.Tensor):
            state = from_list(state).to(device)

        if state.dim() == 1:
            # Add batch dimension if missing
            state = state.unsqueeze(0)

        x = self.network(state)
        mean = torch.tanh(self.mean(x))  # Use tanh to bound the actions

        # Use a fixed std with a learnable parameter
        std = torch.exp(self.log_std).expand_as(mean)

        return mean, std


class Critic(nn.Module):
    """Value network for PPO agent"""

    def __init__(self, state_dim):
        super(Critic, self).__init__()

        # Print dimensions for debugging
        print(f"Critic Network - Input: {state_dim}")

        self.network = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )

    def forward(self, state):
        # Convert to tensor if needed
        if not isinstance(state, torch.Tensor):
            state = from_list(state).to(device)

        if state.dim() == 1:
            # Add batch dimension if missing
            state = state.unsqueeze(0)

        return self.network(state)


class PPOBuffer:
    """Simple buffer for storing experiences"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []

    def add(self, state, action, reward, value, log_prob, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)

    def get(self):
        return (
            self.states,
            self.actions,
            self.rewards,
            self.values,
            self.log_probs,
            self.dones
        )

    def size(self):
        return len(self.states)


class PPOAgent:
    """PPO agent for learning sword fighting"""

    def __init__(self, state_dim, action_dim, save_dir="./models_simple"):
        # Hyperparameters
        self.gamma = 0.99
        self.clip_ratio = 0.2
        self.lr = 3e-4
        self.batch_size = 32

        # Print dimensions for debugging
        print(f"PPO Agent - State Dim: {state_dim}, Action Dim: {action_dim}")

        # Initialize networks
        self.actor = Actor(state_dim, action_dim).to(device)
        self.critic = Critic(state_dim).to(device)

        # Initialize optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr)

        # Experience buffer
        self.buffer = PPOBuffer()

        # Save directory
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def act(self, state):
        """Select an action using the current policy"""
        with torch.no_grad():
            # Convert state to tensor if needed
            if not isinstance(state, torch.Tensor):
                state_tensor = from_list(state).to(device)
            else:
                state_tensor = state.to(device)

            if state_tensor.dim() == 1:
                state_tensor = state_tensor.unsqueeze(0)

            # Get action from policy
            mean, std = self.actor(state_tensor)

            # Create normal distribution
            dist = Normal(mean, std)

            # Sample action
            action = dist.sample()

            # Get log probability and value
            log_prob = dist.log_prob(action).sum(dim=-1)
            value = self.critic(state_tensor).squeeze()

        return to_list(action[0]), log_prob.item(), value.item()

    def store_experience(self, state, action, reward, value, log_prob, done):
        """Store experience in buffer"""
        self.buffer.add(state, action, reward, value, log_prob, done)

    def compute_advantages(self, rewards, values, dones, gamma=0.99, lam=0.95):
        """Compute Generalized Advantage Estimation (GAE)"""
        advantages = []
        gae = 0

        # Loop backwards through rewards
        for i in reversed(range(len(rewards) - 1)):
            # Calculate TD error
            delta = rewards[i] + gamma * values[i + 1] * (1 - dones[i]) - values[i]

            # Calculate GAE
            gae = delta + gamma * lam * (1 - dones[i]) * gae
            advantages.insert(0, gae)

        # Return advantages
        if len(advantages) == 0:
            return []
        return advantages

    def learn(self, update_epochs=4):
        """Update policy and value networks"""
        # Get experiences
        states, actions, rewards, values, old_log_probs, dones = self.buffer.get()

        # Early return if buffer is empty
        if len(states) < 2:
            self.buffer.reset()
            return

        # Calculate advantages
        advantages = self.compute_advantages(rewards, values, dones)

        # Convert to tensors
        states_tensor = torch.tensor(states, dtype=torch.float32).to(device)
        actions_tensor = torch.tensor(actions, dtype=torch.float32).to(device)
        old_log_probs_tensor = torch.tensor(old_log_probs, dtype=torch.float32).to(device)
        advantages_tensor = torch.tensor(advantages, dtype=torch.float32).to(device)

        # Calculate returns (advantages + values)
        returns = advantages_tensor + torch.tensor(values[:-1], dtype=torch.float32).to(device)

        # Normalize advantages
        if len(advantages) > 1:
            advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)

        # Update policy for several epochs
        for _ in range(update_epochs):
            # Generate random mini-batch indices
            indices = torch.randperm(len(advantages)).to(device)

            # Process mini-batches
            for start_idx in range(0, len(advantages), self.batch_size):
                # Get mini-batch indices
                batch_indices = indices[start_idx:start_idx + self.batch_size]

                # Get mini-batch data
                mb_states = states_tensor[batch_indices]
                mb_actions = actions_tensor[batch_indices]
                mb_old_log_probs = old_log_probs_tensor[batch_indices]
                mb_advantages = advantages_tensor[batch_indices]
                mb_returns = returns[batch_indices]

                # Calculate current policy probabilities
                mean, std = self.actor(mb_states)
                dist = Normal(mean, std)
                new_log_probs = dist.log_prob(mb_actions).sum(dim=-1)
                entropy = dist.entropy().mean()

                # Get current value estimates
                values = self.critic(mb_states).squeeze()

                # Calculate ratios
                ratios = torch.exp(new_log_probs - mb_old_log_probs)

                # Calculate surrogate losses
                surr1 = ratios * mb_advantages
                surr2 = torch.clamp(ratios, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * mb_advantages

                # Calculate total losses
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = 0.5 * ((values - mb_returns) ** 2).mean()

                # Update actor
                self.actor_optimizer.zero_grad()
                (policy_loss - 0.01 * entropy).backward()
                self.actor_optimizer.step()

                # Update critic
                self.critic_optimizer.zero_grad()
                value_loss.backward()
                self.critic_optimizer.step()

        # Reset buffer
        self.buffer.reset()

    def save_model(self, name):
        """Save the model"""
        try:
            torch.save({
                'actor_state_dict': self.actor.state_dict(),
                'critic_state_dict': self.critic.state_dict(),
                'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
                'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            }, f"{self.save_dir}/{name}.pt")
            print(f"Model saved to {self.save_dir}/{name}.pt")
        except Exception as e:
            print(f"Error saving model: {e}")

    def load_model(self, name):
        """Load the model"""
        try:
            checkpoint = torch.load(f"{self.save_dir}/{name}.pt")
            self.actor.load_state_dict(checkpoint['actor_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            print(f"Model loaded from {self.save_dir}/{name}.pt")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False


def train(num_episodes=100, max_steps=500, render=False):
    """Train the agent"""
    # Create environment
    env = SwordFighterEnv(render=render)

    # Create agent
    agent = PPOAgent(env.observation_space_size, env.action_space_size)

    # Training loop
    total_rewards = []

    try:
        for episode in range(num_episodes):
            # Reset environment
            state = env.reset()
            episode_reward = 0

            for step in range(max_steps):
                # Get action
                action, log_prob, value = agent.act(state)

                # Take step
                next_state, reward, done, _ = env.step(action)

                # Store experience
                agent.store_experience(state, action, reward, value, log_prob, done)

                # Update state and reward
                state = next_state
                episode_reward += reward

                # Learn if buffer is large enough
                if agent.buffer.size() >= 128:
                    agent.learn()

                # Check if episode is done
                if done:
                    break

            # Final learning step
            if agent.buffer.size() > 0:
                agent.learn()

            # Print progress
            total_rewards.append(episode_reward)
            print(f"Episode {episode + 1}/{num_episodes}, Reward: {episode_reward:.2f}")

            # Save model periodically
            if (episode + 1) % 10 == 0:
                agent.save_model(f"model_ep{episode + 1}")

            # Save best model
            if episode == 0 or episode_reward > max(total_rewards[:-1]):
                agent.save_model("best_model")

    except KeyboardInterrupt:
        print("Training interrupted by user")
    except Exception as e:
        print(f"Error during training: {e}")
    finally:
        # Save final model
        agent.save_model("final_model")

        # Close environment
        env.close()

    return total_rewards


def test(episodes=5, model_name="best_model"):
    """Test the agent"""
    # Create environment
    env = SwordFighterEnv(render=True)

    # Create agent
    agent = PPOAgent(env.observation_space_size, env.action_space_size)

    # Load model
    loaded = agent.load_model(model_name)
    if not loaded:
        print("Using untrained model")

    # Testing loop
    total_rewards = []

    try:
        for episode in range(episodes):
            # Reset environment
            state = env.reset()
            episode_reward = 0

            print(f"Starting test episode {episode + 1}/{episodes}")

            # Step through episode
            done = False
            step = 0

            while not done and step < 500:
                # Get action
                action, _, _ = agent.act(state)

                # Take step
                next_state, reward, done, _ = env.step(action)

                # Update state and reward
                state = next_state
                episode_reward += reward
                step += 1

                # Print progress occasionally
                if step % 50 == 0:
                    print(f"  Step {step}, Current reward: {episode_reward:.2f}")

                # Slow down for better visualization
                time.sleep(0.001)

            # Print episode results
            total_rewards.append(episode_reward)
            print(f"Test episode {episode + 1}/{episodes}, Reward: {episode_reward:.2f}")

    except KeyboardInterrupt:
        print("Testing interrupted by user")
    except Exception as e:
        print(f"Error during testing: {e}")
    finally:
        # Close environment
        env.close()

    return total_rewards


if __name__ == "__main__":
    import argparse

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Train the agent")
    parser.add_argument("--test", action="store_true", help="Test the agent")
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes")
    parser.add_argument("--model", type=str, default="best_model", help="Model name")
    args = parser.parse_args()

    # Create models_simple directory
    os.makedirs("./models_simple", exist_ok=True)

    # Run training or testing
    if args.train:
        print("Training agent...")
        train(num_episodes=args.episodes, render=False)

    if args.test:
        print("Testing agent...")
        test(episodes=5, model_name=args.model)

    # Default to training if no arguments are provided
    if not args.train and not args.test:
        print("No arguments provided, running default training...")
        train(num_episodes=args.episodes)
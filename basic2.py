import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import pybullet as p
import pybullet_data
import os
import math

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

    def __init__(self, render=True):
        # Initialize PyBullet
        self.render_mode = render
        if render:
            p.connect(p.GUI)  # For visualization
        else:
            p.connect(p.DIRECT)  # Headless mode for faster training

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)

        p.setPhysicsEngineParameter(fixedTimeStep=1.0 / 240.0)  # Smaller timestep for stability
        p.setPhysicsEngineParameter(numSolverIterations=50)  # More solver iterations
        p.setPhysicsEngineParameter(numSubSteps=10)

        # Load ground plane
        self.plane_id = p.loadURDF("plane.urdf")

        # Create the fighters
        p.changeDynamics(self.plane_id, -1, lateralFriction=1.0, restitution=0.1)

        self.fighter1 = self._create_fighter(position=[0, 1, 1])
        self.fighter2 = self._create_fighter(position=[0, -1, 1])

        # Simplified action space - we'll control joint motors
        self.joints_per_fighter = 8  # Total controllable joints
        self.action_space_size = self.joints_per_fighter  # One action per joint

        # Actual observation size from _get_observation method
        test_obs = self._get_observation()
        self.observation_space_size = len(test_obs)
        print(f"Actual observation space size: {self.observation_space_size}")
        print(f"Action space size: {self.action_space_size}")

        # Episode parameters
        self.max_episode_steps = 500
        self.current_step = 0

    def _create_fighter(self, position):
        """Create a simplified skeleton fighter with legs and a sword"""
        # Basic dimensions
        torso_height = 0.6
        torso_width = 0.4
        torso_depth = 0.2
        leg_length = 0.6
        leg_radius = 0.05
        arm_length = 0.4
        arm_radius = 0.04
        sword_length = 1.0

        # Create the main body (torso)
        torso = p.createMultiBody(
            baseMass=5.0,  # Reduced mass to reduce gravitational effects
            basePosition=[position[0], position[1], position[2] + torso_height / 2],
            baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=[torso_width / 2, torso_depth / 2,
                                                                                    torso_height / 2]),
            baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX,
                                                     halfExtents=[torso_width / 2, torso_depth / 2, torso_height / 2],
                                                     rgbaColor=[0.7, 0.7, 0.7, 1])
        )

        # Store joint IDs
        joints = {}

        # Create legs
        legs = []
        for i, offset in enumerate([(-1, -1), (1, -1), (-1, 1), (1, 1)]):  # Four corners of torso
            # Upper leg (thigh)
            upper_leg = p.createMultiBody(
                baseMass=1.0,  # Reduced mass
                basePosition=[position[0] + offset[0] * torso_width / 4, position[1] + offset[1] * torso_depth / 4,
                              position[2] + torso_height / 2 - leg_length / 4],
                baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_CAPSULE, radius=leg_radius,
                                                               height=leg_length / 2),
                baseVisualShapeIndex=p.createVisualShape(p.GEOM_CAPSULE, radius=leg_radius, length=leg_length / 2,
                                                         rgbaColor=[0.6, 0.6, 0.6, 1])
            )

            # Connect upper leg to torso with a constraint
            hip_joint = p.createConstraint(
                torso, -1, upper_leg, -1, p.JOINT_POINT2POINT,
                jointAxis=[0, 0, 1],
                parentFramePosition=[offset[0] * torso_width / 4, offset[1] * torso_depth / 4, -torso_height / 2],
                childFramePosition=[0, 0, leg_length / 4]
            )

            # Set joint limits and motor with higher force for stability
            p.changeConstraint(hip_joint, maxForce=500)  # Increased force
            joints[f"hip_{i}"] = hip_joint

            # Lower leg (calf)
            lower_leg = p.createMultiBody(
                baseMass=0.5,  # Reduced mass
                basePosition=[position[0] + offset[0] * torso_width / 4, position[1] + offset[1] * torso_depth / 4,
                              position[2] + torso_height / 2 - 3 * leg_length / 4],
                baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_CAPSULE, radius=leg_radius,
                                                               height=leg_length / 2),
                baseVisualShapeIndex=p.createVisualShape(p.GEOM_CAPSULE, radius=leg_radius, length=leg_length / 2,
                                                         rgbaColor=[0.5, 0.5, 0.5, 1])
            )

            # Connect lower leg to upper leg with a constraint
            knee_joint = p.createConstraint(
                upper_leg, -1, lower_leg, -1, p.JOINT_POINT2POINT,
                jointAxis=[1, 0, 0],
                parentFramePosition=[0, 0, -leg_length / 4],
                childFramePosition=[0, 0, leg_length / 4]
            )

            # Set joint limits and motor with higher force
            p.changeConstraint(knee_joint, maxForce=400)  # Increased force
            joints[f"knee_{i}"] = knee_joint

            # Add "foot" at the bottom of the leg for better stability
            foot = p.createMultiBody(
                baseMass=0.2,
                basePosition=[position[0] + offset[0] * torso_width / 4, position[1] + offset[1] * torso_depth / 4,
                              position[2] + torso_height / 2 - 3 * leg_length / 4 - leg_length / 4 - 0.05],
                baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_SPHERE, radius=0.06),  # Larger sphere for foot
                baseVisualShapeIndex=p.createVisualShape(p.GEOM_SPHERE, radius=0.06,
                                                         rgbaColor=[0.4, 0.4, 0.4, 1])
            )

            # Connect foot to lower leg
            ankle_joint = p.createConstraint(
                lower_leg, -1, foot, -1, p.JOINT_POINT2POINT,
                jointAxis=[0, 0, 1],
                parentFramePosition=[0, 0, -leg_length / 4],
                childFramePosition=[0, 0, 0]
            )

            p.changeConstraint(ankle_joint, maxForce=300)
            joints[f"ankle_{i}"] = ankle_joint

            legs.append((upper_leg, lower_leg, hip_joint, knee_joint))

        # Create arm for holding sword
        arm = p.createMultiBody(
            baseMass=0.8,  # Reduced mass
            basePosition=[position[0] + torso_width / 2 + arm_length / 2, position[1], position[2] + torso_height / 2],
            baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_CAPSULE, radius=arm_radius, height=arm_length),
            baseVisualShapeIndex=p.createVisualShape(p.GEOM_CAPSULE, radius=arm_radius, length=arm_length,
                                                     rgbaColor=[0.65, 0.65, 0.65, 1])
        )

        # Connect arm to torso with a point-to-point joint
        shoulder_joint = p.createConstraint(
            torso, -1, arm, -1, p.JOINT_POINT2POINT,
            jointAxis=[0, 0, 1],
            parentFramePosition=[torso_width / 2, 0, 0],
            childFramePosition=[-arm_length / 2, 0, 0]
        )

        # Set joint limits and motor with higher force
        p.changeConstraint(shoulder_joint, maxForce=300)  # Increased force
        joints["shoulder"] = shoulder_joint

        # Create sword
        sword = p.createMultiBody(
            baseMass=0.3,  # Reduced mass
            basePosition=[position[0] + torso_width / 2 + arm_length + sword_length / 2, position[1],
                          position[2] + torso_height / 2],
            baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=[sword_length / 2, 0.05, 0.01]),
            baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=[sword_length / 2, 0.05, 0.01],
                                                     rgbaColor=[0.8, 0.8, 0.8, 1])
        )

        # Connect sword to arm with a fixed joint
        try:
            wrist_joint = p.createConstraint(
                arm, -1, sword, -1, p.JOINT_FIXED,
                jointAxis=[0, 0, 1],
                parentFramePosition=[arm_length / 2, 0, 0],
                childFramePosition=[-sword_length / 2, 0, 0]
            )
        except:
            # Fallback to POINT2POINT if FIXED is not available
            wrist_joint = p.createConstraint(
                arm, -1, sword, -1, p.JOINT_POINT2POINT,
                jointAxis=[0, 0, 1],
                parentFramePosition=[arm_length / 2, 0, 0],
                childFramePosition=[-sword_length / 2, 0, 0]
            )
            p.changeConstraint(wrist_joint, maxForce=300)

        return {
            "torso": torso,
            "legs": legs,
            "arm": arm,
            "sword": sword,
            "joints": joints
        }

    def reset(self):
        """Reset the environment for a new episode"""
        # Reset fighter 1
        p.resetBasePositionAndOrientation(
            self.fighter1["torso"],
            [0, 1, 1],
            p.getQuaternionFromEuler([0, 0, 0])
        )

        # Reset fighter 2
        p.resetBasePositionAndOrientation(
            self.fighter2["torso"],
            [0, -1, 1],
            p.getQuaternionFromEuler([0, 0, math.pi])  # Face the other fighter
        )

        # Reset all joints to neutral positions
        self._reset_joints(self.fighter1)
        self._reset_joints(self.fighter2)

        # Add an initial "pose" phase to let the fighters stabilize
        # Apply strong constraints to keep the fighters in a standing position
        self._apply_standing_pose(self.fighter1)
        self._apply_standing_pose(self.fighter2)

        # Let the simulation settle with the standing pose
        for _ in range(100):  # More settling steps
            p.stepSimulation()
            if self.render_mode:
                # time.sleep(0.01)
                pass

        self.current_step = 0
        return self._get_observation()

    def _apply_standing_pose(self, fighter):
        """Apply a standing pose to help stabilize the fighter"""
        # Get the torso position
        torso_pos, _ = p.getBasePositionAndOrientation(fighter["torso"])

        # For each leg, set it in a standing position
        for i, (upper_leg, lower_leg, hip_joint, knee_joint) in enumerate(fighter["legs"]):
            # Get offset pattern based on leg index
            offset = [(-1, -1), (1, -1), (-1, 1), (1, 1)][i]

            # Calculate positions for straight legs with slight outward angle
            # Upper leg position - slightly outward from torso
            upper_leg_pos = [
                torso_pos[0] + offset[0] * 0.1,  # Offset in x
                torso_pos[1] + offset[1] * 0.1,  # Offset in y
                torso_pos[2] - 0.15  # Positioned just below torso
            ]

            # Lower leg position - directly below upper leg
            lower_leg_pos = [
                torso_pos[0] + offset[0] * 0.12,  # Slightly more offset to create slight angle
                torso_pos[1] + offset[1] * 0.12,  # Slightly more offset to create slight angle
                torso_pos[2] - 0.45  # Positioned below upper leg
            ]

            # Reset upper leg position with slight outward orientation
            # Using a small angle around y-axis to create stability
            upper_leg_orientation = p.getQuaternionFromEuler([0, offset[0] * 0.05, 0])
            p.resetBasePositionAndOrientation(
                upper_leg,
                upper_leg_pos,
                upper_leg_orientation
            )

            # Reset lower leg position with vertical orientation
            # Using a minimal angle to keep legs straight but stable
            lower_leg_orientation = p.getQuaternionFromEuler([0, offset[0] * 0.02, 0])
            p.resetBasePositionAndOrientation(
                lower_leg,
                lower_leg_pos,
                lower_leg_orientation
            )

            # Apply stronger constraint forces to maintain position
            p.changeConstraint(hip_joint, maxForce=2000)  # Increased force
            p.changeConstraint(knee_joint, maxForce=2000)  # Increased force

            # Apply higher damping to reduce oscillation
            p.changeDynamics(upper_leg, -1, linearDamping=0.95, angularDamping=0.95)
            p.changeDynamics(lower_leg, -1, linearDamping=0.95, angularDamping=0.95)

            # Get foot linked to this leg (assuming it's the next object after lower_leg)
            # This is a best guess since the code doesn't directly track which foot belongs to which leg
            try:
                # Find associated ankle joint
                ankle_joint = fighter["joints"][f"ankle_{i}"]
                # Get the constraint info to find the child body (foot)
                constraint_info = p.getConstraintInfo(ankle_joint)
                foot_id = constraint_info[2]  # Child body ID

                # Position the foot directly under the lower leg
                foot_pos = [
                    lower_leg_pos[0],  # Same x as lower leg
                    lower_leg_pos[1],  # Same y as lower leg
                    torso_pos[2] - 0.6  # Position at bottom of leg
                ]
                p.resetBasePositionAndOrientation(
                    foot_id,
                    foot_pos,
                    p.getQuaternionFromEuler([0, 0, 0])  # Flat orientation
                )

                # Apply strong constraint to ankle
                p.changeConstraint(ankle_joint, maxForce=1500)
                p.changeDynamics(foot_id, -1, lateralFriction=0.9, linearDamping=0.9, angularDamping=0.9)
            except Exception as e:
                print(f"Error positioning foot {i}: {e}")

        # Position the arm and sword appropriately
        p.changeDynamics(fighter["arm"], -1, linearDamping=0.9, angularDamping=0.9)
        p.changeDynamics(fighter["sword"], -1, linearDamping=0.9, angularDamping=0.9)

        # Apply damping to the torso as well
        p.changeDynamics(fighter["torso"], -1, linearDamping=0.8, angularDamping=0.8)

        # Apply additional gravity compensation to help the fighter stay upright
        # This applies an upward force to counteract some gravity effects
        p.applyExternalForce(
            fighter["torso"],
            -1,  # Link index (-1 means base link)
            [0, 0, 50],  # Upward force
            [0, 0, 0],  # Force position (at center of mass)
            p.WORLD_FRAME
        )

    def _reset_joints(self, fighter):
        """Reset joints to neutral positions with stronger forces"""
        for joint_name, joint_id in fighter["joints"].items():
            if "hip" in joint_name:
                # Extract the leg index from the joint name
                leg_idx = int(joint_name.split('_')[1])
                offset = [(-1, -1), (1, -1), (-1, 1), (1, 1)][leg_idx]

                # Set hip joints with slight outward angle for stability
                # Create a pivot point that positions the leg slightly outward
                pivot = [offset[0] * 0.05, offset[1] * 0.05, 0]

                p.changeConstraint(
                    joint_id,
                    jointChildPivot=pivot,
                    maxForce=1000  # Higher force for hip stability
                )

            elif "knee" in joint_name:
                # Keep knees straight for standing position
                # Using a neutral pivot point to maintain straight legs
                p.changeConstraint(
                    joint_id,
                    jointChildPivot=[0, 0, 0],
                    maxForce=800  # High force to keep legs straight
                )

            elif "shoulder" in joint_name:
                # Neutral arm position
                p.changeConstraint(
                    joint_id,
                    jointChildPivot=[0, 0, 0],
                    maxForce=500
                )

            elif "ankle" in joint_name:
                # Stabilize ankles with flat foot position
                p.changeConstraint(
                    joint_id,
                    jointChildPivot=[0, 0, 0],
                    maxForce=600  # Strong force for ankles
                )

        # Adjust dynamics parameters for better stability
        p.changeDynamics(
            fighter["torso"],
            -1,
            linearDamping=0.7,
            angularDamping=0.9,
            jointDamping=0.5
        )

    def _get_observation(self):
        """Get the current state observation"""
        observation = []

        # Get fighter 1 state
        pos1, orn1 = p.getBasePositionAndOrientation(self.fighter1["torso"])
        vel1, ang_vel1 = p.getBaseVelocity(self.fighter1["torso"])

        # Get fighter 2 state
        pos2, orn2 = p.getBasePositionAndOrientation(self.fighter2["torso"])
        vel2, ang_vel2 = p.getBaseVelocity(self.fighter2["torso"])

        # Get sword positions
        sword1_pos, sword1_orn = p.getBasePositionAndOrientation(self.fighter1["sword"])
        sword2_pos, sword2_orn = p.getBasePositionAndOrientation(self.fighter2["sword"])

        # Get joint states for fighter 1
        joint_states1 = []
        for joint_name, joint_id in self.fighter1["joints"].items():
            joint_info = p.getConstraintInfo(joint_id)
            joint_states1.extend([joint_info[2], joint_info[3], joint_info[4]])  # position and angles

        # Combine observations
        observation = list(pos1) + list(orn1) + list(vel1) + list(sword1_pos) + \
                      list(pos2) + list(orn2) + list(sword2_pos) + joint_states1

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

        # Apply actions to fighter 1's joints
        joint_names = list(self.fighter1["joints"].keys())
        for i, joint_name in enumerate(joint_names):
            if i < len(action):
                joint_id = self.fighter1["joints"][joint_name]

                # Scale force based on joint type
                force = 50  # Base force
                if "hip" in joint_name:
                    force = 100
                elif "knee" in joint_name:
                    force = 80
                elif "shoulder" in joint_name:
                    force = 60

                # For constraints, we use changeConstraint instead of setJointMotorControl2
                # Calculate new target position based on action
                # Map from [-1, 1] to a small movement range
                pos_change = action[i] * 0.05  # Small positional change based on action

                # Get current constraint info to determine appropriate modifications
                constraint_info = p.getConstraintInfo(joint_id)
                pivot_in_parent = constraint_info[8]  # Parent frame position
                pivot_in_child = constraint_info[9]  # Child frame position

                # Modify constraint to implement the "motion"
                # For simplicity, we're just modifying the child pivot point
                # In a more sophisticated version, you might want different behavior for different joint types
                new_pivot = [pivot_in_child[0],
                             pivot_in_child[1],
                             pivot_in_child[2] + pos_change]  # Apply change to z-axis

                p.changeConstraint(
                    joint_id,
                    jointChildPivot=new_pivot,
                    maxForce=force
                )

        # Step the simulation multiple times per action for stability
        for _ in range(5):
            p.stepSimulation()
            if self.render_mode:
                # time.sleep(1. / 240.)  # Slow down visualization
                pass

        # Get new observation
        observation = self._get_observation()

        # Calculate reward
        reward = self._calculate_reward()

        # Check if episode is done
        self.current_step += 1
        done = self.current_step >= self.max_episode_steps

        # Check for early termination
        torso_height = p.getBasePositionAndOrientation(self.fighter1["torso"])[0][2]
        if torso_height < 0.3:  # Fallen over
            reward -= 10  # Penalty for falling
            done = True

        # Additional info
        info = {}

        return observation, reward, done, info

    def _calculate_reward(self):
        """Calculate the reward based on the current state"""
        reward = 0

        # Get positions
        pos1, _ = p.getBasePositionAndOrientation(self.fighter1["torso"])
        pos2, _ = p.getBasePositionAndOrientation(self.fighter2["torso"])
        sword1_pos, _ = p.getBasePositionAndOrientation(self.fighter1["sword"])

        # Reward for maintaining height (not falling)
        height_reward = min(3.0, pos1[2])  # Cap at 3.0
        reward += height_reward

        # Check for sword hits
        contact_points = p.getContactPoints(self.fighter1["sword"], self.fighter2["torso"])
        if len(contact_points) > 0:
            # Reward for hitting opponent
            reward += 10.0

        # Penalty for getting hit
        contact_points = p.getContactPoints(self.fighter2["sword"], self.fighter1["torso"])
        if len(contact_points) > 0:
            reward -= 5.0

        # Small reward for moving the sword (encourages action)
        sword_vel, _ = p.getBaseVelocity(self.fighter1["sword"])
        movement_reward = 0.1 * (sword_vel[0] ** 2 + sword_vel[1] ** 2 + sword_vel[2] ** 2) ** 0.5
        reward += movement_reward

        # Reward for facing opponent
        torso1_pos, torso1_orn = p.getBasePositionAndOrientation(self.fighter1["torso"])
        direction = [pos2[0] - pos1[0], pos2[1] - pos1[1], 0]  # Direction to opponent
        direction_normalized = [
            (d / (sum(d ** 2 for d in direction) ** 0.5)) if sum(d ** 2 for d in direction) > 0 else 0 for d in
            direction]

        # Convert orientation quaternion to forward vector
        forward = [0, 1, 0]  # Assuming forward is along y-axis
        forward_world = p.multiplyTransforms([0, 0, 0], torso1_orn, forward, [0, 0, 0, 1])[0]
        forward_normalized = [(f / (sum(f ** 2 for f in forward_world[:2]) ** 0.5)) if sum(
            f ** 2 for f in forward_world[:2]) > 0 else 0 for f in forward_world[:2]] + [0]

        # Dot product to measure alignment (1 = perfect alignment, -1 = opposite direction)
        alignment = sum(a * b for a, b in zip(direction_normalized, forward_normalized))
        facing_reward = 2.0 * max(0, alignment)  # Only reward positive alignment
        reward += facing_reward

        return reward

    def close(self):
        """Clean up resources"""
        p.disconnect()


class Actor(nn.Module):
    """Policy network for PPO agent"""

    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()

        # Print dimensions for debugging
        print(f"Actor Network - Input: {state_dim}, Output: {action_dim}")

        self.network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
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
        mean = torch.tanh(self.mean(x))  # Use tanh to bound the actions to [-1, 1]

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
            nn.Linear(state_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
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

    def __init__(self, state_dim, action_dim, save_dir="./models"):
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

                if len(batch_indices) == 0:
                    continue

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
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.actor_optimizer.step()

                # Update critic
                self.critic_optimizer.zero_grad()
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
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
                #time.sleep(0.01)

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
    parser.add_argument("--render", action="store_true", help="Render the environment")
    parser.add_argument("--train", action="store_true", help="Train the agent")
    parser.add_argument("--test", action="store_true", help="Test the agent")
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes")
    parser.add_argument("--model", type=str, default="best_model", help="Model name")
    args = parser.parse_args()

    # Create models directory
    os.makedirs("./models", exist_ok=True)

    # Run training or testing
    if args.train:
        print("Training agent...")
        train(num_episodes=args.episodes, render=args.render)

    if args.test:
        print("Testing agent...")
        test(episodes=5, model_name=args.model)

    # Default to training if no arguments are provided
    if not args.train and not args.test:
        print("No arguments provided, running default training...")
        train(num_episodes=args.episodes)
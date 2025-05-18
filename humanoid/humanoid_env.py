import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import pybullet as p
import pybullet_data


class HumanoidEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, render=False):
        super(HumanoidEnv, self).__init__()

        # Initialize the random number generator
        self.np_random = None
        self.seed()

        # PyBullet setup
        try:
            if render:
                self.client = p.connect(p.GUI)
            else:
                self.client = p.connect(p.DIRECT)

            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            p.setGravity(0, 0, -10)
            p.setPhysicsEngineParameter(numSolverIterations=10)

            # Load the humanoid
            self.reset()

            # Define action and observation space
            self.num_joints = len(self.joint_ids)

            # Action space: joint positions for each controllable joint
            self.action_space = spaces.Box(
                low=-1.0, high=1.0, shape=(self.num_joints,), dtype=np.float32
            )

            # Observation space: joint positions, velocities, torso position and orientation, etc.
            # 5 values for base pose and orientation (x, y, z, pitch, roll)
            # 2 values for each joint (position, velocity)
            obs_dim = 5 + 2 * self.num_joints
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            )

            # Parameters
            self.max_episode_steps = 1000
            self.current_step = 0
        except Exception as e:
            print(f"Error initializing environment: {e}")
            raise

    def reset(self):
        p.resetSimulation()
        p.setGravity(0, 0, -10)

        # Load the ground plane
        self.plane_id = p.loadURDF("plane.urdf")

        # Load the humanoid
        ob_uids = p.loadMJCF("mjcf/humanoid.xml")
        self.humanoid_id = ob_uids[1]

        # Add a sword


        # Setup dynamics
        p.changeDynamics(self.humanoid_id, -1, linearDamping=0, angularDamping=0)

        # Get joint info
        self.joint_ids = []
        for j in range(p.getNumJoints(self.humanoid_id)):
            p.changeDynamics(self.humanoid_id, j, linearDamping=0, angularDamping=0)
            info = p.getJointInfo(self.humanoid_id, j)
            joint_type = info[2]
            if joint_type == p.JOINT_PRISMATIC or joint_type == p.JOINT_REVOLUTE:
                self.joint_ids.append(j)

        position = p.getBasePositionAndOrientation(self.humanoid_id)[0]
        sword_length = 1.0
        sword = p.createMultiBody(
            baseMass=1.0,
            basePosition=[position[0] + 0.5, position[1], position[2]],
            baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=[sword_length / 2, 0.05, 0.01]),
            baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=[sword_length / 2, 0.05, 0.01],
                                                     rgbaColor=[0.8, 0.8, 0.8, 1])
        )

        p.createConstraint(
            self.humanoid_id, -1, sword, -1, p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=[0.3, 0, 0],
            childFramePosition=[-sword_length / 2 - 0.1, 0, 0]
        )

        # Reset counters and state
        self.current_step = 0

        # Apply small random joint movements to start in a non-trivial pose
        for j in self.joint_ids:
            p.resetJointState(self.humanoid_id, j, np.random.uniform(-0.1, 0.1))

        # Let the humanoid fall to the ground and stabilize
        for _ in range(50):
            p.stepSimulation()

        # Get observation
        return self._get_observation()

    def step(self, action):
        # Clip actions to valid range
        action = np.clip(action, -1.0, 1.0)

        # Scale actions from [-1, 1] to actual joint angle range
        scaled_action = action * 1.0  # Consider actual joint limits for better scaling

        # Apply actions (joint positions)
        for i, joint_id in enumerate(self.joint_ids):
            p.setJointMotorControl2(
                self.humanoid_id, joint_id,
                p.POSITION_CONTROL,
                targetPosition=scaled_action[i],
                force=240.0
            )

        # Simulate one step
        p.stepSimulation()

        # Get new observation
        observation = self._get_observation()

        # Calculate reward
        reward = self._compute_reward()

        # Check if done
        self.current_step += 1
        done = self._is_done()

        # Additional info
        info = {}

        return observation, reward, done, info

    def _get_observation(self):
        # Get humanoid base position and orientation
        pos, orn = p.getBasePositionAndOrientation(self.humanoid_id)

        # Convert quaternion to euler angles (roll, pitch, yaw)
        euler = p.getEulerFromQuaternion(orn)

        # We'll use only position (x, y, z) and orientation (pitch, roll) for simplicity
        # Yaw (rotation around z) is less important for walking
        base_state = [pos[0], pos[1], pos[2], euler[0], euler[1]]

        # Get joint states
        joint_states = []
        for j in self.joint_ids:
            state = p.getJointState(self.humanoid_id, j)
            joint_pos = state[0]
            joint_vel = state[1]
            joint_states.extend([joint_pos, joint_vel])

        # Combine base and joint data
        observation = np.array(base_state + joint_states, dtype=np.float32)

        return observation

    def _compute_reward(self):
        # Get humanoid position
        pos, orn = p.getBasePositionAndOrientation(self.humanoid_id)

        # Get linear velocity
        linear_vel, _ = p.getBaseVelocity(self.humanoid_id)

        # Reward forward motion along x-axis
        forward_reward = linear_vel[0]  # X-axis velocity


        # Penalty for y-axis velocity (sideways movement)
        lateral_penalty = -0.1 * abs(linear_vel[1])

        # Penalty for not being upright
        euler = p.getEulerFromQuaternion(orn)
        upright_penalty = -0.2 * (abs(euler[0]) + abs(euler[1]))  # Penalize roll and pitch

        # Height penalty if too low (fallen)
        height_penalty = 0
        if pos[2] < 0.7:  # If the humanoid is lower than 0.7 meters
            height_penalty = -10

        # Energy efficiency penalty
        energy_penalty = -0.001 * np.sum(np.square(self._get_joint_velocities()))

        # Combined reward
        reward = forward_reward + lateral_penalty + upright_penalty + height_penalty + energy_penalty

        return reward

    def _is_done(self):
        # Episode is done if:
        # 1. Humanoid has fallen (height is too low)
        # 2. Maximum episode length reached
        # 3. Humanoid is not making progress (optional)

        pos, _ = p.getBasePositionAndOrientation(self.humanoid_id)

        if pos[2] < 0.5:  # Humanoid has fallen if height < 0.5 meters
            return True

        if self.current_step >= self.max_episode_steps:
            return True

        return False

    def _get_joint_velocities(self):
        """Helper method to get all joint velocities"""
        velocities = []
        for j in self.joint_ids:
            state = p.getJointState(self.humanoid_id, j)
            velocities.append(state[1])
        return np.array(velocities)

    def render(self, mode='human'):
        # PyBullet already handles rendering if we're in GUI mode
        pass

    def close(self):
        p.disconnect(self.client)

    def seed(self, seed=None):
        """Sets the seed for this env's random number generator."""
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_humanoid_position(self):
        """Helper method to get the humanoid position and orientation"""
        return p.getBasePositionAndOrientation(self.humanoid_id)
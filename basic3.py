import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import pybullet as P
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
            P.connect(P.GUI)  # For visualization
        else:
            P.connect(P.DIRECT)  # Headless mode for faster training

        P.setAdditionalSearchPath(pybullet_data.getDataPath())
        P.setGravity(0, 0, -7.0)

        P.setPhysicsEngineParameter(fixedTimeStep=1.0 / 240.0)  # Smaller timestep for stability
        P.setPhysicsEngineParameter(numSolverIterations=50)  # More solver iterations
        P.setPhysicsEngineParameter(numSubSteps=10)

        # Load ground plane
        self.plane_id = P.loadURDF("plane.urdf")

        # Create the fighters
        P.changeDynamics(self.plane_id, -1, lateralFriction=1.0, restitution=0.1)

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
        """Create a bipedal fighter with legs, torso, arms and sword"""
        # Basic dimensions
        torso_height = 0.6
        torso_width = 0.4
        torso_depth = 0.2
        leg_length = 0.6  # Total leg length
        upper_leg_length = 0.3  # Thigh
        lower_leg_length = 0.3  # Calf
        leg_radius = 0.05
        arm_length = 0.4
        arm_radius = 0.04
        sword_length = 1.0

        # Create the main body (torso)
        torso = P.createMultiBody(
            baseMass=5.0,
            basePosition=[position[0], position[1], position[2] + torso_height / 2 + leg_length],
            baseCollisionShapeIndex=P.createCollisionShape(P.GEOM_BOX, halfExtents=[torso_width / 2, torso_depth / 2,
                                                                                    torso_height / 2]),
            baseVisualShapeIndex=P.createVisualShape(P.GEOM_BOX,
                                                     halfExtents=[torso_width / 2, torso_depth / 2, torso_height / 2],
                                                     rgbaColor=[0.7, 0.7, 0.7, 1])
        )

        # Store joint IDs
        joints = {}

        # Create legs (just 2 now, left and right)
        legs = []
        for i, offset in enumerate([(-1, 0), (1, 0)]):  # Left and right legs
            leg_side = "left" if i == 0 else "right"

            # Calculate leg positions
            leg_x_offset = offset[0] * torso_width / 3  # Hip width

            # Hip joint position - at bottom of torso
            hip_pos = [
                position[0] + leg_x_offset,
                position[1],
                position[2] + torso_height / 2 + leg_length - upper_leg_length / 2  # At bottom of torso
            ]

            # Create upper leg (thigh)
            upper_leg = P.createMultiBody(
                baseMass=1.0,
                basePosition=hip_pos,
                baseCollisionShapeIndex=P.createCollisionShape(P.GEOM_CAPSULE, radius=leg_radius,
                                                               height=upper_leg_length),
                baseVisualShapeIndex=P.createVisualShape(P.GEOM_CAPSULE, radius=leg_radius, length=upper_leg_length,
                                                         rgbaColor=[0.6, 0.6, 0.6, 1]),
                baseOrientation=P.getQuaternionFromEuler([0, 0, 0])  # Vertical orientation
            )

            # Connect upper leg to torso with hip joint
            hip_joint = P.createConstraint(
                torso, -1, upper_leg, -1, P.JOINT_POINT2POINT,
                jointAxis=[0, 0, 1],
                parentFramePosition=[leg_x_offset, 0, -torso_height / 2],  # Bottom center of torso
                childFramePosition=[0, 0, upper_leg_length / 2]  # Top of upper leg
            )

            # Set hip joint constraints
            P.changeConstraint(hip_joint, maxForce=500)
            joints[f"hip_{leg_side}"] = hip_joint

            # Knee position
            knee_pos = [
                hip_pos[0],  # Same x as hip
                hip_pos[1],  # Same y as hip
                hip_pos[2] - upper_leg_length  # Below upper leg
            ]

            # Create lower leg (calf)
            lower_leg = P.createMultiBody(
                baseMass=0.5,
                basePosition=knee_pos,
                baseCollisionShapeIndex=P.createCollisionShape(P.GEOM_CAPSULE, radius=leg_radius,
                                                               height=lower_leg_length),
                baseVisualShapeIndex=P.createVisualShape(P.GEOM_CAPSULE, radius=leg_radius, length=lower_leg_length,
                                                         rgbaColor=[0.5, 0.5, 0.5, 1]),
                baseOrientation=P.getQuaternionFromEuler([0, 0, 0])  # Vertical orientation
            )

            # Connect lower leg to upper leg with knee joint
            knee_joint = P.createConstraint(
                upper_leg, -1, lower_leg, -1, P.JOINT_POINT2POINT,
                jointAxis=[1, 0, 0],  # Bend around x-axis
                parentFramePosition=[0, 0, -upper_leg_length / 2],  # Bottom of upper leg
                childFramePosition=[0, 0, lower_leg_length / 2]  # Top of lower leg
            )

            # Set knee joint constraints
            P.changeConstraint(knee_joint, maxForce=400)
            joints[f"knee_{leg_side}"] = knee_joint

            # Foot position
            foot_pos = [
                knee_pos[0],  # Same x as knee
                knee_pos[1] + 0.05,  # Slightly forward for stability
                knee_pos[2] - lower_leg_length - 0.02  # Below lower leg
            ]

            # Create foot
            foot = P.createMultiBody(
                baseMass=0.2,
                basePosition=foot_pos,
                baseCollisionShapeIndex=P.createCollisionShape(P.GEOM_BOX, halfExtents=[0.08, 0.15, 0.02]),
                # Longer foot
                baseVisualShapeIndex=P.createVisualShape(P.GEOM_BOX, halfExtents=[0.08, 0.15, 0.02],
                                                         rgbaColor=[0.4, 0.4, 0.4, 1])
            )

            # Connect foot to lower leg with ankle joint
            ankle_joint = P.createConstraint(
                lower_leg, -1, foot, -1, P.JOINT_POINT2POINT,
                jointAxis=[1, 0, 0],  # Bend around x-axis
                parentFramePosition=[0, 0, -lower_leg_length / 2],  # Bottom of lower leg
                childFramePosition=[0, -0.05, 0]  # Slightly back on foot for natural stance
            )

            # Set ankle joint constraints and foot properties
            P.changeConstraint(ankle_joint, maxForce=300)
            P.changeDynamics(foot, -1, lateralFriction=0.9, spinningFriction=0.1, rollingFriction=0.1)
            joints[f"ankle_{leg_side}"] = ankle_joint

            legs.append((upper_leg, lower_leg, hip_joint, knee_joint))

        # Create left arm
        left_arm_pos = [
            position[0] - torso_width / 2 - arm_length / 2,
            position[1],
            position[2] + torso_height / 2 + leg_length
        ]

        left_arm = P.createMultiBody(
            baseMass=0.8,
            basePosition=left_arm_pos,
            baseCollisionShapeIndex=P.createCollisionShape(P.GEOM_CAPSULE, radius=arm_radius, height=arm_length),
            baseVisualShapeIndex=P.createVisualShape(P.GEOM_CAPSULE, radius=arm_radius, length=arm_length,
                                                     rgbaColor=[0.65, 0.65, 0.65, 1]),
            baseOrientation=P.getQuaternionFromEuler([0, 1.57, 0])  # Horizontal orientation
        )

        # Connect left arm to torso
        left_shoulder = P.createConstraint(
            torso, -1, left_arm, -1, P.JOINT_POINT2POINT,
            jointAxis=[0, 0, 1],
            parentFramePosition=[-torso_width / 2, 0, 0],  # Left side of torso
            childFramePosition=[arm_length / 2, 0, 0]  # Right end of arm
        )

        P.changeConstraint(left_shoulder, maxForce=300)
        joints["shoulder_left"] = left_shoulder

        # Create right arm for sword
        right_arm_pos = [
            position[0] + torso_width / 2 + arm_length / 2,
            position[1],
            position[2] + torso_height / 2 + leg_length
        ]

        right_arm = P.createMultiBody(
            baseMass=0.8,
            basePosition=right_arm_pos,
            baseCollisionShapeIndex=P.createCollisionShape(P.GEOM_CAPSULE, radius=arm_radius, height=arm_length),
            baseVisualShapeIndex=P.createVisualShape(P.GEOM_CAPSULE, radius=arm_radius, length=arm_length,
                                                     rgbaColor=[0.65, 0.65, 0.65, 1]),
            baseOrientation=P.getQuaternionFromEuler([0, 1.57, 0])  # Horizontal orientation
        )

        # Connect right arm to torso
        right_shoulder = P.createConstraint(
            torso, -1, right_arm, -1, P.JOINT_POINT2POINT,
            jointAxis=[0, 0, 1],
            parentFramePosition=[torso_width / 2, 0, 0],  # Right side of torso
            childFramePosition=[-arm_length / 2, 0, 0]  # Left end of arm
        )

        P.changeConstraint(right_shoulder, maxForce=300)
        joints["shoulder_right"] = right_shoulder

        # Create sword
        sword_pos = [
            position[0] + torso_width / 2 + arm_length + sword_length / 2,
            position[1],
            position[2] + torso_height / 2 + leg_length
        ]

        sword = P.createMultiBody(
            baseMass=0.3,
            basePosition=sword_pos,
            baseCollisionShapeIndex=P.createCollisionShape(P.GEOM_BOX, halfExtents=[sword_length / 2, 0.05, 0.01]),
            baseVisualShapeIndex=P.createVisualShape(P.GEOM_BOX, halfExtents=[sword_length / 2, 0.05, 0.01],
                                                     rgbaColor=[0.8, 0.8, 0.8, 1])
        )

        # Connect sword to right arm
        wrist_joint = P.createConstraint(
            right_arm, -1, sword, -1, P.JOINT_POINT2POINT,
            jointAxis=[0, 0, 1],
            parentFramePosition=[arm_length / 2, 0, 0],  # End of arm
            childFramePosition=[-sword_length / 2, 0, 0]  # Start of sword
        )

        P.changeConstraint(wrist_joint, maxForce=300)
        joints["wrist_right"] = wrist_joint

        return {
            "torso": torso,
            "legs": legs,
            "arms": [left_arm, right_arm],
            "sword": sword,
            "joints": joints
        }

    def reset(self):
        """Reset the environment for a new episode with improved bipedal initialization"""
        # Reset fighter 1
        P.resetBasePositionAndOrientation(
            self.fighter1["torso"],
            [0, 1, 1.2],  # Higher position for bipedal
            P.getQuaternionFromEuler([0, 0.05, 0])  # Slight forward lean for stability
        )

        # Reset fighter 2
        P.resetBasePositionAndOrientation(
            self.fighter2["torso"],
            [0, -1, 1.2],  # Higher position for bipedal
            P.getQuaternionFromEuler([0, 0.05, math.pi])  # Face the other fighter with slight lean
        )

        # Reset all joints to neutral positions
        self._reset_joints(self.fighter1)
        self._reset_joints(self.fighter2)

        # Apply standing pose for stability
        self._apply_standing_pose(self.fighter1)
        self._apply_standing_pose(self.fighter2)

        # Let the simulation settle with the standing pose
        # More settling steps for bipedal stability
        for step_idx in range(200):  # Doubled settling time
            P.stepSimulation()

            # Add additional stabilization during settling
            for fighter in [self.fighter1, self.fighter2]:
                torso_pos, _ = P.getBasePositionAndOrientation(fighter["torso"])

                # Add upward force to counter gravity during settling
                # Calculate force that decreases over time
                force_value = 60.0 - (step_idx * 0.2)  # Use step_idx instead of _

                P.applyExternalForce(
                    fighter["torso"], -1,
                    [0, 0, force_value],  # Gradually decrease force
                    [0, 0, 0],
                    P.WORLD_FRAME
                )

                # Check for leg stability
                for leg_idx, leg_side in enumerate(["left", "right"]):
                    try:
                        # Apply outward force to feet to maintain stance width
                        ankle_joint = fighter["joints"][f"ankle_{leg_side}"]
                        constraint_info = P.getConstraintInfo(ankle_joint)
                        foot_id = constraint_info[2]

                        # Calculate outward direction based on leg side
                        x_direction = -1 if leg_side == "left" else 1

                        # Only apply force in first half of settling
                        if step_idx < 100:
                            P.applyExternalForce(
                                foot_id, -1,
                                [x_direction * 3, 0, 0],  # Outward force
                                [0, 0, 0],
                                P.WORLD_FRAME
                            )
                    except:
                        pass

            if self.render_mode:
                pass

        # Reset episode variables
        self.current_step = 0
        return self._get_observation()

    def _apply_standing_pose(self, fighter):
        """Apply a standing pose to bipedal fighter for stability"""
        # Get the torso position
        torso_pos, _ = P.getBasePositionAndOrientation(fighter["torso"])

        # Reset torso velocity
        P.resetBaseVelocity(fighter["torso"], linearVelocity=[0, 0, 0], angularVelocity=[0, 0, 0])

        # Apply damping to torso
        P.changeDynamics(
            fighter["torso"], -1,
            linearDamping=0.9,
            angularDamping=0.9,
            jointDamping=0.9
        )

        # Configure legs
        leg_sides = ["left", "right"]
        for i, (upper_leg, lower_leg, hip_joint, knee_joint) in enumerate(fighter["legs"]):
            leg_side = leg_sides[i]

            # Get leg offset (left = -1, right = 1)
            x_offset = -1 if leg_side == "left" else 1

            # ===== Position upper leg (thigh) =====
            upper_leg_pos = [
                torso_pos[0] + x_offset * 0.13,  # Offset from center for hip width
                torso_pos[1] + 0.01,  # Slightly forward for stability
                torso_pos[2] - 0.3  # Below torso
            ]

            # Position with a very slight bend forward for natural stance
            upper_leg_orientation = P.getQuaternionFromEuler([0, 0.05, 0])
            P.resetBasePositionAndOrientation(upper_leg, upper_leg_pos, upper_leg_orientation)
            P.resetBaseVelocity(upper_leg, [0, 0, 0], [0, 0, 0])

            # ===== Position lower leg (calf) =====
            lower_leg_pos = [
                upper_leg_pos[0],  # Same x as upper leg
                upper_leg_pos[1] + 0.02,  # Slightly more forward
                upper_leg_pos[2] - 0.3  # Below upper leg
            ]

            # Position with a very slight bend for natural stance
            lower_leg_orientation = P.getQuaternionFromEuler([0, 0.1, 0])
            P.resetBasePositionAndOrientation(lower_leg, lower_leg_pos, lower_leg_orientation)
            P.resetBaseVelocity(lower_leg, [0, 0, 0], [0, 0, 0])

            # Apply stronger constraint forces for stability
            P.changeConstraint(hip_joint, maxForce=2000, erp=0.9)
            P.changeConstraint(knee_joint, maxForce=2000, erp=0.9)

            # Apply high damping to reduce oscillation
            P.changeDynamics(upper_leg, -1, linearDamping=0.95, angularDamping=0.95, jointDamping=0.95)
            P.changeDynamics(lower_leg, -1, linearDamping=0.95, angularDamping=0.95, jointDamping=0.95)

            # ===== Position foot =====
            try:
                ankle_joint = fighter["joints"][f"ankle_{leg_side}"]
                constraint_info = P.getConstraintInfo(ankle_joint)
                foot_id = constraint_info[2]

                foot_pos = [
                    lower_leg_pos[0],  # Same x as lower leg
                    lower_leg_pos[1] + 0.1,  # More forward for stability
                    lower_leg_pos[2] - 0.3  # Below lower leg
                ]

                # Flat orientation for stability
                P.resetBasePositionAndOrientation(
                    foot_id,
                    foot_pos,
                    P.getQuaternionFromEuler([0, 0, 0])
                )
                P.resetBaseVelocity(foot_id, [0, 0, 0], [0, 0, 0])

                # Apply strong constraints and friction
                P.changeConstraint(ankle_joint, maxForce=1500, erp=0.9)
                P.changeDynamics(
                    foot_id, -1,
                    lateralFriction=0.9,
                    spinningFriction=0.2,
                    rollingFriction=0.1,
                    restitution=0.1,
                    linearDamping=0.9,
                    angularDamping=0.9
                )
            except Exception as e:
                print(f"Error positioning foot {leg_side}: {e}")

        # Configure arms
        arms = fighter["arms"]
        if len(arms) >= 2:
            # Left arm
            left_arm = arms[0]
            left_shoulder = fighter["joints"]["shoulder_left"]
            P.resetBaseVelocity(left_arm, [0, 0, 0], [0, 0, 0])
            P.changeConstraint(left_shoulder, maxForce=500, erp=0.8)
            P.changeDynamics(left_arm, -1, linearDamping=0.8, angularDamping=0.8)

            # Right arm (sword arm)
            right_arm = arms[1]
            right_shoulder = fighter["joints"]["shoulder_right"]
            P.resetBaseVelocity(right_arm, [0, 0, 0], [0, 0, 0])
            P.changeConstraint(right_shoulder, maxForce=500, erp=0.8)
            P.changeDynamics(right_arm, -1, linearDamping=0.8, angularDamping=0.8)

            # Sword
            sword = fighter["sword"]
            wrist_joint = fighter["joints"]["wrist_right"]
            P.resetBaseVelocity(sword, [0, 0, 0], [0, 0, 0])
            P.changeConstraint(wrist_joint, maxForce=400, erp=0.8)
            P.changeDynamics(sword, -1, linearDamping=0.8, angularDamping=0.8)

        # Apply additional stabilizing forces
        # Extra upward force on torso to counter gravity during initialization
        P.applyExternalForce(
            fighter["torso"], -1,
            [0, 0, 80],  # Strong upward force
            [0, 0, 0],
            P.WORLD_FRAME
        )

        # Apply slight forward lean to the torso for better stability
        torso_orientation = P.getQuaternionFromEuler([0, 0.05, 0])  # Very slight forward lean
        P.resetBasePositionAndOrientation(
            fighter["torso"],
            torso_pos,
            torso_orientation
        )

    def _reset_joints(self, fighter):
        """Reset joints to natural bipedal standing positions"""
        # Reset leg joints
        for leg_idx, leg_side in enumerate(["left", "right"]):
            # Hip joint - slightly wider stance than neutral
            hip_joint = fighter["joints"][f"hip_{leg_side}"]
            x_offset = -0.05 if leg_side == "left" else 0.05  # Outward position
            P.changeConstraint(
                hip_joint,
                jointChildPivot=[x_offset, 0.01, 0],  # Slight forward and outward stance
                maxForce=1000
            )

            # Knee joint - slight bend for natural stance
            knee_joint = fighter["joints"][f"knee_{leg_side}"]
            P.changeConstraint(
                knee_joint,
                jointChildPivot=[0, 0.03, 0],  # Slight bend at knee
                maxForce=800
            )

            # Ankle joint - flat against ground but angled for balance
            ankle_joint = fighter["joints"][f"ankle_{leg_side}"]
            P.changeConstraint(
                ankle_joint,
                jointChildPivot=[0, 0.02, 0],  # Angle for stability
                maxForce=600
            )

        # Reset arm joints
        # Left arm - relaxed position
        left_shoulder = fighter["joints"]["shoulder_left"]
        P.changeConstraint(
            left_shoulder,
            jointChildPivot=[0, 0, -0.05],  # Slightly down
            maxForce=400
        )

        # Right arm (sword arm) - ready position
        right_shoulder = fighter["joints"]["shoulder_right"]
        P.changeConstraint(
            right_shoulder,
            jointChildPivot=[0, 0.1, 0],  # Forward position
            maxForce=400
        )

        # Sword wrist - ready position
        if "wrist_right" in fighter["joints"]:
            wrist_joint = fighter["joints"]["wrist_right"]
            P.changeConstraint(
                wrist_joint,
                jointChildPivot=[0, 0, 0],  # Neutral position
                maxForce=300
            )

        # Adjust dynamics parameters for better stability
        P.changeDynamics(
            fighter["torso"],
            -1,
            linearDamping=0.8,
            angularDamping=0.9,
            jointDamping=0.7
        )

    def _get_observation(self):
        """Get the current state observation for bipedal fighter"""
        observation = []

        # Get fighter 1 state
        pos1, orn1 = P.getBasePositionAndOrientation(self.fighter1["torso"])
        vel1, ang_vel1 = P.getBaseVelocity(self.fighter1["torso"])

        # Get fighter 2 state
        pos2, orn2 = P.getBasePositionAndOrientation(self.fighter2["torso"])
        vel2, ang_vel2 = P.getBaseVelocity(self.fighter2["torso"])

        # Get sword positions
        sword1_pos, sword1_orn = P.getBasePositionAndOrientation(self.fighter1["sword"])
        sword2_pos, sword2_orn = P.getBasePositionAndOrientation(self.fighter2["sword"])

        # Get sword velocities
        sword1_vel, sword1_ang_vel = P.getBaseVelocity(self.fighter1["sword"])

        # Get arm positions and velocities
        arm_positions = []
        arm_velocities = []
        for arm in self.fighter1["arms"]:
            arm_pos, arm_orn = P.getBasePositionAndOrientation(arm)
            arm_vel, arm_ang_vel = P.getBaseVelocity(arm)
            arm_positions.extend(list(arm_pos) + list(arm_orn))
            arm_velocities.extend(list(arm_vel) + list(arm_ang_vel))

        # Get leg state (for each leg)
        leg_states = []
        for leg_idx, (upper_leg, lower_leg, hip_joint, knee_joint) in enumerate(self.fighter1["legs"]):
            # Get upper leg state
            upper_leg_pos, upper_leg_orn = P.getBasePositionAndOrientation(upper_leg)
            upper_leg_vel, upper_leg_ang_vel = P.getBaseVelocity(upper_leg)

            # Get lower leg state
            lower_leg_pos, lower_leg_orn = P.getBasePositionAndOrientation(lower_leg)
            lower_leg_vel, lower_leg_ang_vel = P.getBaseVelocity(lower_leg)

            # Get foot state
            leg_side = "left" if leg_idx == 0 else "right"
            try:
                ankle_joint = self.fighter1["joints"][f"ankle_{leg_side}"]
                constraint_info = P.getConstraintInfo(ankle_joint)
                foot_id = constraint_info[2]

                foot_pos, foot_orn = P.getBasePositionAndOrientation(foot_id)
                foot_vel, foot_ang_vel = P.getBaseVelocity(foot_id)

                # Check if foot is touching ground (important feature for balance)
                foot_ground_contact = len(P.getContactPoints(foot_id, self.plane_id)) > 0
                foot_ground_contact = 1.0 if foot_ground_contact else 0.0
            except:
                # Default values if foot not found
                foot_pos = [0, 0, 0]
                foot_orn = [0, 0, 0, 1]
                foot_vel = [0, 0, 0]
                foot_ang_vel = [0, 0, 0]
                foot_ground_contact = 0.0

            # Combine leg state features
            leg_state = (
                    list(upper_leg_pos) + list(upper_leg_orn) + list(upper_leg_vel) +
                    list(lower_leg_pos) + list(lower_leg_orn) + list(lower_leg_vel) +
                    list(foot_pos) + list(foot_vel) + [foot_ground_contact]
            )
            leg_states.extend(leg_state)

        # Extract relative features (distances, angles)
        # Vector from fighter1 to fighter2
        relative_pos = [pos2[0] - pos1[0], pos2[1] - pos1[1], pos2[2] - pos1[2]]
        distance_to_opponent = sum(x * x for x in relative_pos) ** 0.5

        # Direction vector from torso to sword (for fighter 1)
        sword_vector = [sword1_pos[0] - pos1[0], sword1_pos[1] - pos1[1], sword1_pos[2] - pos1[2]]

        # Up vector from orientation (for balance)
        rotation_matrix = P.getMatrixFromQuaternion(orn1)
        up_vector = [rotation_matrix[2], rotation_matrix[5], rotation_matrix[8]]

        # Combine all observations
        observation = (
                list(pos1) + list(orn1) + list(vel1) + list(ang_vel1) +  # Torso state
                list(relative_pos) + [distance_to_opponent] +  # Relative to opponent
                list(sword1_pos) + list(sword1_vel) +  # Sword state
                arm_positions + arm_velocities +  # Arms state
                leg_states +  # Legs state
                list(up_vector) +  # Balance indicator
                list(sword_vector)  # Sword control
        )

        return observation

    def step(self, action):
        """Execute one step with the bipedal fighter"""
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
                elif "ankle" in joint_name:
                    force = 40
                elif "wrist" in joint_name:
                    force = 30

                # Get current constraint info
                constraint_info = P.getConstraintInfo(joint_id)
                pivot_in_child = constraint_info[9]  # Child frame position

                # Determine which leg/arm this joint belongs to
                is_left = "left" in joint_name
                is_right = "right" in joint_name
                side_factor = -1 if is_left else 1  # For asymmetric movements

                # Different control for different joint types
                if "hip" in joint_name:
                    # Hip movements - forward/backward and side-to-side
                    pos_change_x = action[i] * 0.03 * side_factor  # Side movement
                    pos_change_y = action[i] * 0.05  # Forward/backward movement

                    new_pivot = [
                        pivot_in_child[0] + pos_change_x,
                        pivot_in_child[1] + pos_change_y,
                        pivot_in_child[2]  # Maintain height
                    ]

                    P.changeConstraint(joint_id, jointChildPivot=new_pivot, maxForce=force)

                elif "knee" in joint_name:
                    # Knee bending - only positive (natural) direction
                    # For bipedal walking, we need more knee flexibility
                    knee_action = max(0, action[i])  # Only positive bending
                    pos_change = knee_action * 0.06  # Larger range for walking

                    new_pivot = [
                        pivot_in_child[0],
                        pivot_in_child[1] + pos_change,  # Forward bend
                        pivot_in_child[2]
                    ]

                    P.changeConstraint(joint_id, jointChildPivot=new_pivot, maxForce=force)

                elif "ankle" in joint_name:
                    # Ankle movements - mainly for balance adjustments
                    ankle_action = action[i] * 0.02  # Small range

                    new_pivot = [
                        pivot_in_child[0] + ankle_action * side_factor,  # Side tilting
                        pivot_in_child[1] + ankle_action,  # Forward/backward tilting
                        pivot_in_child[2]
                    ]

                    P.changeConstraint(joint_id, jointChildPivot=new_pivot, maxForce=force)

                elif "shoulder" in joint_name:
                    # Arm movements - wider range especially for sword arm
                    # More movement on sword arm
                    mov_scale = 0.1 if "right" in joint_name else 0.05

                    pos_change_x = action[i] * mov_scale * side_factor
                    pos_change_y = action[i] * mov_scale
                    pos_change_z = action[i] * mov_scale * 0.5  # Less vertical movement

                    new_pivot = [
                        pivot_in_child[0] + pos_change_x,
                        pivot_in_child[1] + pos_change_y,
                        pivot_in_child[2] + pos_change_z
                    ]

                    P.changeConstraint(joint_id, jointChildPivot=new_pivot, maxForce=force)

                elif "wrist" in joint_name:
                    # Wrist movement for sword control
                    wrist_action = action[i] * 0.04

                    new_pivot = [
                        pivot_in_child[0],
                        pivot_in_child[1] + wrist_action,
                        pivot_in_child[2] + wrist_action * 0.5
                    ]

                    P.changeConstraint(joint_id, jointChildPivot=new_pivot, maxForce=force)

        # Step the simulation multiple times per action for stability
        for _ in range(10):  # More substeps for bipedal stability
            P.stepSimulation()
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
        torso_height = P.getBasePositionAndOrientation(self.fighter1["torso"])[0][2]
        if torso_height < 0.7:  # Fallen over - higher threshold for bipedal
            reward -= 10  # Penalty for falling
            done = True

        # Additional info
        info = {}

        return observation, reward, done, info

    def _calculate_reward(self):
        """Calculate reward with bipedal balance components"""
        reward = 0

        # Get positions
        pos1, orn1 = P.getBasePositionAndOrientation(self.fighter1["torso"])
        pos2, _ = P.getBasePositionAndOrientation(self.fighter2["torso"])
        sword1_pos, _ = P.getBasePositionAndOrientation(self.fighter1["sword"])

        # ====== BALANCE REWARD COMPONENTS ======

        # 1. Reward for maintaining upright torso orientation
        rotation_matrix = P.getMatrixFromQuaternion(orn1)
        up_vector = [rotation_matrix[2], rotation_matrix[5], rotation_matrix[8]]  # Z-axis component
        upright_alignment = up_vector[2]  # Z component is the dot product with [0,0,1]

        # Stronger upright reward for bipedal balance
        upright_reward = 8.0 * upright_alignment
        reward += upright_reward

        # 2. Penalize torso contact with ground
        contact_points = P.getContactPoints(self.fighter1["torso"], self.plane_id)
        if len(contact_points) > 0:
            reward -= 20.0  # Harsher penalty for bipedal

        # 3. Check leg parts for ground contact
        for upper_leg, lower_leg, _, _ in self.fighter1["legs"]:
            # Only lower legs should touch ground
            contact_points = P.getContactPoints(upper_leg, self.plane_id)
            if len(contact_points) > 0:
                reward -= 10.0  # Penalty for upper leg touching ground

            # Lower leg touching ground is acceptable but not ideal
            contact_points = P.getContactPoints(lower_leg, self.plane_id)
            if len(contact_points) > 0:
                reward -= 3.0  # Small penalty

        # 4. Reward for height of torso (staying upright)
        # Bipedal needs a higher threshold
        target_height = 1.2  # Expected height when standing
        height_diff = abs(pos1[2] - target_height)
        height_reward = 5.0 * math.exp(-height_diff * 2)  # Gaussian reward, peaks at target height
        reward += height_reward

        # 5. Reward for stable orientation (penalize fast rotation)
        _, ang_vel1 = P.getBaseVelocity(self.fighter1["torso"])
        rotation_penalty = -0.2 * sum(v * v for v in ang_vel1)  # Stronger penalty for rotation
        reward += rotation_penalty

        # 6. Reward for feet touching ground (exactly what we want)
        feet_on_ground = 0
        for i, leg_side in enumerate(["left", "right"]):
            try:
                ankle_joint = self.fighter1["joints"][f"ankle_{leg_side}"]
                constraint_info = P.getConstraintInfo(ankle_joint)
                foot_id = constraint_info[2]

                # Check if foot touches ground
                contact_points = P.getContactPoints(foot_id, self.plane_id)
                if len(contact_points) > 0:
                    feet_on_ground += 1
            except:
                pass

        # Maximum reward when both feet are on ground
        if feet_on_ground == 2:
            reward += 4.0
        elif feet_on_ground == 1:
            reward += 1.0  # Partial reward for one foot (allows for walking)
        else:
            reward -= 2.0  # Penalty for no feet on ground

        # ====== SWORD FIGHTING REWARD COMPONENTS ======

        # Reward for hitting opponent
        contact_points = P.getContactPoints(self.fighter1["sword"], self.fighter2["torso"])
        if len(contact_points) > 0:
            reward += 10.0

        # Penalty for getting hit
        contact_points = P.getContactPoints(self.fighter2["sword"], self.fighter1["torso"])
        if len(contact_points) > 0:
            reward -= 5.0

        # Small reward for moving the sword (encourages action)
        sword_vel, _ = P.getBaseVelocity(self.fighter1["sword"])
        movement_reward = 0.1 * (sword_vel[0] ** 2 + sword_vel[1] ** 2 + sword_vel[2] ** 2) ** 0.5
        reward += movement_reward

        # Reward for facing opponent
        direction = [pos2[0] - pos1[0], pos2[1] - pos1[1], 0]
        direction_magnitude = sum(d ** 2 for d in direction) ** 0.5
        if direction_magnitude > 0:
            direction_normalized = [d / direction_magnitude for d in direction]

            # Convert orientation quaternion to forward vector
            forward = [0, 1, 0]  # Assuming forward is along y-axis
            forward_world = P.multiplyTransforms([0, 0, 0], orn1, forward, [0, 0, 0, 1])[0]
            forward_magnitude = sum(f ** 2 for f in forward_world[:2]) ** 0.5

            if forward_magnitude > 0:
                forward_normalized = [f / forward_magnitude for f in forward_world[:2]] + [0]
                alignment = sum(a * b for a, b in zip(direction_normalized, forward_normalized))
                facing_reward = 2.0 * max(0, alignment)
                reward += facing_reward

        return reward

    def close(self):
        """Clean up resources"""
        P.disconnect()


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

    # Training metrics
    total_rewards = []
    balance_metrics = {
        "upright_alignment": [],
        "falls_per_episode": [],
        "avg_height": []
    }

    try:
        for episode in range(num_episodes):
            # Reset environment
            state = env.reset()
            episode_reward = 0

            # Balance tracking for this episode
            falls_this_episode = 0
            upright_sum = 0
            height_sum = 0
            step = 0

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

                # Track balance metrics
                pos1, orn1 = P.getBasePositionAndOrientation(env.fighter1["torso"])

                # Track height
                height_sum += pos1[2]

                # Track uprightness
                rotation_matrix = P.getMatrixFromQuaternion(orn1)
                up_vector = [rotation_matrix[2], rotation_matrix[5], rotation_matrix[8]]
                upright_alignment = up_vector[2]
                upright_sum += upright_alignment

                # Check for falls (torso touching ground)
                contact_points = P.getContactPoints(env.fighter1["torso"], env.plane_id)
                if len(contact_points) > 0:
                    falls_this_episode += 1

                # Learn if buffer is large enough
                if agent.buffer.size() >= 128:
                    agent.learn()

                # Check if episode is done
                if done:
                    break

            # Final learning step
            if agent.buffer.size() > 0:
                agent.learn()

            # Store metrics
            total_rewards.append(episode_reward)

            # Store balance metrics
            steps_completed = step + 1  # Account for 0-indexing
            balance_metrics["upright_alignment"].append(upright_sum / steps_completed)
            balance_metrics["falls_per_episode"].append(falls_this_episode)
            balance_metrics["avg_height"].append(height_sum / steps_completed)

            # Print progress with balance metrics
            print(f"Episode {episode + 1}/{num_episodes}, Reward: {episode_reward:.2f}")
            print(f"  Balance Metrics - Upright: {balance_metrics['upright_alignment'][-1]:.2f}, " +
                  f"Falls: {falls_this_episode}, Avg Height: {balance_metrics['avg_height'][-1]:.2f}")

            # Save model periodically
            if (episode + 1) % 10 == 0:
                agent.save_model(f"model_ep{episode + 1}")

                # Periodically visualize balance progress
                if hasattr(env, 'render_mode') and not env.render_mode:
                    print("Balance improvement over last 10 episodes:")
                    print(f"  Upright alignment: {sum(balance_metrics['upright_alignment'][-10:]) / 10:.3f}")
                    print(f"  Falls per episode: {sum(balance_metrics['falls_per_episode'][-10:]) / 10:.2f}")
                    print(f"  Average height: {sum(balance_metrics['avg_height'][-10:]) / 10:.3f}")

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

        # Save balance metrics for later analysis
        try:
            import json
            with open("balance_metrics.json", "w") as f:
                json.dump(balance_metrics, f)
            print("Balance metrics saved to balance_metrics.json")
        except:
            print("Could not save balance metrics to file")

        # Close environment
        env.close()


def curriculum_training(episodes_per_phase=50, render_final=True):
    """
    Implement curriculum learning for bipedal sword fighting
    - Phase 1: Learn to stand and balance
    - Phase 2: Learn to move while maintaining balance
    - Phase 3: Learn to sword fight while balanced
    """
    # Each phase uses a modified reward function emphasizing different skills
    print("Starting curriculum training for bipedal fighters...")
    # Create environment
    env = SwordFighterEnv(render=False)  # No rendering during training phases
    # Create agent
    agent = PPOAgent(env.observation_space_size, env.action_space_size)
    # Phase 1: Balance Training
    print("\nPhase 1: Balance Training")
    print("-------------------------")
    # Override reward calculation to focus on balance
    original_reward_fn = env._calculate_reward

    def balance_reward(self):
        # Get positions
        pos1, orn1 = P.getBasePositionAndOrientation(self.fighter1["torso"])
        # Calculate reward focused only on balance
        reward = 0
        # Strong reward for upright orientation
        rotation_matrix = P.getMatrixFromQuaternion(orn1)
        up_vector = [rotation_matrix[2], rotation_matrix[5], rotation_matrix[8]]
        upright_alignment = up_vector[2]
        reward += 10.0 * upright_alignment
        # Reward for proper height
        target_height = 1.2
        height_diff = abs(pos1[2] - target_height)
        reward += 8.0 * math.exp(-height_diff * 3)
        # Penalty for torso contact with ground
        if len(P.getContactPoints(self.fighter1["torso"], self.plane_id)) > 0:
            reward -= 20.0
        # Reward for feet touching ground
        feet_on_ground = 0
        for leg_side in ["left", "right"]:
            try:
                ankle_joint = self.fighter1["joints"][f"ankle_{leg_side}"]
                constraint_info = P.getConstraintInfo(ankle_joint)
                foot_id = constraint_info[2]
                if len(P.getContactPoints(foot_id, self.plane_id)) > 0:
                    feet_on_ground += 1
            except:
                pass

        if feet_on_ground == 2:
            reward += 5.0
        elif feet_on_ground == 1:
            reward += 1.0

        # Penalty for excessive movement (want static balance first)
        vel1, ang_vel1 = P.getBaseVelocity(self.fighter1["torso"])
        vel_penalty = -0.5 * sum(v * v for v in vel1)
        ang_vel_penalty = -1.0 * sum(v * v for v in ang_vel1)
        reward += vel_penalty + ang_vel_penalty
        return reward

    # Override reward method for phase 1
    env._calculate_reward = types.MethodType(balance_reward, env)
    # Train for balance
    for episode in range(episodes_per_phase):
        state = env.reset()
        episode_reward = 0
        done = False
        while not done:
            action, log_prob, value = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.store_experience(state, action, reward, value, log_prob, done)
            if agent.buffer.size() >= 128:
                agent.learn()
            state = next_state
            episode_reward += reward
        if agent.buffer.size() > 0:
            agent.learn()
        print(f"Episode {episode + 1}/{episodes_per_phase}, Reward: {episode_reward:.2f}")
        # Save phase 1 model
        if (episode + 1) % 10 == 0:
            agent.save_model(f"bipedal_phase1_ep{episode + 1}")

    # Save final phase 1 model
    agent.save_model("bipedal_phase1_final")

    # Phase 2: Movement Training
    print("\nPhase 2: Movement Training")
    print("--------------------------")

    def movement_reward(self):
        # Get positions
        pos1, orn1 = P.getBasePositionAndOrientation(self.fighter1["torso"])
        vel1, ang_vel1 = P.getBaseVelocity(self.fighter1["torso"])

        # Balance component (reduced weight)
        reward = 0

        # Upright orientation (still important)
        rotation_matrix = P.getMatrixFromQuaternion(orn1)
        up_vector = [rotation_matrix[2], rotation_matrix[5], rotation_matrix[8]]
        upright_alignment = up_vector[2]
        reward += 6.0 * upright_alignment

        # Height component (still important)
        target_height = 1.2
        height_diff = abs(pos1[2] - target_height)
        reward += 5.0 * math.exp(-height_diff * 3)

        # Penalty for torso contact
        if len(P.getContactPoints(self.fighter1["torso"], self.plane_id)) > 0:
            reward -= 20.0

        # Foot contact - now we care about alternating steps
        left_foot_contact = 0
        right_foot_contact = 0
        try:
            # Left foot
            ankle_joint = self.fighter1["joints"]["ankle_left"]
            constraint_info = P.getConstraintInfo(ankle_joint)
            foot_id = constraint_info[2]
            left_foot_contact = 1 if len(P.getContactPoints(foot_id, self.plane_id)) > 0 else 0
            # Right foot
            ankle_joint = self.fighter1["joints"]["ankle_right"]
            constraint_info = P.getConstraintInfo(ankle_joint)
            foot_id = constraint_info[2]
            right_foot_contact = 1 if len(P.getContactPoints(foot_id, self.plane_id)) > 0 else 0
        except:
            pass
        # Reward walking pattern - at least one foot on ground is good
        if left_foot_contact + right_foot_contact >= 1:
            reward += 3.0
        else:
            reward -= 2.0  # Penalty for no feet on ground
        # Movement reward - forward speed is good
        forward_velocity = vel1[1]  # Y is forward
        reward += 2.0 * forward_velocity if forward_velocity > 0 else 0
        # Penalize sideways and vertical velocity
        side_velocity = abs(vel1[0])
        reward -= 1.0 * side_velocity
        # Penalize excessive angular velocity
        reward -= 0.5 * sum(v * v for v in ang_vel1)
        return reward
    # Override reward method for phase 2
    env._calculate_reward = types.MethodType(movement_reward, env)
    # Train for movement

    for episode in range(episodes_per_phase):
        state = env.reset()
        episode_reward = 0
        done = False
        while not done:
            action, log_prob, value = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.store_experience(state, action, reward, value, log_prob, done)
            if agent.buffer.size() >= 128:
                agent.learn()
            state = next_state
            episode_reward += reward
        if agent.buffer.size() > 0:
            agent.learn()
        print(f"Episode {episode + 1}/{episodes_per_phase}, Reward: {episode_reward:.2f}")
        # Save phase 2 model
        if (episode + 1) % 10 == 0:
            agent.save_model(f"bipedal_phase2_ep{episode + 1}")
    # Save final phase 2 model
    agent.save_model("bipedal_phase2_final")
    # Phase 3: Sword Fighting Training
    print("\nPhase 3: Sword Fighting Training")
    print("-------------------------------")
    # Restore original reward function with sword fighting components

    env._calculate_reward = original_reward_fn

    # Train for sword fighting with balance
    for episode in range(episodes_per_phase):
        state = env.reset()
        episode_reward = 0
        done = False
        while not done:
            action, log_prob, value = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.store_experience(state, action, reward, value, log_prob, done)
            if agent.buffer.size() >= 128:
                agent.learn()
            state = next_state
            episode_reward += reward
        if agent.buffer.size() > 0:
            agent.learn()
        print(f"Episode {episode + 1}/{episodes_per_phase}, Reward: {episode_reward:.2f}")
        # Save phase 3 model
        if (episode + 1) % 10 == 0:
            agent.save_model(f"bipedal_phase3_ep{episode + 1}")

    # Save final complete model
    agent.save_model("bipedal_final")

    # Final visualization with rendering
    if render_final:
        print("\nFinal Training Visualization")
        env.close()
        # Create new environment with rendering
        vis_env = SwordFighterEnv(render=True)
        # Load the trained model
        agent.load_model("bipedal_final")
        # Run a few episodes for visualization
        for episode in range(5):
            state = vis_env.reset()
            episode_reward = 0
            done = False
            step = 0
            while not done and step < 500:
                action, _, _ = agent.act(state)
                next_state, reward, done, _ = vis_env.step(action)
                state = next_state
                episode_reward += reward
                step += 1
                import time
                time.sleep(0.01)  # Slow down for better visualization
            print(f"Visualization Episode {episode + 1}/5, Reward: {episode_reward:.2f}")
        vis_env.close()
    print("\nCurriculum training complete!")
    return agent


def test(episodes=5, model_name="best_model"):
    """Test the agent with balance visualization"""
    # Create environment
    env = SwordFighterEnv(render=True)

    # Enable debug visualization for balance
    P.configureDebugVisualizer(P.COV_ENABLE_GUI, 1)
    P.configureDebugVisualizer(P.COV_ENABLE_RGB_BUFFER_PREVIEW, 1)
    P.configureDebugVisualizer(P.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
    P.configureDebugVisualizer(P.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)

    # Add visual indicators for balance points
    balance_indicators = {}

    # Create agent
    agent = PPOAgent(env.observation_space_size, env.action_space_size)

    # Load model
    loaded = agent.load_model(model_name)
    if not loaded:
        print("Using untrained model")

    # Testing loop
    total_rewards = []
    balance_metrics = []

    try:
        for episode in range(episodes):
            # Reset environment
            state = env.reset()
            episode_reward = 0
            episode_balance_metrics = []

            print(f"Starting test episode {episode + 1}/{episodes}")

            # Clear previous balance indicators
            for item_id in balance_indicators.values():
                P.removeBody(item_id)
            balance_indicators = {}

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

                # Visualize balance - every 10 steps to avoid visual clutter
                if step % 10 == 0:
                    # Get torso position and orientation
                    pos, orn = P.getBasePositionAndOrientation(env.fighter1["torso"])

                    # Get rotation matrix to extract up vector
                    rotation_matrix = P.getMatrixFromQuaternion(orn)
                    up_vector = [rotation_matrix[2], rotation_matrix[5], rotation_matrix[8]]

                    # Calculate upright alignment score
                    upright_alignment = up_vector[2]

                    # Get feet positions
                    feet_on_ground = 0
                    foot_positions = []
                    for i in range(4):
                        try:
                            ankle_joint = env.fighter1["joints"][f"ankle_{i}"]
                            constraint_info = P.getConstraintInfo(ankle_joint)
                            foot_id = constraint_info[2]

                            # Get foot position
                            foot_pos, _ = P.getBasePositionAndOrientation(foot_id)
                            foot_positions.append(foot_pos)

                            # Check if foot touches ground
                            contact_points = P.getContactPoints(foot_id, env.plane_id)
                            if len(contact_points) > 0:
                                feet_on_ground += 1
                        except:
                            pass

                    # Calculate center of support (average of feet on ground)
                    if foot_positions:
                        center_of_support = [sum(p[0] for p in foot_positions) / len(foot_positions),
                                             sum(p[1] for p in foot_positions) / len(foot_positions),
                                             0.01]  # Slightly above ground
                    else:
                        center_of_support = [pos[0], pos[1], 0.01]

                    # Project center of mass onto ground
                    projected_com = [pos[0], pos[1], 0.01]

                    # Calculate balance metrics
                    balance_status = {
                        "upright_alignment": upright_alignment,
                        "feet_on_ground": feet_on_ground,
                        "height": pos[2],
                        "distance_com_to_support": ((projected_com[0] - center_of_support[0]) ** 2 +
                                                    (projected_com[1] - center_of_support[1]) ** 2) ** 0.5
                    }
                    episode_balance_metrics.append(balance_status)

                    # Visualize balance indicators
                    # Center of support (green sphere)
                    if "support_center" in balance_indicators:
                        P.resetBasePositionAndOrientation(
                            balance_indicators["support_center"],
                            center_of_support,
                            [0, 0, 0, 1]
                        )
                    else:
                        balance_indicators["support_center"] = P.createMultiBody(
                            baseMass=0,
                            basePosition=center_of_support,
                            baseOrientation=[0, 0, 0, 1],
                            baseCollisionShapeIndex=P.createCollisionShape(P.GEOM_SPHERE, radius=0.1),
                            baseVisualShapeIndex=P.createVisualShape(P.GEOM_SPHERE, radius=0.1,
                                                                     rgbaColor=[0, 1, 0, 0.5])
                        )

                    # Projected center of mass (red sphere)
                    if "projected_com" in balance_indicators:
                        P.resetBasePositionAndOrientation(
                            balance_indicators["projected_com"],
                            projected_com,
                            [0, 0, 0, 1]
                        )
                    else:
                        balance_indicators["projected_com"] = P.createMultiBody(
                            baseMass=0,
                            basePosition=projected_com,
                            baseOrientation=[0, 0, 0, 1],
                            baseCollisionShapeIndex=P.createCollisionShape(P.GEOM_SPHERE, radius=0.1),
                            baseVisualShapeIndex=P.createVisualShape(P.GEOM_SPHERE, radius=0.1,
                                                                     rgbaColor=[1, 0, 0, 0.5])
                        )

                    # Line connecting them (stability indicator)
                    P.addUserDebugLine(
                        center_of_support,
                        projected_com,
                        lineColorRGB=[1, 1, 0],
                        lineWidth=2,
                        lifeTime=0.5  # Short lifetime so it updates
                    )

                    # Add upright vector indicator
                    up_line_end = [
                        pos[0] + up_vector[0] * 0.5,
                        pos[1] + up_vector[1] * 0.5,
                        pos[2] + up_vector[2] * 0.5
                    ]
                    P.addUserDebugLine(
                        pos,
                        up_line_end,
                        lineColorRGB=[0, 0, 1],
                        lineWidth=3,
                        lifeTime=0.5
                    )

                # Print progress occasionally
                if step % 50 == 0:
                    if episode_balance_metrics:
                        latest = episode_balance_metrics[-1]
                        print(f"  Step {step}, Reward: {episode_reward:.2f}")
                        print(f"  Balance - Upright: {latest['upright_alignment']:.2f}, " +
                              f"Feet on ground: {latest['feet_on_ground']}, Height: {latest['height']:.2f}")

                # Slow down for better visualization
                import time
                time.sleep(0.01)

            # Print episode results
            total_rewards.append(episode_reward)
            balance_metrics.append(episode_balance_metrics)

            # Calculate average balance metrics for this episode
            if episode_balance_metrics:
                avg_upright = sum(m['upright_alignment'] for m in episode_balance_metrics) / len(
                    episode_balance_metrics)
                avg_feet = sum(m['feet_on_ground'] for m in episode_balance_metrics) / len(episode_balance_metrics)
                avg_height = sum(m['height'] for m in episode_balance_metrics) / len(episode_balance_metrics)

                print(f"Test episode {episode + 1}/{episodes}, Reward: {episode_reward:.2f}")
                print(f"  Average Balance - Upright: {avg_upright:.2f}, " +
                      f"Feet on ground: {avg_feet:.2f}, Height: {avg_height:.2f}")

    except KeyboardInterrupt:
        print("Testing interrupted by user")
    except Exception as e:
        print(f"Error during testing: {e}")
    finally:
        # Clear all visual indicators
        for item_id in balance_indicators.values():
            P.removeBody(item_id)

        # Close environment
        env.close()

    return total_rewards, balance_metrics


if __name__ == "__main__":
    import argparse
    import types  # Need to import this for method binding

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", action="store_true", help="Render the environment")
    parser.add_argument("--train", action="store_true", help="Train the agent")
    parser.add_argument("--test", action="store_true", help="Test the agent")
    parser.add_argument("--curriculum", action="store_true", help="Use curriculum training for bipedal fighter")
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes")
    parser.add_argument("--model", type=str, default="best_model", help="Model name")
    args = parser.parse_args()

    # Create models directory
    os.makedirs("./models", exist_ok=True)

    # Run training or testing
    if args.curriculum:
        print("Starting curriculum training for bipedal fighter...")
        # Define number of phases and episodes per phase
        num_phases = 3
        episodes_per_phase = args.episodes // num_phases  # Divide total episodes across phases
        if episodes_per_phase < 10:
            episodes_per_phase = 10  # Ensure at least 10 episodes per phase

        # Call the curriculum training function
        curriculum_training(
            episodes_per_phase=episodes_per_phase,
            render_final=args.render
        )
    elif args.train:
        print("Training agent...")
        train(num_episodes=args.episodes, render=args.render)
    elif args.test:
        print("Testing agent...")
        test(episodes=5, model_name=args.model)
    # Default to training if no arguments are provided
    else:
        print("No arguments provided, running default training...")
        train(num_episodes=args.episodes)

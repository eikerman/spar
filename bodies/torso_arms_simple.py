import pybullet as p


class FighterTorsoArmsSimple:

    torso_height = 0.6
    torso_width = 0.4
    torso_depth = 0.2
    arm_length = 0.4
    arm_radius = 0.04
    sword_length = 1.0
    base_mass = 10
    arm_mass = base_mass * 0.2
    sword_mass = 1

    def __init__(self, position):
        self.position = position
        self.fighter = self._create_fighter(position)

    def get_fighter(self):
        return self.fighter

    def get_position(self):
        return self.position

    def _create_fighter(self, position):
        """Create a simplified skeleton fighter with a sword"""

        # Create torso with mass centered at the bottom for better balance
        torso = p.createMultiBody(
            baseMass=self.base_mass,
            basePosition=position,
            baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.2, 0.1, 0.3]),
            baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=[0.2, 0.1, 0.3],
                                                     rgbaColor=[0.7, 0.7, 0.7, 1])
        )
        right_arm = p.createMultiBody(
            baseMass=self.arm_mass,
            basePosition=[position[0] + 0.1, position[1], position[2] + 0.3],
            baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=[self.arm_length / 2, self.arm_radius, self.arm_radius]),
            baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=[self.arm_length / 2, self.arm_radius, self.arm_radius],
                                                     rgbaColor=[0.7, 0.7, 0.7, 1])

        )

        right_shoulder = p.createConstraint(
            parentBodyUniqueId=torso,
            parentLinkIndex=-1,
            childBodyUniqueId=right_arm,
            childLinkIndex=-1,
            jointType=p.JOINT_POINT2POINT,
            jointAxis=[1, 1, 1],
            parentFramePosition=[0.2, 0, 0],
            childFramePosition=[-self.arm_length / 2, 0, 0]
        )

        left_arm = p.createMultiBody(
            baseMass=self.arm_mass,
            basePosition=[position[0] - 0.1, position[1], position[2] + 0.3],
            baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=[self.arm_length / 2, self.arm_radius, self.arm_radius]),
            baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=[self.arm_length / 2, self.arm_radius, self.arm_radius],
                                                     rgbaColor=[0.7, 0.7, 0.7, 1])

        )

        left_shoulder = p.createConstraint(
            parentBodyUniqueId=torso,
            parentLinkIndex=-1,
            childBodyUniqueId=left_arm,
            childLinkIndex=-1,
            jointType=p.JOINT_POINT2POINT,
            jointAxis=[1, 1, 1],
            parentFramePosition=[-0.2, 0, 0],
            childFramePosition=[-self.arm_length / 2, 0, 0]
        )

        #p.changeDynamics(
        #    torso,
        #    -1,
        #    mass=self.base_mass,
        #    lateralFriction=0.5,
        #    spinningFriction=0.5,
        #    rollingFriction=0.5
        #)
        #p.changeDynamics(
        #    right_arm,
        #    -1,
        #    mass=self.arm_mass,
        #    lateralFriction=0.5,
        #    spinningFriction=0.5,
        #    rollingFriction=0.5
        #)
#
        #p.changeDynamics(
        #    left_arm,
        #    -1,
        #    mass=self.arm_mass,
        #    lateralFriction=0.5,
        #    spinningFriction=0.5,
        #    rollingFriction=0.5
        #)


        return {
            "torso": torso,
            "right_arm": right_arm,
            "left_arm": left_arm,
            "joints": [left_shoulder, right_shoulder]  # Would store joint IDs in full implementation
        }

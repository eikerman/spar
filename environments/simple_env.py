import pybullet as p
import pybullet_data


class SimpleFighterEnv:
    """Environment for sword fighting simulation using PyBullet"""

    def __init__(self, render=True):
        # Initialize PyBullet
        self.render_mode = render
        if render:
            self.client = p.connect(p.GUI)  # For visualization
            # Configure GUI
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
            p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)

            # Add a title to the visualizer window
            p.setPhysicsEngineParameter(enableFileCaching=0)
            p.resetDebugVisualizerCamera(3, 90, -30, [0, 0, 0])

        else:
            self.client = p.connect(p.DIRECT)  # Headless mode for faster training

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, 9.81)  # Earth-like gravity

        # Set up physics parameters for better stability and interaction
        p.setPhysicsEngineParameter(fixedTimeStep=1.0 / 240.0)  # Smaller timestep for stability
        p.setPhysicsEngineParameter(numSolverIterations=50)  # More solver iterations
        p.setPhysicsEngineParameter(numSubSteps=10)

        # Load ground plane with a more interesting texture
        self.plane_id = p.loadURDF("plane.urdf")
        p.changeDynamics(
            self.plane_id,
            -1,
            lateralFriction=0,
            restitution=0
        )

        # Set a nice background color
        p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
        p.resetDebugVisualizerCamera(
            cameraDistance=3.0,
            cameraYaw=0,
            cameraPitch=-30,
            cameraTargetPosition=[0, 0, 0]
        )

        # Add some objects to the environment for interaction

    def reset(self):
        """Reset the environment"""
        p.resetSimulation()
        p.setGravity(0, 0, )

        # Load ground plane
        self.plane_id = p.loadURDF("plane.urdf")
        p.changeDynamics(self.plane_id, -1, lateralFriction=0, restitution=0.0)

        # Add environment objects again
        return None  # Would return observation in a full RL environment

    def close(self):
        """Close the environment"""
        p.disconnect()
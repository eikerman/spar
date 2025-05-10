from bodies import torso_arms_simple
from environments import simple_env
import pybullet as p
import time


def main():
    # Create a simple environment
    simple_env.SimpleFighterEnv()

    # Create a torso and arms
    torso_arms = torso_arms_simple.FighterTorsoArmsSimple(position=[0, 0, 0])
    fighter = torso_arms.get_fighter()
    # Set up camera view
    p.resetDebugVisualizerCamera(
        cameraDistance=3,
        cameraYaw=0,
        cameraPitch=-30,
        cameraTargetPosition=[0, 0, 1]
    )
    gravity = p.getPhysicsEngineParameters()['gravityAccelerationX'], p.getPhysicsEngineParameters()[
        'gravityAccelerationY'], p.getPhysicsEngineParameters()['gravityAccelerationZ']
    print(f"Current gravity setting: {gravity}")
    # Add button for reset
    reset_button = p.addUserDebugParameter("Reset Simulation", 1, 0, 0)

    # Add key binding information display
    key_info = [
        "Controls:",
        "Reset - Reset the simulation"
    ]

    # Store previous button states to detect changes
    prev_reset = 0

    # Current sword angle

    # Main simulation loop
    while True:
        # Get current slider values for interactive control
        curr_reset = p.readUserDebugParameter(reset_button)

        # Check for button presses (detect changes)
        if curr_reset != prev_reset:
            # Reset the simulation
            p.resetBasePositionAndOrientation(
                fighter["torso"],
                [0, 0, 0],
                p.getQuaternionFromEuler([0, 0, 0])
            )
            p.resetBaseVelocity(fighter["torso"], [0, 0, 0], [0, 0, 0])
            p.resetBasePositionAndOrientation(
                fighter["right_arm"],
                [0.1, 0, 0],
                p.getQuaternionFromEuler([0, 0, 0])
            )
            p.resetBaseVelocity(fighter["right_arm"], [0, 0, 0], [0, 0, 0])
            p.resetBasePositionAndOrientation(
                fighter["left_arm"],
                [-0.1, 0, 0],
                p.getQuaternionFromEuler([0, 0, 0])
            )
            p.resetBaseVelocity(fighter["left_arm"], [0, 0, 0], [0, 0, 0])

            prev_reset = curr_reset

        # Step the simulation
        p.stepSimulation()

        # Display some debug info
        torso_pos, _ = p.getBasePositionAndOrientation(fighter["torso"])

        # Add a small delay to not overwhelm the CPU
        time.sleep(0.0005)

        # Exit if window is closed
        if not p.isConnected():
            break

    # Disconnect when done
    p.disconnect()


if __name__ == "__main__":
    main()

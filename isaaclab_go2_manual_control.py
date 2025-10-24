from isaacsim.simulation_app import SimulationApp

# Launch Isaac Sim
simulation_app = SimulationApp({"headless": False})

from isaacsim.core.api import World
from isaacsim.core.experimental.prims import Articulation
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.storage.native import get_assets_root_path
import numpy as np

# --------------------------------------------------
# Replace this path with your local or Nucleus USD
# --------------------------------------------------
GO2_USD_PATH = get_assets_root_path() + "/Isaac/Robots/Unitree/Go2/go2.usd"
GO2_PRIM_PATH = "/World/Go2"

# Create world and ground plane
world = World(stage_units_in_meters=1.0)
world.scene.add_default_ground_plane()

# Add the Go2 robot to the stage
add_reference_to_stage(GO2_USD_PATH, GO2_PRIM_PATH)
go2 = Articulation(paths=GO2_PRIM_PATH)

# Initialize physics scene
world.reset()

# --------------------------------------------------
# After stage loads, you can set arbitrary joint positions
# --------------------------------------------------

# Wait until articulation is initialized
world.step(render=True)
joint_names = go2.dof_names
num_joints = len(joint_names)
print(f"Go2 DOF count: {num_joints}")

# Create some arbitrary joint positions
# For example, small alternating offsets
q = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float32)

# Apply joint positions
go2.set_dof_positions(q)

# Print applied joints
for name, val in zip(joint_names, q):
    print(f"{name}: {val:.3f}")

# Step a few frames to visualize
for _ in range(2000):
    world.step(render=True)

simulation_app.close()

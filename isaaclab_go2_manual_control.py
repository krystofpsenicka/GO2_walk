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
go2.set_world_poses([[0, 0, 0.4]])
go2.set_link_masses([0])

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
q_mj = np.array([-0.2, -0.2, -1, -0.2, -0.2, -1, -0.2, -0.2, -1, -0.2, -0.2, -1], dtype=np.float32)

ISAAC_TO_MJ = np.array([0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11], dtype=int)
MJ_TO_ISAAC = np.argsort(ISAAC_TO_MJ)

# Apply joint positions
go2.set_dof_positions(q_mj[MJ_TO_ISAAC])

# Print applied joints
for name, val in zip(joint_names, q_mj[MJ_TO_ISAAC]):
    print(f"{name}: {val:.3f}")

# Step a few frames to visualize
for _ in range(20000):
    world.step(render=True)

simulation_app.close()

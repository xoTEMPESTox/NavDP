"""
Script to fix the LeKiwi USD file by:
1. Removing/deactivating the root_joint
2. Ensuring only base_plate_layer1_v5 has ArticulationRootAPI
"""
import os
os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

from pxr import Usd, UsdPhysics, Sdf, PhysxSchema

# Load the ORIGINAL USD file with edit intent
usd_path = "/home/tempest/code/NavDP/assets/robots/lekiwi/lekiwi.usd"
output_path = "/home/tempest/code/NavDP/assets/robots/lekiwi/lekiwi_floating.usd"

# Open with edit intent
stage = Usd.Stage.Open(usd_path)

print("=== Analyzing original structure ===")

# Find the root layer to edit
root_layer = stage.GetRootLayer()
print(f"Root layer: {root_layer.identifier}")

# Create a new stage for the fixed version
new_stage = Usd.Stage.CreateNew(output_path)
new_stage.GetRootLayer().subLayerPaths.append(usd_path)

# Set default prim
default_prim = stage.GetDefaultPrim()
if default_prim:
    new_stage.SetDefaultPrim(new_stage.GetPrimAtPath(default_prim.GetPath()))

print(f"\n=== Making fixes in new layer ===")

# Get the root_joint in new stage and mark it as inactive
root_joint_path = "/LeKiwi/root_joint"
root_joint_prim = new_stage.GetPrimAtPath(root_joint_path)
if root_joint_prim.IsValid():
    # Set the prim as inactive - this effectively "deletes" it in this layer
    root_joint_prim.SetActive(False)
    print(f"Deactivated {root_joint_path}")

# Ensure base_plate_layer1_v5 has ArticulationRootAPI
base_body_path = "/LeKiwi/base_plate_layer1_v5"  
base_body_prim = new_stage.GetPrimAtPath(base_body_path)
if base_body_prim.IsValid():
    if not base_body_prim.HasAPI(UsdPhysics.ArticulationRootAPI):
        UsdPhysics.ArticulationRootAPI.Apply(base_body_prim)
        print(f"Applied ArticulationRootAPI to {base_body_path}")
    
    if not base_body_prim.HasAPI(PhysxSchema.PhysxArticulationAPI):
        PhysxSchema.PhysxArticulationAPI.Apply(base_body_prim)
        print(f"Applied PhysxArticulationAPI to {base_body_path}")
else:
    print(f"WARNING: {base_body_path} not found!")

# Verify
print(f"\n=== Verification ===")
for prim in new_stage.Traverse():
    if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
        print(f"ArticulationRootAPI on: {prim.GetPath()} (active: {prim.IsActive()})")

# Check root_joint status
rj = new_stage.GetPrimAtPath(root_joint_path)
print(f"root_joint active: {rj.IsActive() if rj.IsValid() else 'N/A'}")

new_stage.Save()
print(f"\nSaved to: {output_path}")

simulation_app.close()
print("Done!")

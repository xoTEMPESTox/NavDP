"""
Inspect the LeKiwi USD joint structure to understand wheel connections.
"""
import os
os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

from pxr import Usd, UsdPhysics, UsdGeom

usd_path = "/home/tempest/code/NavDP/assets/robots/lekiwi/lekiwi_floating.usd"
stage = Usd.Stage.Open(usd_path)

print("=" * 60)
print("JOINT ANALYSIS FOR LEKIWI ROBOT")
print("=" * 60)

# Find all revolute joints
print("\n=== ALL REVOLUTE JOINTS ===")
revolute_joints = []
for prim in stage.Traverse():
    if prim.GetTypeName() == "PhysicsRevoluteJoint":
        revolute_joints.append(prim)
        print(f"\nJoint: {prim.GetPath()}")
        
        # Get joint connections
        if prim.IsA(UsdPhysics.Joint):
            joint = UsdPhysics.Joint(prim)
            body0 = joint.GetBody0Rel().GetTargets()
            body1 = joint.GetBody1Rel().GetTargets()
            print(f"  Body0 (parent): {body0}")
            print(f"  Body1 (child): {body1}")

print(f"\nTotal revolute joints: {len(revolute_joints)}")

# Find all rigid bodies
print("\n\n=== RIGID BODIES WITH 'wheel' OR 'motor' IN NAME ===")
for prim in stage.Traverse():
    name_lower = prim.GetName().lower()
    if "wheel" in name_lower or "motor" in name_lower:
        print(f"{prim.GetPath()} - Type: {prim.GetTypeName()}")

# Check for fixed joints that might be locking wheels
print("\n\n=== FIXED JOINTS (may prevent rotation) ===")
for prim in stage.Traverse():
    if prim.GetTypeName() == "PhysicsFixedJoint":
        print(f"\nFixed Joint: {prim.GetPath()}")
        if prim.IsA(UsdPhysics.Joint):
            joint = UsdPhysics.Joint(prim)
            body0 = joint.GetBody0Rel().GetTargets()
            body1 = joint.GetBody1Rel().GetTargets()
            print(f"  Body0: {body0}")
            print(f"  Body1: {body1}")

simulation_app.close()
print("\nDone!")

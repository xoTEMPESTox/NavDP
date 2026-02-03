import os
os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

from pxr import Usd, UsdPhysics, PhysxSchema
import omni.usd

# Load the USD file
stage = Usd.Stage.Open("/home/tempest/code/NavDP/assets/robots/lekiwi/lekiwi.usd")

# Find root_joint
root_joint_prim = stage.GetPrimAtPath("/root_joint")
if not root_joint_prim.IsValid():
    # Try without leading slash
    for prim in stage.Traverse():
        if "root_joint" in prim.GetPath().pathString:
            root_joint_prim = prim
            break

if root_joint_prim.IsValid():
    print(f"Found root_joint at: {root_joint_prim.GetPath()}")
    print(f"  Type: {root_joint_prim.GetTypeName()}")
    print(f"  Has ArticulationRootAPI: {root_joint_prim.HasAPI(UsdPhysics.ArticulationRootAPI)}")
    print(f"  Has PhysxArticulationAPI: {root_joint_prim.HasAPI(PhysxSchema.PhysxArticulationAPI)}")
    print(f"  Has RevoluteJoint: {root_joint_prim.HasAPI(UsdPhysics.RevoluteJoint) if hasattr(UsdPhysics, 'RevoluteJoint') else 'N/A'}")
    
    # Get all applied schemas
    schemas = root_joint_prim.GetAppliedSchemas()
    print(f"  Applied Schemas: {schemas}")
    
    # Check if it's a joint and what type
    if root_joint_prim.IsA(UsdPhysics.Joint):
        print("  Is a UsdPhysics.Joint")
        joint = UsdPhysics.Joint(root_joint_prim)
        body0 = joint.GetBody0Rel().GetTargets()
        body1 = joint.GetBody1Rel().GetTargets()
        print(f"  Body0 targets: {body0}")
        print(f"  Body1 targets: {body1}")
else:
    print("root_joint not found. Listing all prims:")
    for prim in stage.Traverse():
        if prim.GetTypeName() and "Joint" in prim.GetTypeName():
            print(f"  {prim.GetPath()} - {prim.GetTypeName()}")

# Check default prim
default_prim = stage.GetDefaultPrim()
if default_prim:
    print(f"\nDefault Prim: {default_prim.GetPath()}")
    print(f"  Type: {default_prim.GetTypeName()}")

simulation_app.close()

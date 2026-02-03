"""
Check physics properties of wheel bodies
"""
import os
os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

from pxr import Usd, UsdPhysics, UsdGeom, PhysxSchema

usd_path = "/home/tempest/code/NavDP/assets/robots/lekiwi/lekiwi_floating.usd"
stage = Usd.Stage.Open(usd_path)

print("=" * 60)
print("WHEEL PHYSICS ANALYSIS")
print("=" * 60)

wheel_paths = [
    "/LeKiwi/a__Omni_Directional_Wheel_Single_Body_v1",
    "/LeKiwi/a__Omni_Directional_Wheel_Single_Body_v1_1", 
    "/LeKiwi/a__Omni_Directional_Wheel_Single_Body_v1_2",
    "/LeKiwi/omni_wheel_mount_v5",
    "/LeKiwi/omni_wheel_mount_v5_1",
    "/LeKiwi/omni_wheel_mount_v5_2",
]

for path in wheel_paths:
    prim = stage.GetPrimAtPath(path)
    if prim.IsValid():
        print(f"\n{path}")
        print(f"  Type: {prim.GetTypeName()}")
        print(f"  Has RigidBodyAPI: {prim.HasAPI(UsdPhysics.RigidBodyAPI)}")
        print(f"  Has CollisionAPI: {prim.HasAPI(UsdPhysics.CollisionAPI)}")
        print(f"  Has MassAPI: {prim.HasAPI(UsdPhysics.MassAPI)}")
        
        # Look for collision children
        has_collision = False
        for child in prim.GetAllChildren():
            if child.HasAPI(UsdPhysics.CollisionAPI) or "collision" in child.GetName().lower():
                has_collision = True
                print(f"  Child with collision: {child.GetPath()}")
        
        if not has_collision:
            print("  WARNING: No collision found!")
            
        # Check for mass
        if prim.HasAPI(UsdPhysics.MassAPI):
            mass_api = UsdPhysics.MassAPI(prim)
            mass = mass_api.GetMassAttr().Get()
            print(f"  Mass: {mass}")

# Also check the revolute joints for drive properties
print("\n\n" + "=" * 60)
print("WHEEL JOINT DRIVE PROPERTIES")
print("=" * 60)

wheel_joints = [
    "/LeKiwi/joints/ST3215_Servo_Motor_v1_Revolute_64",
    "/LeKiwi/joints/ST3215_Servo_Motor_v1_1_Revolute_62",
    "/LeKiwi/joints/ST3215_Servo_Motor_v1_2_Revolute_60",
]

for path in wheel_joints:
    prim = stage.GetPrimAtPath(path)
    if prim.IsValid():
        print(f"\n{path}")
        
        # Check for drive API
        if prim.HasAPI(UsdPhysics.DriveAPI):
            print("  Has DriveAPI: True")
        else:
            print("  Has DriveAPI: False - NO DRIVE CONFIGURED IN USD!")
            
        # Check for PhysX joint API
        if prim.HasAPI(PhysxSchema.PhysxJointAPI):
            print("  Has PhysxJointAPI: True")

simulation_app.close()
print("\nDone!")

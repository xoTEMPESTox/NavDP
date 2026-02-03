import os
os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})
from pxr import Usd, UsdPhysics

stage = Usd.Stage.Open("/home/tempest/code/NavDP/assets/robots/lekiwi/lekiwi_floating.usd")

# Check wheel bodies
for name in ["a__Omni_Directional_Wheel_Single_Body_v1", "omni_wheel_mount_v5"]:
    prim = stage.GetPrimAtPath(f"/LeKiwi/{name}")
    if prim:
        print(f"{name}: RigidBody={prim.HasAPI(UsdPhysics.RigidBodyAPI)}, Collision={prim.HasAPI(UsdPhysics.CollisionAPI)}")

# Check revolute joint for DriveAPI
for name in ["ST3215_Servo_Motor_v1_Revolute_64"]:
    prim = stage.GetPrimAtPath(f"/LeKiwi/joints/{name}")
    if prim:
        print(f"{name}: DriveAPI={prim.HasAPI(UsdPhysics.DriveAPI)}")

simulation_app.close()

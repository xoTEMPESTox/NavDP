import os
os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})
from pxr import Usd, UsdPhysics, UsdGeom

stage = Usd.Stage.Open("/home/tempest/code/NavDP/assets/robots/lekiwi/lekiwi_floating.usd")

print("=== ALL PRIMS WITH COLLISION ===")
collision_count = 0
for prim in stage.Traverse():
    if prim.HasAPI(UsdPhysics.CollisionAPI):
        collision_count += 1
        # Only print wheel-related ones
        if "wheel" in prim.GetPath().pathString.lower():
            print(f"{prim.GetPath()}")

print(f"\nTotal collision prims: {collision_count}")

print("\n=== CHECKING WHEEL CHILDREN ===")
wheel_path = "/LeKiwi/a__Omni_Directional_Wheel_Single_Body_v1"
prim = stage.GetPrimAtPath(wheel_path)
if prim:
    for child in Usd.PrimRange(prim):
        if child.HasAPI(UsdPhysics.CollisionAPI) or "collision" in child.GetName().lower():
            print(f"  Found collision: {child.GetPath()}")

simulation_app.close()

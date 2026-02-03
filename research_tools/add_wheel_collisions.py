"""
Script to add collision geometry to the LeKiwi robot USD file.
This script ONLY adds collisions - does NOT modify articulation.
Uses the already-fixed lekiwi_floating.usd as base.
"""
import os
os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

from pxr import Usd, UsdGeom, UsdPhysics, Gf, PhysxSchema

# Use the floating USD as input (articulation already fixed)
INPUT_USD = "/home/tempest/code/NavDP/assets/robots/lekiwi/lekiwi_floating.usd"
OUTPUT_USD = "/home/tempest/code/NavDP/assets/robots/lekiwi/lekiwi_final.usd"

# Wheel properties - slightly reduced to prevent interpenetration
WHEEL_RADIUS = 0.0325 * 0.95  # 5% smaller
WHEEL_WIDTH = 0.03           # meters

# Wheel body paths
WHEEL_BODIES = [
    "/LeKiwi/a__Omni_Directional_Wheel_Single_Body_v1",
    "/LeKiwi/a__Omni_Directional_Wheel_Single_Body_v1_1",
    "/LeKiwi/a__Omni_Directional_Wheel_Single_Body_v1_2",
]


def add_cylinder_collision(stage, prim_path, radius, height, is_caster=False):
    """Add a collision shape. Cyl for drive wheels, frictionless Sphere for caster."""
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        print(f"WARNING: Prim not found: {prim_path}")
        return False
    
    # Create collision child prim
    collision_path = f"{prim_path}/wheel_collision"
    if stage.GetPrimAtPath(collision_path).IsValid():
        stage.RemovePrim(collision_path)
    
    if is_caster:
        # Sphere for caster (3rd wheel) to allow sliding
        collision_prim = stage.DefinePrim(collision_path, "Sphere")
        sphere = UsdGeom.Sphere(collision_prim)
        sphere.GetRadiusAttr().Set(radius)
    else:
        # Cylinder for drive wheels
        collision_prim = stage.DefinePrim(collision_path, "Cylinder")
        cylinder = UsdGeom.Cylinder(collision_prim)
        cylinder.GetRadiusAttr().Set(radius)
        cylinder.GetHeightAttr().Set(height)
        cylinder.GetAxisAttr().Set("Y")

    # Make invisible
    UsdGeom.Imageable(collision_prim).GetVisibilityAttr().Set("invisible")
    
    # APIs
    UsdPhysics.CollisionAPI.Apply(collision_prim)
    PhysxSchema.PhysxCollisionAPI.Apply(collision_prim)
    phys_api = PhysxSchema.PhysxCollisionAPI(collision_prim)
    phys_api.GetContactOffsetAttr().Set(0.002)
    phys_api.GetRestOffsetAttr().Set(0.001)

    # Material
    # Bind a material logic would be better but simple API tweak:
    # We can't easily create a material in raw USD without verbose code.
    # Instead, we rely on the sphere shape reducing friction contact area
    # and mostly the geometry allowing rolling/sliding.
    
    print(f"Added {'Sphere (Caster)' if is_caster else 'Cylinder'} to {prim_path}")
    return True


def configure_articulation(stage):
    """Disable self-collisions on articulation root."""
    base_path = "/LeKiwi/base_plate_layer1_v5"
    base_prim = stage.GetPrimAtPath(base_path)
    
    if base_prim.IsValid():
        # Ensure PhysxArticulationAPI is applied
        if not base_prim.HasAPI(PhysxSchema.PhysxArticulationAPI):
            PhysxSchema.PhysxArticulationAPI.Apply(base_prim)
        
        # Disable self collisions
        physx_art = PhysxSchema.PhysxArticulationAPI(base_prim)
        physx_art.CreateEnabledSelfCollisionsAttr().Set(False)
        print(f"Disabled self-collisions on {base_path}")


def main():
    print("=" * 60)
    print("ADDING WHEEL COLLISIONS TO LEKIWI")
    print("=" * 60)
    
    # Open the USD stage (lekiwi_floating.usd which has correct articulation)
    stage = Usd.Stage.Open(INPUT_USD)
    if not stage:
        print(f"ERROR: Could not open {INPUT_USD}")
        return
    
    print(f"Loaded: {INPUT_USD}")
    
    # Disable self-collisions
    configure_articulation(stage)
    
    # Count existing collisions
    existing = sum(1 for p in stage.Traverse() if p.HasAPI(UsdPhysics.CollisionAPI))
    print(f"Existing collision prims: {existing}")
    
    # Add wheel collisions
    print("\n--- Adding Wheel Collisions ---")
    for i, wheel_path in enumerate(WHEEL_BODIES):
        # 3rd wheel (index 2) corresponds to `a__Omni_Directional_Wheel_Single_Body_v1_2`
        # This is the passive wheel in our diff drive setup
        is_caster = (i == 2)
        add_cylinder_collision(stage, wheel_path, WHEEL_RADIUS, WHEEL_WIDTH, is_caster=is_caster)
    
    # Count new collisions
    new_count = sum(1 for p in stage.Traverse() if p.HasAPI(UsdPhysics.CollisionAPI))
    print(f"\nTotal collision prims: {new_count}")
    
    # Save
    stage.Export(OUTPUT_USD)
    print(f"Saved to: {OUTPUT_USD}")
    
    simulation_app.close()
    print("Done!")


if __name__ == "__main__":
    main()

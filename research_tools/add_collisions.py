"""
Script to add collision geometry to the LeKiwi robot USD file.
Uses proper physics material and collision configuration.
"""
import os
os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

from pxr import Usd, UsdGeom, UsdPhysics, Gf, Sdf, PhysxSchema
import omni.usd

# Input and output paths - use the original lekiwi.usd as base
INPUT_USD = "/home/tempest/code/NavDP/assets/robots/lekiwi/lekiwi.usd"  
OUTPUT_USD = "/home/tempest/code/NavDP/assets/robots/lekiwi/lekiwi_with_collisions.usd"

# Wheel properties
WHEEL_RADIUS = 0.0325  # meters
WHEEL_WIDTH = 0.03    # meters

# Bodies to add collision to
WHEEL_BODIES = [
    "/LeKiwi/a__Omni_Directional_Wheel_Single_Body_v1",
    "/LeKiwi/a__Omni_Directional_Wheel_Single_Body_v1_1",
    "/LeKiwi/a__Omni_Directional_Wheel_Single_Body_v1_2",
]

# Main body parts that need collision
BODY_PARTS = [
    ("/LeKiwi/base_plate_layer1_v5", (0.18, 0.18, 0.015)),
]


def add_cylinder_collision(stage, prim_path, radius, height):
    """Add a cylinder collision shape as a child of the prim."""
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        print(f"WARNING: Prim not found: {prim_path}")
        return False
    
    # Create collision child prim
    collision_path = f"{prim_path}/collision_shape"
    
    # Remove if exists
    existing = stage.GetPrimAtPath(collision_path)
    if existing.IsValid():
        stage.RemovePrim(collision_path)
    
    # Create a Cylinder for collision
    collision_prim = stage.DefinePrim(collision_path, "Cylinder")
    
    # Set cylinder geometry
    cylinder = UsdGeom.Cylinder(collision_prim)
    cylinder.GetRadiusAttr().Set(radius)
    cylinder.GetHeightAttr().Set(height)
    cylinder.GetAxisAttr().Set("Y")  # Wheel axis
    
    # Make it invisible (collision only, no rendering)
    imageable = UsdGeom.Imageable(collision_prim)
    imageable.GetVisibilityAttr().Set("invisible")
    
    # Apply collision API  
    UsdPhysics.CollisionAPI.Apply(collision_prim)
    
    # Set physics material properties for better friction
    if collision_prim.HasAPI(PhysxSchema.PhysxCollisionAPI):
        phys_api = PhysxSchema.PhysxCollisionAPI(collision_prim)
    else:
        phys_api = PhysxSchema.PhysxCollisionAPI.Apply(collision_prim)
    
    # Set contact offset and rest offset for better collision detection
    phys_api.GetContactOffsetAttr().Set(0.002)
    phys_api.GetRestOffsetAttr().Set(0.001)
    
    print(f"Added cylinder collision to {prim_path} (r={radius:.4f}, h={height:.4f})")
    return True


def add_box_collision(stage, prim_path, size):
    """Add a box collision shape as a child of the prim."""
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        print(f"WARNING: Prim not found: {prim_path}")
        return False
    
    # Create collision child prim
    collision_path = f"{prim_path}/collision_shape"
    
    # Remove if exists
    existing = stage.GetPrimAtPath(collision_path)
    if existing.IsValid():
        stage.RemovePrim(collision_path)
    
    # Create a Cube for collision
    collision_prim = stage.DefinePrim(collision_path, "Cube")
    
    # Set box geometry using scale
    cube = UsdGeom.Cube(collision_prim)
    cube.GetSizeAttr().Set(1.0)  # Unit cube
    
    # Apply scale to make it a box
    xform = UsdGeom.Xformable(collision_prim)
    xform.ClearXformOpOrder()
    xform.AddScaleOp().Set(Gf.Vec3f(size[0], size[1], size[2]))
    
    # Make it invisible
    imageable = UsdGeom.Imageable(collision_prim)
    imageable.GetVisibilityAttr().Set("invisible")
    
    # Apply collision API
    UsdPhysics.CollisionAPI.Apply(collision_prim)
    
    print(f"Added box collision to {prim_path} (size={size})")
    return True


def remove_root_joint_and_fix_articulation(stage):
    """Remove root_joint and setup proper articulation root on base."""
    # Deactivate root_joint (the fixed joint to world)
    root_joint_path = "/LeKiwi/root_joint"
    root_joint_prim = stage.GetPrimAtPath(root_joint_path)
    if root_joint_prim.IsValid():
        root_joint_prim.SetActive(False)
        print(f"Deactivated {root_joint_path}")
    
    # Ensure base has ArticulationRootAPI
    base_path = "/LeKiwi/base_plate_layer1_v5"
    base_prim = stage.GetPrimAtPath(base_path)
    if base_prim.IsValid():
        if not base_prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            UsdPhysics.ArticulationRootAPI.Apply(base_prim)
            print(f"Applied ArticulationRootAPI to {base_path}")
        if not base_prim.HasAPI(PhysxSchema.PhysxArticulationAPI):
            PhysxSchema.PhysxArticulationAPI.Apply(base_prim)
            print(f"Applied PhysxArticulationAPI to {base_path}")


def main():
    print("=" * 60)
    print("ADDING COLLISION GEOMETRY TO LEKIWI ROBOT")
    print("=" * 60)
    
    # Open the USD stage
    stage = Usd.Stage.Open(INPUT_USD)
    if not stage:
        print(f"ERROR: Could not open {INPUT_USD}")
        return
    
    print(f"\nLoaded: {INPUT_USD}")
    
    # Fix articulation (remove fixed root joint)
    print("\n--- Fixing Articulation Root ---")
    remove_root_joint_and_fix_articulation(stage)
    
    # Add wheel collisions
    print("\n--- Adding Wheel Collisions ---")
    for wheel_path in WHEEL_BODIES:
        add_cylinder_collision(stage, wheel_path, WHEEL_RADIUS, WHEEL_WIDTH)
    
    # Add body collisions
    print("\n--- Adding Body Collisions ---") 
    for body_path, size in BODY_PARTS:
        add_box_collision(stage, body_path, size)
    
    # Verify collisions
    print("\n--- Verifying Collisions ---")
    collision_count = 0
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            collision_count += 1
            if "collision" in prim.GetPath().pathString.lower():
                print(f"  {prim.GetPath()}")
    print(f"\nTotal collision prims: {collision_count}")
    
    # Save to output file
    stage.Export(OUTPUT_USD)
    print(f"\nSaved to: {OUTPUT_USD}")
    
    simulation_app.close()
    print("\nDone!")


if __name__ == "__main__":
    main()

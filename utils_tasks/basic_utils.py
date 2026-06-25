
import os
import numpy as np
import csv
import cv2
import torch
import open3d as o3d
from typing import Optional, List, Tuple
from dataclasses import dataclass

@dataclass
class PlanningInput:
    current_goal: Optional[np.ndarray] = None
    current_image: Optional[np.ndarray] = None
    current_depth: Optional[np.ndarray] = None
    camera_pos: Optional[np.ndarray] = None
    camera_rot: Optional[np.ndarray] = None

@dataclass
class PlanningOutput:
    trajectory_points_world: Optional[np.ndarray] = None
    all_trajectories_world: Optional[List[np.ndarray]] = None
    all_values_camera: Optional[np.ndarray] = None
    sub_pointgoal_pd: Optional[np.ndarray] = None
    is_planning: bool = False
    planning_error: Optional[str] = None

def find_usd_path(dir,task='pointgoal'):
    paths = os.listdir(dir)
    usd_path = ""
    init_path = ""
    for p in paths:
        if ".usd" in p and 'noMDL' not in p:
            usd_path = os.path.join(dir,p)
        if ".npy" in p and task in p:
            init_path = os.path.join(dir,p)
    return usd_path,init_path

def write_metrics(metrics, path="exploration.csv"):
    with open(path, mode="w", newline="") as csv_file:
        fieldnames = metrics[0].keys()
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics)
        
def draw_box_with_text(image, x, y, width, height, text, 
                       box_color=(0, 255, 0), text_color=(255, 255, 255), 
                       thickness=2, font_scale=1.0):
    cv2.rectangle(image, (x, y), (x + width, y + height), box_color, thickness)
    (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    text_x = x + (width - text_width) // 2
    text_y = y + (height + text_height) // 2
    cv2.putText(image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 
                font_scale, text_color, thickness)
    return image

def cpu_pointcloud_from_array(points,colors):
    pointcloud = o3d.geometry.PointCloud()
    pointcloud.points = o3d.utility.Vector3dVector(points)
    pointcloud.colors = o3d.utility.Vector3dVector(colors)
    return pointcloud

def find_prim_recursively(current_prim, keyword='blanket'):
    blanket_prims = []
    prim_name = current_prim.GetName().lower()
    if keyword in prim_name:
        blanket_prims.append(current_prim)
        print(f"Find Prim：<{current_prim.GetPath()}>")
    for child_prim in current_prim.GetChildren():
        child_blanket_prims = find_prim_recursively(child_prim, keyword)
        blanket_prims.extend(child_blanket_prims)
    return blanket_prims

def adjust_usd_scale(prim_path="/World/Scene/terrain", scale=1.0):
    import omni
    from pxr import UsdGeom, Usd, Sdf, Gf, UsdPhysics, PhysxSchema
    stage = omni.usd.get_context().get_stage()
    scene_prim = stage.GetPrimAtPath(prim_path)
    if scene_prim.IsValid():
        print(f"Directly setting scale for prim: <{scene_prim.GetPath()}>")
        # 1. Get or create the scale attribute and set its value.
        scale_attr = scene_prim.GetAttribute("xformOp:scale")
        if not scale_attr:
            scale_attr = scene_prim.CreateAttribute("xformOp:scale", Sdf.ValueTypeNames.Double3, False)
        scale_attr.Set(Gf.Vec3d(scale, scale, scale))
        # 2. Ensure 'xformOp:scale' is in the transformation order.
        order_attr = scene_prim.GetAttribute("xformOpOrder")
        if not order_attr.HasValue():
            # If order doesn't exist, create it with a default that includes scale.
            scene_prim.CreateAttribute("xformOpOrder", Sdf.ValueTypeNames.TokenArray, False).Set(["xformOp:translate", "xformOp:orient", "xformOp:scale"])
        else:
            order = list(order_attr.Get())
            if "xformOp:scale" not in order:
                order.append("xformOp:scale")
                order_attr.Set(order)
        print(f"Successfully set scale for prim <{scene_prim.GetPath()}>")
    else:
        print("Warning: Could not find prim at /World/Scene to apply scale.")

    print(f"\nStart searching for blanket prim under <{scene_prim.GetPath()}>...")
    blanket_scope = find_prim_recursively(scene_prim)
    blanket_prim_list = []
    for blanket in blanket_scope: 
        blanket_prim_list.extend(blanket.GetAllChildren())
    for blanket_prim in blanket_prim_list:
        if blanket_prim is not None:
            print(f"Found blanket prim: <{blanket_prim.GetPath()}>")
            blanket_scale_attr = blanket_prim.GetAttribute("xformOp:scale")
            if not blanket_scale_attr:
                blanket_scale_attr = blanket_prim.CreateAttribute(
                    "xformOp:scale", Sdf.ValueTypeNames.Double3, False
                )
                blanket_scale_attr.Set(Gf.Vec3d(1.0, 1.0, 1.0))
            current_blanket_scale = blanket_scale_attr.Get()
            target_blanket_scale = Gf.Vec3d(0.1,0.1,0.0)
            blanket_scale_attr.Set(target_blanket_scale)
            blanket_order_attr = blanket_prim.GetAttribute("xformOpOrder")
            if not blanket_order_attr.HasValue():
                blanket_prim.CreateAttribute(
                    "xformOpOrder", Sdf.ValueTypeNames.TokenArray, False
                ).Set(["xformOp:translate", "xformOp:orient", "xformOp:scale"])
            else:
                blanket_order = list(blanket_order_attr.Get())
                if "xformOp:scale" not in blanket_order:
                    blanket_order.append("xformOp:scale")
                    blanket_order_attr.Set(blanket_order)
            print(f"Successfully corrected blanket prim: only Z-axis scale changed to 0")
            print(f"Blanket current scale: X={current_blanket_scale[0]}, Y={current_blanket_scale[1]}, Z=0.0")
        else:
            print(f"Did not find any blanket prim under <{scene_prim.GetPath()}>.")

    def get_all_collision_child_prims(parent_prim, target_list):
        if not parent_prim.IsValid():
            return
        direct_children = parent_prim.GetAllChildren()
        for child in direct_children:
            # child_name = child.GetName().lower()
            # if "collision" in child_name:
            if child.GetAttribute("collection:collisionmeshes").IsValid():
            # if child.GetAttribute("physics:collisionEnabled").IsValid():
                target_list.append(child)
                print(f"add collision prim to list: {child.GetPath()}")
            else:
                print(f"Skip non-collisionPrim: {child.GetPath()}")
            get_all_collision_child_prims(child, target_list)

    target_list = []
    get_all_collision_child_prims(scene_prim, target_list)
    for collision_target in target_list:
        schema_list = collision_target.GetAppliedSchemas()
        if "PhysicsCollisionAPI" in schema_list:
            collision_target.RemoveAPI(UsdPhysics.CollisionAPI)
        if "PhysicsMeshCollisionAPI" in schema_list:
            collision_target.RemoveAPI(UsdPhysics.MeshCollisionAPI)
        if "PhysxConvexHullCollisionAPI" in schema_list:
            collision_target.RemoveAPI(PhysxSchema.PhysxConvexHullCollisionAPI)
        if "PhysxConvexDecompositionCollisionAPI" in schema_list:
            collision_target.RemoveAPI(PhysxSchema.PhysxConvexDecompositionCollisionAPI)
        if "PhysxSDFMeshCollisionAPI" in schema_list:
            collision_target.RemoveAPI(PhysxSchema.PhysxSDFMeshCollisionAPI)
        if "PhysxTriangleMeshCollisionAPI" in schema_list:
            collision_target.RemoveAPI(PhysxSchema.PhysxTriangleMeshCollisionAPI)
        if "PhysxMeshMergeCollisionAPI" in schema_list:
            collision_target.RemoveAPI(PhysxSchema.PhysxMeshMergeCollisionAPI)

        UsdPhysics.CollisionAPI.Apply(collision_target)
        collisionMeshAPI = UsdPhysics.MeshCollisionAPI.Apply(collision_target)
        PhysxSchema.PhysxTriangleMeshCollisionAPI.Apply(collision_target)
        meshMergeCollision = PhysxSchema.PhysxMeshMergeCollisionAPI.Apply(collision_target)
        collision_api = PhysxSchema.PhysxCollisionAPI.Apply(collision_target)
        collision_api.CreateContactOffsetAttr().Set(0.03)
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.assets.articulation import ArticulationCfg
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
from omni.isaac.lab.sensors import ContactSensorCfg, patterns, CameraCfg, RayCasterCfg, OffsetCfg

# Define relative paths
LEKIWI_BASE_LINK = 'joints/base_plate_layer1_v5_Rigid_15'

LEKIWI_CFG = ArticulationCfg(
    prim_path = "{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        # Final USD with correct articulation + wheel collisions
        usd_path="/home/tempest/code/NavDP/assets/robots/lekiwi/lekiwi_final.usd",
        activate_contact_sensors=True,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={
            ".*": 0.0,
        },
    ),
    actuators={
        # All 3 omni-wheel drive motors - conservative gains for stability
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=["ST3215_Servo_Motor_v1_Revolute_64", "ST3215_Servo_Motor_v1_1_Revolute_62", "ST3215_Servo_Motor_v1_2_Revolute_60"],
            effort_limit=20.0,  # Match Dingo
            velocity_limit=100.0,
            stiffness=0.0,
            damping=5.0,  # Low gain for stability
        ),
        # 6 arm joints
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["STS3215_03a.*"],
            effort_limit=100.0,
            velocity_limit=50.0,
            stiffness=40.0,
            damping=10.0,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)

# For differential drive control, we use left and right wheels (front two)
# Third wheel is passive/caster in this simplified control mode
LEKIWI_WHEEL_JOINTS = ["ST3215_Servo_Motor_v1_Revolute_64", "ST3215_Servo_Motor_v1_1_Revolute_62"]
LEKIWI_WHEEL_RADIUS = 0.0325
LEKIWI_WHEEL_BASE = 0.18
LEKIWI_THRESHOLD = 15.0

# Camera placement 
LEKIWI_CAMERA_TRANS = [0.0, 0.0, 0.2] 
LEKIWI_CAMERA_ROTS = [0.5, -0.5, 0.5, -0.5]
LEKIWI_IMAGEGOAL_TRANS = [5.0, 0.0, 0.2]
LEKIWI_IMAGEGOAL_ROTS = [0.5, -0.5, 0.5, -0.5]

# Note: Since Articulation is now rooted at base_link, paths relative to it for sensors
# might need adjustment OR absolute paths.
# ContactSensorCfg uses prim_path.
# If I use absolute path with {ENV_REGEX_NS}, it's fine.
# Note that LEKIWI_BASE_LINK string is just the name "joints/..."
# We should probably point sensors to the same rigid body.

LEKIWI_ContactCfg = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/%s"%LEKIWI_BASE_LINK, 
                                    history_length=10, 
                                    track_air_time=True,
                                    update_period=0.02)

LEKIWI_CameraCfg = CameraCfg(
    prim_path="{ENV_REGEX_NS}/Robot/%s/front_cam"%LEKIWI_BASE_LINK,
    update_period=0.05,
    height=360,
    width=640,
    data_types=["rgb", "distance_to_image_plane"],
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=1.4, focus_distance=0.205, horizontal_aperture=1.88, clipping_range=(0.01, 100.0)
    ),
    offset=CameraCfg.OffsetCfg(pos=LEKIWI_CAMERA_TRANS, rot=LEKIWI_CAMERA_ROTS, convention="ros"),
)

LEKIWI_ImageGoal_CameraCfg = CameraCfg(
    prim_path="{ENV_REGEX_NS}/goal_cam",
    update_period=0.05,
    height=360,
    width=640,
    data_types=["rgb"],
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=1.4, focus_distance=0.205, horizontal_aperture=1.88, clipping_range=(0.01, 100.0)
    ),
    offset=CameraCfg.OffsetCfg(pos=LEKIWI_IMAGEGOAL_TRANS, rot=LEKIWI_IMAGEGOAL_ROTS, convention="ros"),
)

LEKIWI_MetricCameraCfg = CameraCfg(
    prim_path="{ENV_REGEX_NS}/Robot/%s/narritor_cam"%LEKIWI_BASE_LINK,
    update_period=0.05,
    height=90,
    width=160,
    data_types=["rgb", "distance_to_image_plane"],
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=1.4, focus_distance=0.205, horizontal_aperture=1.88, clipping_range=(0.01, 100.0)
    ),
    offset=CameraCfg.OffsetCfg(pos=[0.0,0.0,1.0], rot=LEKIWI_CAMERA_ROTS, convention="ros"),
)

<p align="center">
<h1 align="center"><strong>LoGoPlanner: Localization Grounded Navigation Policy with Metric-aware Visual Geometry</strong></h1>
  <p align="center">
    <!--   	<strong>CVPR 2024</strong><br> -->
	<a href='https://steinate.github.io/' target='_blank'>Jiaqi Peng</a>&emsp;
    <a href='https://wzcai99.github.io/' target='_blank'>Wenzhe Cai</a>&emsp;
    <a href='https://yuqiang-yang.github.io/' target='_blank'>Yuqiang Yang</a>&emsp;
    <a href='https://tai-wang.github.io/' target='_blank'>Tai Wang</a>&emsp;
    <a href='https://oa.ee.tsinghua.edu.cn/~shenyuan/' target='_blank'>Yuan Shen</a>&emsp;
	<a href='https://oceanpang.github.io/' target='_blank'>Jiangmiao Pang</a>&emsp;
    <br>
    Tsinghua University&emsp;
    Shanghai AI Laboratory&emsp;
    <br>
  </p>
</p>
<div id="top" align="center">

[![Project](https://img.shields.io/badge/Project-%239c403d?style=flat&logoColor=%23FA7F6F
)](https://steinate.github.io/logoplanner.github.io/)
[![arXiv](https://img.shields.io/badge/Arxiv-%233b6291?style=flat&logoColor=%23FA7F6F
)](https://arxiv.org/abs/)
[![Video](https://img.shields.io/badge/Video-%23c97937?style=flat&logoColor=%23FA7F6F
)](https://www.youtube.com/)
[![Benchmark](https://img.shields.io/badge/Benchmark-8A2BE2?style=flat
)](https://github.com/InternRobotics/NavDP)
[![Dataset](https://img.shields.io/badge/Dataset-548B54?style=flat
)](https://huggingface.co/datasets/InternRobotics/InternData-N1/)
[![GitHub star chart](https://img.shields.io/github/stars/InternRobotics/NavDP?style=square)](https://github.com/InternRobotics/NavDP)
[![GitHub Issues](https://img.shields.io/github/issues/InternRobotics/NavDP)](https://github.com/InternRobotics/NavDP)
</div>

# 🏡 Introduction
Most prior end-to-end approaches still rely on separate localization modules that depend on accurate sensor extrinsic calibration for self-state estimation, thereby limiting generalization across embodiments and environments. We introduce LoGoPlanner, a localization-grounded, end-to-end navigation framework that addresses these limitations by: (1) finetuning a long-horizon visual-geometry backbone to ground predictions with absolute metric scale, thereby providing implicit state estimation for accurate localization; (2) reconstructing surrounding scene geometry from historical observations to supply dense, fine-grained environmental awareness for reliable obstacle avoidance; and (3) conditioning the policy on implicit geometry bootstrapped by the aforementioned auxiliary tasks, thereby reducing error propagation. 
<div style="text-align: center;">
    <img src="https://steinate.github.io/logoplanner.github.io/static/images/motivation.svg" alt="Teaser" width=100% >
</div>

# 💻 Hands on Simulation
### 🛠️ Installation
We use the same environment with NavDP. Please follow the [instructions](https://github.com/InternRobotics/NavDP/blob/master/README.md#%EF%B8%8F-installation) of NavDP to config the environment.
```bash
conda activavte navdp
```
Then add install the required packages of viusal geometry model [Pi3](https://github.com/yyfz/Pi3)
```bash
cd baselines/logoplanner
pip install plyfile huggingface_hub safetensors
```
### 🤔 Run LoGoPlanner Model
In the path of `baselines/logoplanner`, run the following command
```bash
python logoplanner_server.py --port ${YOUR_PORT} --checkpoint ${SAVE_PTH_PATH}
```

### 📊 Running Evaluation
Open a new terminal and run the following command in the home path `{NavDP_HOME}`
```bash
conda activate isaaclab
python eval_startgoal_wheeled.py --port {PORT} --scene_dir {ASSET_SCENE} --scene_index {INDEX} --scene_scale {SCALE}
```


<div align=center>
    <img src="./assets/simulation.gif" alt="Teaser" width=800>
</div>


### 😉 Example:

```bash
# starting the server
conda activavte navdp && python logoplanner_server.py --port 19999 --checkpoint logoplanner_policy.ckpt
# running on scenes_home
conda activate isaaclab && python eval_startgoal_wheeled.py --port 19999 --scene_dir scenes_home --scene_index 0 --scene_scale 0.01
# or running on cluttered_hard
conda activate isaaclab && python eval_startgoal_wheeled.py --port 19999 --scene_dir cluttered_hard --scene_index 0 --scene_scale 1.0
```


# 🤖 Hands on Real-robot
<div align=center>
    <img src=./assets/realworld.gif width=800>
</div>


The [Lekiwi](https://github.com/SIGRobotics-UIUC/LeKiwi) is a fully open-source robotic car project launched by [SIGRobotics-UIUC](https://github.com/SIGRobotics-UIUC). It includes the detailed 3D printing files and operation guides, designed to be compatible with the [LeRobot](https://github.com/huggingface/lerobot/tree/main) imitation learning framework. It supports the SO101 robotic arm to enable a complete imitation learning pipeline,
<div align=center>
    <img width=400 src=https://files.seeedstudio.com/wiki/robotics/projects/lerobot/lekiwi/lekiwi_cad_v1.png>
</div>

## 🛠️ hardware
#### Compute
- Raspberry Pi 5
- Streaming to a Laptop
#### Drive
- 3-wheel Kiwi (holonomic) drive with omni wheels
#### Robot Arm (Optional)
- [SO-ARM101](https://github.com/TheRobotStudio/SO-ARM100)
#### Sensors
- rgbd camera (e.g. Intel Realsense D455)

### 1️⃣ 3D Printing
## Parts
SIGRobotics provide ready-to-print STL files for the 3D-printed parts below. These can be printed with generic PLA filament on consumer-grade FDM printers. Refer to the [3D Printing](https://github.com/SIGRobotics-UIUC/LeKiwi/blob/main/3DPrinting.md) section for more details.

| Item | Quantity | Notes | 
|:---|:---:|:---:|
| [Base plate Top](https://github.com/SIGRobotics-UIUC/LeKiwi/tree/main/3DPrintMeshes/base_plate_layer2.stl) | 1 | |
| [Base plate Bottom](https://github.com/SIGRobotics-UIUC/LeKiwi/tree/main/3DPrintMeshes/base_plate_layer1.stl) | 1 | |
| [Drive motor mount](https://github.com/SIGRobotics-UIUC/LeKiwi/tree/main/3DPrintMeshes/drive_motor_mount_v2.stl) | 3 | |
| [Servo wheel hub](https://github.com/SIGRobotics-UIUC/LeKiwi/tree/main/3DPrintMeshes/servo_wheel_hub.stl) | 3 | Use Supports<sup>[1](#footnote1)</sup> |
| [Servo controller mount](https://github.com/SIGRobotics-UIUC/LeKiwi/tree/main/3DPrintMeshes/servo_controller_mount.stl) | 1 | |
| [12v Battery mount](https://github.com/SIGRobotics-UIUC/LeKiwi/tree/main/3DPrintMeshes/battery_mount.stl) **or** [12v EU Battery mount](https://github.com/SIGRobotics-UIUC/LeKiwi/tree/main/3DPrintMeshes/battery_mount_eu.stl) **or** [5v Battery mount](https://github.com/SIGRobotics-UIUC/LeKiwi/tree/main/3DPrintMeshes/5v_specific/5v_power_bank_holder.stl)| 1 | |
| [RasPi case Top](https://github.com/SIGRobotics-UIUC/LeKiwi/tree/main/3DPrintMeshes/pi_case_top.stl) | 1 | <sup>[2](#footnote2)</sup> |
| [RasPi case Bottom](https://github.com/SIGRobotics-UIUC/LeKiwi/tree/main/3DPrintMeshes/pi_case_bottom.stl) | 1 | <sup>[2](#footnote2)</sup> |
| Arducam [base mount](https://github.com/SIGRobotics-UIUC/LeKiwi/tree/main/3DPrintMeshes/base_camera_mount.stl) and [Wrist mount](/3DPrintMeshes/wrist_camera_mount.stl)| 1 | **Compatible with [this camera](https://www.amazon.com/Arducam-Camera-Computer-Without-Microphone/dp/B0972KK7BC)** |
| Webcam [base mount](https://github.com/SIGRobotics-UIUC/LeKiwi/tree/main/3DPrintMeshes/webcam_mount/webcam_mount.stl), [gripper insert](https://github.com/SIGRobotics-UIUC/LeKiwi/tree/main/3DPrintMeshes/webcam_mount/so100_gripper_cam_mount_insert.stl), and [wrist mount](https://github.com/SIGRobotics-UIUC/LeKiwi/tree/main/3DPrintMeshes/webcam_mount/webcam_mount_wrist.stl) | 1 | **Compatible with [this camera](https://www.amazon.fr/Vinmooog-equipement-Microphone-Enregistrement-conférences/dp/B0BG1YJWFN/)** |
| [Modified Follower Arm Base](https://github.com/SIGRobotics-UIUC/LeKiwi/tree/main/3DPrintMeshes/modified_base_arm.stl) | 1 | Use Tree Supports, **Optional, but recommended if you have not built SO-100 already** |
| [Follower arm](https://github.com/TheRobotStudio/SO-ARM100) | 1 | |
| [Leader arm](https://github.com/TheRobotStudio/SO-ARM100) | 1 | |

### 2️⃣ Assembly
Refer to the [Assembly](https://github.com/SIGRobotics-UIUC/LeKiwi/blob/main/Assembly.md) section for more details. 

We also highly recommend the following detailed tutorial [seeedstudio](https://wiki.seeedstudio.com/lerobot_lekiwi/) and [accompanying videos](https://www.bilibili.com/video/BV1TLUhBWE2D?p=1): 

[![How to Assemble & Set Up LeKiwi (Mobile robot Tutorial)
](https://img.youtube.com/vi/cKWAjEV4aSg/0.jpg)](https://www.youtube.com/watch?v=cKWAjEV4aSg)

### 3️⃣ Install
#### Install LeRobot on Raspberry Pi
1. Install Miniconda
```bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
```

2. Restart shell
Copy paste in your shell: `source ~/.bashrc` or for Mac: `source ~/.bash_profile` or `source ~/.zshrc` if you're using zshell

3. Create and activate a fresh conda environment for lerobot
```bash
conda create -y -n lerobot python=3.10
```

Then activate your conda environment (do this each time you open a shell to use lerobot!):
```bash
conda activate lerobot
```
4. Clone LeRobot
```bash
git clone https://github.com/huggingface/lerobot.git ~/lerobot
```

5. Install ffmpeg in your environment
When using miniconda, install ffmpeg in your environment:
```bash
conda install ffmpeg -c conda-forge
```
6. Install LeRobot with dependencies for the feetech motors
```bash
cd ~/lerobot && pip install -e ".[lekiwi]"
```

#### Install LeRobot on laptop(PC)
1. Install Miniconda
2. Restart shell
Copy paste in your shell: `source ~/.bashrc` or for Mac: `source ~/.bash_profile` or `source ~/.zshrc` if you're using zshell

3. Create and activate a fresh conda environment for lerobot
```bash
conda create -y -n lerobot python=3.10
```
Then activate your conda environment (do this each time you open a shell to use lerobot!):
```bash
conda activate lerobot
```
4. Clone LeRobot
```bash
git clone https://github.com/huggingface/lerobot.git ~/lerobot
```
5. Install ffmpeg in your environment
When using miniconda, install ffmpeg in your environment:
```bash
conda install ffmpeg -c conda-forge
```
6. Install LeRobot with dependencies for the feetech motors
```bash
cd ~/lerobot && pip install -e ".[lekiwi]"
```

#### Install LeRobot on Raspberry Pi
Refer to [this repository](https://gitlab.uwaterloo.ca/awerner/librealsense/-/blob/v2.20.0/doc/installation_raspbian.md)
1. Check versions
```bash
uname -a
```
2. Add swap
```bash
sudo vim /etc/dphys-swapfile
CONF_SWAPSIZE=2048

sudo /etc/init.d/dphys-swapfile restart swapon -s
```

3. Install packages
```bash
sudo apt-get install -y libdrm-amdgpu1 libdrm-amdgpu1-dbg libdrm-dev libdrm-exynos1 libdrm-exynos1-dbg libdrm-freedreno1 libdrm-freedreno1-dbg libdrm-nouveau2 libdrm-nouveau2-dbg libdrm-omap1 libdrm-omap1-dbg libdrm-radeon1 libdrm-radeon1-dbg libdrm-tegra0 libdrm-tegra0-dbg libdrm2 libdrm2-dbg

sudo apt-get install -y libglu1-mesa libglu1-mesa-dev glusterfs-common libglu1-mesa libglu1-mesa-dev libglui-dev libglui2c2

sudo apt-get install -y libglu1-mesa libglu1-mesa-dev mesa-utils mesa-utils-extra xorg-dev libgtk-3-dev libusb-1.0-0-dev
```

4. Update udev rule
```bash
cd ~
git clone https://github.com/IntelRealSense/librealsense.git
cd librealsense
sudo cp config/99-realsense-libusb.rules /etc/udev/rules.d/ 
sudo udevadm control --reload-rules && udevadm trigger 
```

5. install RealSense SDK/librealsense
```bash
cd ~/librealsense
mkdir  build  && cd build
cmake .. -DBUILD_EXAMPLES=true -DCMAKE_BUILD_TYPE=Release -DFORCE_LIBUVC=true
make -j1
sudo make install
```

6. install pyrealsense2
```bash
cd ~/librealsense/build

# for python3
cmake .. -DBUILD_PYTHON_BINDINGS=bool:true -DPYTHON_EXECUTABLE=$(which python3)

make -j1
sudo make install

# add python path
vim ~/.zshrc
export PYTHONPATH=$PYTHONPATH:/usr/local/lib

source ~/.zshrc
```
7. Try RealSense D455
```bash
realsense-viewer
```

### 4️⃣ Motors configuration
To find the port for each bus servo adapter, run this script:
```bash
lerobot-find-port
```
Example output:
```bash
Finding all available ports for the MotorBus.
['/dev/ttyACM0']
Remove the USB cable from your MotorsBus and press Enter when done.

[...Disconnect corresponding leader or follower arm and press Enter...]

The port of this MotorsBus is /dev/ttyACM0
Reconnect the USB cable.
```

🚨 Remember to remove the usb, then Press Enter, otherwise the interface will not be detected.

On Linux, you might need to give access to the USB ports by running:
```bash
sudo chmod 666 /dev/ttyACM0
sudo chmod 666 /dev/ttyACM1
```

You can run this command to setup motors for LeKiwi. It will first setup the motors for arm (id 6..1) and then setup motors for wheels (9,8,7).
```bash
lerobot-setup-motors \
    --robot.type=lekiwi \
    --robot.port=/dev/ttyACM0 # <- paste here the port found at previous step
```
<div align=center>
    <img width=500 src=https://files.seeedstudio.com/wiki/robotics/projects/lerobot/lekiwi/motor_ids.png>
</div>

### 5️⃣ Teleoperate
To teleoperate SSH into your Raspberry Pi, and run conda activate lerobot and this script:
```bash
python -m lerobot.robots.lekiwi.lekiwi_host --robot.id=my_awesome_kiwi
```
Then on your laptop, also run conda activate lerobot and run the API example, make sure you set the correct remote_ip and port in examples/lekiwi/teleoperate.py.

<div align=center>
    <img width=800 src=https://files.seeedstudio.com/wiki/robotics/projects/lerobot/lekiwi/teleoperate.png>
</div>

```bash
python examples/lekiwi/teleoperate.py
```

You should see on your laptop something like this: `[INFO] Connected to remote robot at tcp://172.17.133.91:5555 and video stream at tcp://172.17.133.91:5556`. Now you can move the leader arm and use the keyboard (w,a,s,d) to drive forward, left, backwards, right. And use (z,x) to turn left or turn right. You can use (r,f) to increase and decrease the speed of the mobile robot. 

### 6️⃣ Deployment
Mount the RGBD camera to LeKiwi and adjust the SO101 arm not to obstruct the camera:
<div align=center>
    <img width=500 src=./assets/camera_mount.jpg>
</div>

🚨 Before you start our algorithm, you can let LeKiwi to follow some simple trajectories like sin or 's' curve to make sure the MPC tracking work.

### 7️⃣ Deployment

To deploy logoplanner on LeKiwi, you can run this script to start server on the your laptop or PC:
```bash
python logoplanner_realworld_server.py --port 19999 --checkpoint ${CKPT_PATH}
```

Make sure that the correct ip for the Pi is set in the configuration file. To check the server address, run on your laptop or PC:
```bash
hostname -I
```

Then on RaspberyPi, cp the file `lekiwi_logoplanner_host.py` to you work directory and run the following commands to start client:
```bash
conda activate lerobot
python lekiwi_logoplanner_host.py --server-url http://192.168.1.100:8888 --goal-x 10 --goal-y -2
```

You will see the robot move to the target (10, -2). Without any external odometry module, the robot still will know where it is and gradually move to the target then stop.
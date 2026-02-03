# The Ghost, The Anchor, and The Exploding Robot: A LeKiwi Integration Log

Integrating a custom robot into a physics simulation is rarely as simple as "plug and play." Our recent attempt to swap the default **Dingo** robot for a custom 3-wheeled omni-bot named **LeKiwi** in Isaac Lab turned into a masterclass in debugging physics, USD structures, and control theory.

Here is the story of how we took LeKiwi from a static mesh to a moving, autonomous robot.

## Chapter 1: The Immovable Object
**The Issue:**
Our first attempt to spawn `lekiwi.usd` resulted in a statue. The robot spawned but refused to fall, move, or react to gravity. It was essentially glued to the origin `(0, 0, 0)`.

**The Diagnosis:**
The USD file had a fixed `root_joint` connecting the robot to the world frame. This is common in URDF exports intended for static analysis but fatal for mobile robotics simulation.

**The Fix:**
We wrote a script (`fix_lekiwi_usd.py`) to programmatically deactivate the static `root_joint` and apply the `ArticulationRootAPI` to the robot's actual chassis, converting it into a true floating-base object.

## Chapter 2: The Ghost in the Machine
**The Issue:**
With the root fixed, the robot now fell... straight through the floor. Or, if it stayed up, the wheels stuck into the ground. We commanded the motors to spin, and the logs showed them spinning at 20 rad/s, yet the robot didn't move an inch.

**The Diagnosis:**
We had a "Ghost Robot." While the visual meshes looked perfect, a deep dive into the USD structure revealed that **collision meshes were missing**. The physics engine saw the robot as empty space. The wheels were spinning phantoms with no friction to push against the ground.

**The Fix:**
We created a tool (`add_wheel_collisions.py`) to procedurally inject collision geometry. We added:
*   **Cylinders** for the wheels.
*   **Boxes** for the body chassis.
*   Physics schemas (`PhysxCollisionAPI`) to handle contact offsets.

## Chapter 3: The Exploding Robot (NaNs)
**The Issue:**
As soon as we added collisions, the simulation exploded. The console was flooded with `NaN` (Not a Number) errors. The robot's position would vanish into the mathematical void immediately upon spawning.

**The Diagnosis:**
Two culprits were working in tandem:
1.  **Self-Collisions:** The new collision shapes we added were overlapping with the body. The physics engine tried to resolve two solid objects occupying the same space, creating infinite force vectors ($$ F = \infty $$).
2.  **Actuator Gains:** The default PID gains were tuned for a heavier robot. LeKiwi was spasming violently, causing the math in our observation code (specifically matrix inversion `linalg.inv`) to fail.

**The Fix:**
*   **Robust Math:** We replaced `torch.inverse` with `torch.linalg.pinv`.
*   **Physics Triage:** We updated our script to explicitely **disable self-collisions** on the robot's articulation tree and slightly reduced the wheel collision radius by 5% to prevent ground interpenetration.
*   **Safety Wheels:** We wrapped our MPC solver and visualization code in `try/except` blocks to catch `NaN` values before they crashed the production loop.

## Chapter 4: The Anchor
**The Issue:**
The robot finally existed, had mass, and didn't explode. But it still refused to move correctly. The wheels spun, but the robot acted like it was dragging a heavy weight. It spun in place or drifted aimlessly.

**The Diagnosis:**
LeKiwi is a 3-wheel omni-robot, but we were driving it like a 2-wheel differential drive. We had given all three wheels **Cylinder** collision shapes. Since the third back wheel is unpowered (in our test config), and a cylinder resists sideways motion, it acted like a giant rubber anchor. The front wheels tried to turn, but the back wheel refused to slide.

**The Fix:**
We modified the collision script to treat the third wheel as a caster:
*   **Front Wheels:** Cylinders (Traction).
*   **Back Wheel:** **Sphere** (Zero friction to allow sliding).

## Chapter 5: Current Status (It Still Won't Move)
**The Issue:**
Despite fixing the anchor, the ghost, and the explosions... **LeKiwi is not moving.**
The logs show the root position stuck at `[-6.64, 14.05, 0.033]`. The z-height (`0.033`) corresponds exactly to the wheel radius, meaning it is sitting on the floor.
*   The MPC is generating valid velocity commands (`Action: [14.0, 16.0]`).
*   The wheels are physically spinning (we see actual joint velocities matching desired).
*   But the **Base Velocity is ZERO**.

**The Diagnosis:**
This suggests a fundamental disconnect in the **Tire Friction Model**. Even though the collision cylinders are touching the ground, they are generating **zero traction force**. This is likely because we are using simple geometric primitives (Cylinders) without explicitly binding a **Physics Material** with friction coefficients. In Isaac Sim, a default collision shape sometimes defaults to frictionless "ice" unless specified otherwise. We are effectively doing a burnout on zero-friction ice.

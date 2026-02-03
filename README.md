# NavDP Research Sandbox: Omni-directional Robotics & Expansion

This repository is an academic research fork of the [official NavDP implementation](https://github.com/InternRobotics/NavDP). The purpose of this fork is to deepen the understanding of Diffusion Policies in robotics and extend the framework's capabilities through custom hardware integration and enhanced visualization tools.
<img width="1244" height="449" alt="image" src="https://github.com/user-attachments/assets/584fb1eb-ba4a-4226-81b0-0e6a1c53354e" />

## 🚀 Key Research Contributions

In this sandbox, I have implemented several critical enhancements to the core NavDP framework:

### 1. LeKiwi Robot Integration (3-Wheeled Omni-directional)
Successfully integrated the **LeKiwi** robot model, transitioning from the default differential-drive Dingo to a more complex **omni-directional 3-wheeled base**. 

**Technical Resources:**
- **LinkedIn Article:** [The Ghost, The Anchor, and The Exploding Robot: A LeKiwi Integration Log](https://www.linkedin.com/posts/priyanshu123sah_physicalai-robotics-deeplearning-activity-7421533400718770176-hZhs?utm_source=share&utm_medium=member_desktop&rcm=ACoAAEDrLh4BEAfLe_9iKxGIAEOI88fdcEu5t4s)
- **Repo Integration Log:** [LEKIWI_INTEGRATION_LOG.md](development_logs/LEKIWI_INTEGRATION_LOG.md) — *A deep dive into why off-the-shelf USD models are often "simulation-broken" and how to fix them for navigation.*

The integration required significant engineering efforts in:
- **USD Preparation:** Programmatically fixing articulation roots and programmatically injecting collision geometries.
- **Physics Tuning:** Solving simulation "explosions" (NaNs) by optimizing actuator gains and contact offsets.
- **Friction Modeling:** Implementing a hybrid collision model (Cylinders for traction wheels, frictionless Sphere for the caster) to ensure realistic motion and turn-in-place capabilities.

### 2. Multi-Perspective Visualization
Expanded the visual evaluation capabilities by adding two primary research perspectives:
- **Bird's Eye View (BEV):** For clear path-planning analysis and spatial reasoning.
- **Third-Person View:** To better observe the robot's physical interaction with the environment.

### 3. Professional Research Tooling
Developed a suite of diagnostic and integration tools located in the `research_tools/` directory to automate the robot on-boarding process and physics verification.

---

## 📂 Professional Repository Structure

To maintain a research-ready environment, this fork follows a structured hierarchy:

- **`research_tools/`**: Automation scripts for USD fixes, collision generation, and physics validation.
- **`evaluation_outputs/`**: 
    - `benchmark_runs/`: Raw data from evaluation episodes.
    - `videos/`: Recorded simulation runs showcasing navigation behaviors.
- **`development_logs/`**: Detailed troubleshooting logs (e.g., `lekiwi_integration_log.txt`) documenting the technical journey.
- **`README_OFFICIAL.md`**: The original documentation with full technical details of the NavDP paper.

---

## 🛠️ Getting Started

Follow the installation steps in [README_OFFICIAL.md](README_OFFICIAL.md) to set up the environment. 

To run evaluations with the new **LeKiwi** configuration:
```bash
# Example PointGoal evaluation with LeKiwi
python eval_pointgoal_wheeled.py --port 8888 --robot lekiwi --scene_dir [PATH_TO_SCENE]
```

---

## 📜 Credits & Attribution

This work is built upon the **NavDP** (Navigation Diffusion Policy) paper:

> **NavDP: Learning Sim-to-Real Navigation Diffusion Policy with Privileged Information Guidance**  
> *Wenzhe Cai, Jiaqi Peng, Yuqiang Yang, Yujian Zhang, Meng Wei, Hanqing Wang, Yilun Chen, Tai Wang and Jiangmiao Pang*  
> [Official Repository](https://github.com/InternRobotics/NavDP) | [arXiv Paper](https://arxiv.org/abs/2505.08712)

I express my gratitude to the original authors for open-sourcing their impressive work, which provided the foundation for this research exploration.

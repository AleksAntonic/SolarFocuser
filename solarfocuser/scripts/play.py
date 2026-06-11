"""
Play / visualize a trained SolarFocuser policy.

Usage (from the repo root):
    # live pygame window, latest checkpoint of experiment 'myexp'
    python -m solarfocuser.scripts.play --experiment-name myexp

    # explicit checkpoint, record a MP4 + trajectory figure instead of a window
    python -m solarfocuser.scripts.play --checkpoint logs/myexp/<run>/model_500.pt --record

Outputs (with --record, written next to the checkpoint or ./play_out):
    play.mp4            -- rendered rollout video
    trajectory.png      -- asteroid-frame path, distances, speeds, attitude, actions
"""

import argparse
import glob
import os

import numpy as np
import torch
import gymnasium as gym

from solarfocuser.cfg.focuser_cfg import EnvConfig, TrainConfig
# Crucial Fix: Import BOTH the physics world and the vector wrapper
from solarfocuser.env.focuser_task import _SailWorld, SolarFocuser

try:
    from solarfocuser import ROOT_DIR
except Exception:
    ROOT_DIR = os.getcwd()


def find_checkpoint(experiment_name):
    """Latest model_*.pt of the latest run under logs/<experiment_name>."""
    if experiment_name is None:
        return None
    log_root = os.path.join(ROOT_DIR, "logs", experiment_name)
    runs = sorted(glob.glob(os.path.join(log_root, "*")))
    for run_dir in reversed(runs):
        models = glob.glob(os.path.join(run_dir, "model_*.pt"))
        if models:
            models.sort(key=lambda p: int(os.path.splitext(p)[0].split("_")[-1]))
            return models[-1]
    return None


def load_policy(env, checkpoint, device="cpu"):
    """Build a PPO runner around the vectorized wrapper to safely load checkpoint parameters."""
    from solarfocuser.runners.algs.ppo import PPO

    # PPO requires the vectorized version to map weights cleanly
    vector_wrapper = SolarFocuser(args=env.cfg, render_mode=None)
    runner = PPO(env=vector_wrapper, train_cfg=TrainConfig(), log_dir=None, device=device)
    runner.load(checkpoint)
    return runner.get_inference_policy(device=device)


def save_mp4(frames, path, fps=60, scale=0.5):
    """Saves captured layout frames into a compressed MP4 video file using OpenCV."""
    import cv2

    if not frames:
        print("WARN: No frames captured to record.")
        return

    height, width, _ = frames[0].shape
    if scale != 1.0:
        width = int(width * scale)
        height = int(height * scale)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(path, fourcc, fps, (width, height))

    for frame in frames:
        if scale != 1.0:
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        # Convert RGB canvas map directly to OpenCV BGR streaming format
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(bgr_frame)

    writer.release()


def plot_trajectory(states, actions, cfg, path):
    """Asteroid-frame trajectory mapping + rollout analytics."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s = np.asarray(states)            # (T, 12)
    a = np.asarray(actions)           # (T, 3)
    
    # Extract state variables using raw indices matching State Enum configuration
    dx = s[:, 0] - s[:, 6]            
    dy = s[:, 1] - s[:, 7]
    dvx = s[:, 2] - s[:, 8]
    dvy = s[:, 3] - s[:, 9]
    theta = s[:, 4]
    t = np.arange(len(s)) * cfg.sim_dt

    dist = np.hypot(dx, dy)
    surf = dist - cfg.asteroid_radius * 1.20
    speed = np.hypot(dvx, dvy)

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # --- asteroid-frame path ---
    ax = axes[0, 0]
    ang = np.linspace(0, 2 * np.pi, 100)
    ax.fill(cfg.asteroid_radius * np.cos(ang), cfg.asteroid_radius * np.sin(ang),
            color="#6e5f50", alpha=0.9, label="asteroid")
    rcap = cfg.asteroid_radius * 1.20 + cfg.capture_distance
    ax.plot(rcap * np.cos(ang), rcap * np.sin(ang), "g--", lw=1, label="capture zone")
    sc = ax.scatter(dx, dy, c=t, cmap="viridis", s=4)
    fig.colorbar(sc, ax=ax, label="time [s]")
    
    k = max(1, len(s) // 20)
    ax.quiver(dx[::k], dy[::k], np.cos(theta[::k]), np.sin(theta[::k]),
              color="crimson", scale=25, width=3e-3, label="sail normal")
    
    sun = np.asarray(cfg.sun_direction, float)
    sun = sun / np.linalg.norm(sun)
    ax.annotate("sun", xy=(0.85 * rcap * 2 * sun[0], 0.85 * rcap * 2 * sun[1]), color="orange", fontsize=12)
    ax.quiver(0, 0, sun[0], sun[1], color="orange", scale=8, width=4e-3, alpha=0.5)
    ax.set_xlabel("x rel. asteroid [m]"); ax.set_ylabel("y rel. asteroid [m]")
    ax.set_title("Trajectory (asteroid frame)")
    ax.axis("equal"); ax.legend(loc="upper right", fontsize=8)

    # --- distance ---
    ax = axes[0, 1]
    ax.plot(t, surf, lw=1.2)
    ax.axhline(0, color="k", lw=0.8)
    ax.axhspan(0, cfg.capture_distance, color="g", alpha=0.15, label="capture band")
    ax.set_xlabel("time [s]"); ax.set_ylabel("surface distance [m]")
    ax.set_title("Distance to asteroid surface"); ax.legend(fontsize=8)

    # --- speed + attitude ---
    ax = axes[1, 0]
    ax.plot(t, speed, lw=1.2, label="|rel. velocity| [m/s]")
    ax.axhline(cfg.capture_speed, color="g", ls="--", lw=1, label="capture speed")
    sun_angle = float(np.arctan2(sun[1], sun[0]))
    ax.plot(t, np.cos(theta - sun_angle), lw=1.0, label="cos(sun alignment)")
    ax.set_xlabel("time [s]"); ax.set_title("Relative speed & sun alignment")
    ax.legend(fontsize=8)

    # --- actions ---
    ax = axes[1, 1]
    for i, lbl in enumerate(["rcs_x", "rcs_y", "wheel"]):
        ax.plot(t, a[:, i], lw=0.8, label=lbl)
    ax.set_ylim(-1.1, 1.1)
    ax.set_xlabel("time [s]"); ax.set_title("Actions"); ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="focuser")
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    # Create the base configuration template
    env_cfg = EnvConfig()
    object.__setattr__(env_cfg, "num_envs", 1)
    object.__setattr__(env_cfg, "debug", True)
    
    # 1. Set render mode based on recording intent
    render_mode = "rgb_array" if args.record else "human"
    env = _SailWorld(args=env_cfg, render_mode=render_mode)

    # --- policy loading ---
    ckpt = args.checkpoint or find_checkpoint(args.experiment_name)
    if ckpt:
        print(f"Loading checkpoint: {ckpt}")
        policy = load_policy(env, ckpt, device=args.device)
        out_dir = os.path.dirname(ckpt)
    else:
        print("WARN: No checkpoint found -- flying with zero actions")
        policy = lambda obs: torch.zeros(1, env.action_space.shape[0])
        out_dir = os.path.join(os.getcwd(), "play_out")
    os.makedirs(out_dir, exist_ok=True)

    # ------------------------------------------------------------------ #
    # 2. GYMNASIUM VIDEO RECORDING WRAPPER
    # This automatically hooks into env.step() and handles file generation.
    # ------------------------------------------------------------------ #
    if args.record:
        # It writes directly into your run checkpoint log directory
        env = gym.wrappers.RecordVideo(
            env, 
            video_folder=out_dir, 
            episode_trigger=lambda x: True, # Record every episode
            name_prefix="play"
        )

    # --- rollout sequence ---
    states, acts = [], []
    episodes_done, step = 0, 0
    
    # Gymnasium wrapper reset returns (obs, info) 
    obs, info = env.reset(seed=getattr(env_cfg, "seed", 42))

    POS_SCALE = 1000.0
    sd = np.asarray(env_cfg.sun_direction, dtype=float)
    sun_angle = float(np.arctan2(sd[1], sd[0]))

    with torch.inference_mode():
        while episodes_done < args.episodes and step < args.max_steps:
            # Reconstruct normalized PPO tensor shape inputs
            rel_pos_x = obs[6] - obs[0]
            rel_pos_y = obs[7] - obs[1]
            rel_vel_x = obs[8] - obs[2]
            rel_vel_y = obs[9] - obs[3]
            sail_theta = obs[4]
            sail_theta_dot = obs[5]
            surface_dist = np.hypot(rel_pos_x, rel_pos_y) - (env_cfg.asteroid_radius * 1.20)

            ppo_obs_vector = torch.tensor([[
                rel_pos_x / POS_SCALE,
                rel_pos_y / POS_SCALE,
                rel_vel_x,
                rel_vel_y,
                np.sin(sail_theta),
                np.cos(sail_theta),
                sail_theta_dot,
                surface_dist / POS_SCALE,
                np.cos(sail_theta - sun_angle)
            ]], dtype=torch.float32, device=args.device)

            actions_tensor = policy(ppo_obs_vector)
            action_np = actions_tensor.cpu().numpy().flatten()
            
            # Step the wrappered environment
            next_obs, reward, done, truncated, info = env.step(action_np)

            # Access the unwrapped environment variables for trajectory plotting
            states.append(env.unwrapped.state.copy())
            acts.append(action_np.copy())

            if done or truncated:
                episodes_done += 1
                tag = "TIMEOUT" if truncated else ("CAPTURE" if reward >= 100 else "CRASH")
                print(f"Episode {episodes_done} ended at step {step}: {tag} (Terminal reward {reward:.1f})")
                obs, info = env.reset()
            else:
                obs = next_obs
                
            step += 1

    # --- trajectory output ---
    traj_path = os.path.join(out_dir, "trajectory.png")
    plot_trajectory(states, acts, env_cfg, traj_path)
    print(f"wrote {traj_path}")

    # 3. Closing the wrapper ensures the final video frames flush to disk cleanly
    env.close()


if __name__ == "__main__":
    main()
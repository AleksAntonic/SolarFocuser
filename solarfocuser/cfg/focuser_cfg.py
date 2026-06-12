"""
Configuration for the SolarFocuser micro-gravity Box2D task.

EnvConfig merges the former (frozen) physics EnvConfig and the former UserArgs
into ONE mutable class, because the env is now constructed as
``task_class(args=env_cfg, ...)`` and uses the same object as ``self.cfg`` and
``self._args``. Every field the env reads from either object lives here.

It additionally carries the RL-side nested namespaces that the task registry
and PPO runner touch (``seed``, ``env.play``, ``env.num_envs``, ...).

FocuserCfgPPO is the train config consumed by runners/algs/ppo.py.
"""

from enum import Enum
import math


class State(Enum):
    """State definition (unchanged from the Gymnasium env)."""

    x = 0          # sail x position (m, world frame)
    y = 1          # sail y position (m, world frame)
    x_dot = 2      # sail x velocity (m/s)
    y_dot = 3      # sail y velocity (m/s)
    theta = 4      # sail attitude angle (rad) -- normal direction of the sail
    theta_dot = 5  # sail angular velocity (rad/s)
    ast_x = 6      # asteroid x position (m, world frame)
    ast_y = 7      # asteroid y position (m, world frame)
    ast_x_dot = 8  # asteroid x velocity (m/s)
    ast_y_dot = 9  # asteroid y velocity (m/s)
    ast_theta = 10      # asteroid rotation angle (rad)
    ast_theta_dot = 11  # asteroid spin rate (rad/s)


# physical constants
G_GRAV = 6.674e-11       # m^3 kg^-1 s^-2
AU_M = 1.495978707e11    # m
SOLAR_FLUX_1AU = 1361.0  # W/m^2
C_LIGHT = 2.99792458e8   # m/s


class EnvConfig:
    """Single merged env config (physics + user args + RL namespaces)."""

    seed = 42
    device = "cpu"  # torch device the vectorized task returns tensors on

    # read by task_registry.make_env at TOP level: render_mode = "human" if enable_viewer
    # keep False for training (human mode opens a window and throttles to render_fps)
    enable_viewer = False

    # ------------------------------------------------------------------ #
    # RL / VECTORIZED-TASK NAMESPACE (read by FocuserTask, PPO, registry)
    # ------------------------------------------------------------------ #
    class env:
        play = False                  # read by PPO.init_writer
        num_envs = 16                 # parallel Box2D worlds
        num_observations = 9
        num_privileged_obs = None     # None -> critic uses actor obs
        num_actions = 3               # [rcs_x, rcs_y, wheel_torque]
        max_episode_length = 1200     # control steps per episode (steps * sim_dt = sim seconds)

    class viewer:
        enable_viewer = False         # registry writes this into the saved json

    # reward terms: each name maps to REWARDS.reward_<name>(task), the ONLY
    # source of reward in the system (terminals included)
    class rewards:
        # which reward terms to use (order determines logging keys)
        terms = [
            "track_position",
            "track_orientation",
            "base_velocity",
            "action_rate",
            "crash",
            "capture",
        ]

        class scales:
            track_position = 1.0      # exp(-dist) shaping toward the asteroid
            track_orientation = 0.05  # keep the concentrator pointed at the sun
            base_velocity = 0.02      # penalize relative speed
            action_rate = 0.005       # penalize action changes (smoothness)
            crash = 100.0             # terminal: REWARDS.reward_crash returns -1 -> -100
            capture = 100.0           # terminal: REWARDS.reward_capture returns +1 -> +100

        position_sigma = 500.0        # m, length scale of the position shaping

    # ------------------------------------------------------------------ #
    # SIMULATION / RENDERING
    # ------------------------------------------------------------------ #
    viewport_width = 1000
    viewport_height = 800

    world_width_m = 4000.0
    scale = viewport_width / world_width_m            # px per metre
    world_width = world_width_m                       # m
    world_height = world_width_m * viewport_height / viewport_width  # m

    background_color = (5, 5, 20)
    stars = True
    n_stars = 200

    fps = 60
    render_fps = 60
    sim_dt = 1.0                 # seconds of simulated time per env step
    box2d_vel_iters = 10
    box2d_pos_iters = 10

    # ------------------------------------------------------------------ #
    # PHYSICAL CONSTANTS
    # ------------------------------------------------------------------ #
    G = G_GRAV
    AU = AU_M
    solar_flux_1au = SOLAR_FLUX_1AU
    c_light = C_LIGHT
    heliocentric_distance = AU_M

    # ------------------------------------------------------------------ #
    # SUN
    # ------------------------------------------------------------------ #
    sun_direction = (1.0, 0.0)
    sun_render_distance = 0.45
    sun_radius_px = 28.0

    # ------------------------------------------------------------------ #
    # SOLAR SAIL / CONCENTRATOR
    # ------------------------------------------------------------------ #
    sail_reflector_radius = 100.0
    sail_reflectivity = 0.85
    sail_areal_density = 0.05
    sail_total_mass = 1000.0
    sail_render_segments = 12        # Box2D polygon vertex cap is 16
    sail_body_color = (230, 230, 245)
    sail_edge_color = (150, 160, 200)

    # ------------------------------------------------------------------ #
    # ASTEROID
    # ------------------------------------------------------------------ #
    asteroid_diameter = 490.0
    asteroid_radius = 245.0
    asteroid_density = 2700.0
    asteroid_albedo = 0.2
    asteroid_rotation_period = 4.3 * 3600.0
    asteroid_cp = 800.0
    asteroid_render_segments = 14    # Box2D polygon vertex cap is 16
    asteroid_body_color = (110, 95, 80)
    asteroid_edge_color = (70, 60, 50)

    # ------------------------------------------------------------------ #
    # CONTROL HARDWARE
    # ------------------------------------------------------------------ #
    rcs_single_thrust = 1.1          # N, Aerojet MR-103
    rcs_cluster_size = 4
    rcs_thrust_points = 8 # Must be even

    # Scaling down max thrust to account for the variation in the moment arms of the RCS clusters
    rcs_angle = 2.0 * math.pi / rcs_thrust_points
    working_angle = rcs_angle / 2.0
    rcs_max_thrust_per_axis = 0.0
    for cluster in range(rcs_thrust_points / 2.0):
        rcs_max_thrust_per_axis += rcs_single_thrust * rcs_cluster_size * math.sin(working_angle)
        working_angle += rcs_angle

    rcs_fine_thrust = 0.9
    rcs_fine_points = 6

    reaction_wheel_torque = 0.4      # N*m per wheel
    reaction_wheels = 4
    rcs_torque_arm = sail_reflector_radius

    # ------------------------------------------------------------------ #
    # EPISODE LIMITS (single-world internals)
    # ------------------------------------------------------------------ #
    # internal truncation of the underlying gym env; keep it ABOVE
    # env.max_episode_length so the vectorized wrapper controls timeouts
    max_episode_steps = 1_000_000
    capture_distance = 50.0
    capture_speed = 0.5
    bounds_margin = 0.05

    # ------------------------------------------------------------------ #
    # FORMER UserArgs FIELDS (the env reads these off self._args)
    # ------------------------------------------------------------------ #
    initial_position = None              # (x_frac, y_frac, theta)
    initial_state = None                 # (x, y, x_dot, y_dot, theta, theta_dot)
    initial_asteroid_position = None     # (x_frac, y_frac, theta)
    initial_asteroid_velocity = None     # (x_dot, y_dot, theta_dot)

    render_sail_center_position = True
    render_asteroid_center_position = True
    render_sun_direction = True
    render_force_vectors = False

    random_initial_position = True

    rcs_thruster_range = 1.0
    reaction_wheel_range = 1.0
    mass_correction_factor = 1.0
    reflectivity_range = 1.0

    enable_radiation_pressure = True
    enable_mutual_gravity = True


# backwards-compat aliases
UserArgs = EnvConfig
FocuserCfg = EnvConfig


class TrainConfig:
    """Train config consumed by runners/algs/ppo.py."""

    algorithm_name = "PPO"
    seed = 42

    class runner:
        experiment_name = "focuser"
        run_name = "run"
        max_iterations = 1000
        save_interval = 100
        num_steps_per_env = 64

        normalize_observation = True

        # gif recording through env.fake_camera_img()
        record_gif = False
        record_gif_interval = 50
        record_iters = 2

        wandb = False
        wandb_group = "focuser"

        resume = False
        load_run = -1
        checkpoint = -1

    class algorithm:
        learning_rate = 3.0e-4
        schedule = "adaptive"          # uses desired_kl
        desired_kl = 0.01
        gamma = 0.99
        lam = 0.95
        bootstrap = True               # requires infos['time_outs'] (provided)
        clip_param = 0.2
        use_clipped_value_loss = True
        surrogate_coef = 1.0
        value_loss_coef = 1.0
        entropy_coef = 0.005
        max_grad_norm = 1.0
        num_mini_batches = 4
        num_learning_epochs = 5

    class policy:
        actor_hidden_dims = [256, 128, 64]
        critic_hidden_dims = [256, 128, 64]
        activation = "elu"
        log_std_init = 0.0


# alias kept for any older snippets
FocuserCfgPPO = TrainConfig
"""
A Box2D environment for Gymnasium modelling a micro-gravity solar-sail
rendezvous with an asteroid.

Three bodies:
  - The Sun: a non-physical point source, off-screen, that only generates
    radiation pressure on the sail.
  - The solar sail / concentrator: the controllable agent. It is pushed by
    solar radiation pressure (modulated by its attitude relative to the sun)
    and mutually attracted to the asteroid by gravity. It is steered with RCS
    thrusters (translation) and reaction wheels (attitude).
  - The asteroid: a massive dynamic body. It mutually attracts the sail and
    spins at its rotation period.

Box2D world gravity is set to zero; mutual gravity and radiation pressure are
applied manually each step as external forces.

Structural style mirrors the Falcon-9 RocketLander environment by
Reuben Ferrante / Dylan Vogel / Gerasimos Maltezos / Benjamin Stadler.
"""

import copy
import warnings
from typing import List, Tuple

import Box2D
import gymnasium as gym
import numpy as np
import pygame
from Box2D.b2 import (
    circleShape,
    contactListener,
    fixtureDef,
    polygonShape,
)
from gymnasium import spaces

from solarfocuser.cfg.focuser_cfg import EnvConfig, State, UserArgs

# for warnings
YELLOW = "\x1b[33;20m"
ENDL = "\x1b[0m"

# for referencing different parts of the state
DEGTORAD = np.pi / 180
XX = State.x.value
YY = State.y.value
X_DOT = State.x_dot.value
Y_DOT = State.y_dot.value
THETA = State.theta.value
THETA_DOT = State.theta_dot.value
AST_X = State.ast_x.value
AST_Y = State.ast_y.value
AST_X_DOT = State.ast_x_dot.value
AST_Y_DOT = State.ast_y_dot.value
AST_THETA = State.ast_theta.value
AST_THETA_DOT = State.ast_theta_dot.value


## CONTACT DETECTOR


class ContactDetector(contactListener):
    """Callback class for making/breaking contact in the environment.

    A contact between the sail and the asteroid is a collision (the sail has
    crashed into the rock), which ends the episode.
    """

    def __init__(self, env: gym.Env):
        """Constructor method.

        Args:
            env (gym.Env): gym environment to listen on
        """
        contactListener.__init__(self)
        self.env = env

    def BeginContact(self, contact):
        """Called when contact between two bodies begins."""
        bodies = [contact.fixtureA.body, contact.fixtureB.body]
        if self.env.sail in bodies and self.env.asteroid in bodies:
            self.env.contact_flag = True
            # touching the asteroid at speed counts as a crash
            self.env.game_over = True

    def EndContact(self, contact):
        """Called when contact between two bodies ends."""
        pass


## GYMNASIUM METHODS


class _SailWorld(gym.Env):
    """INTERNAL single-world Box2D physics env. Do not register this;
    register SolarFocuser (below), which vectorizes N of these."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, args: EnvConfig, render_mode=None):
        """Constructor method."""
        if not isinstance(args, EnvConfig):
            raise TypeError(
                f"args must be an EnvConfig instance, got {type(args).__name__}"
            )

        self.cfg = args
        self._args = args
        self._parse_user_args()

        # environment -- ZERO world gravity; gravity is applied manually
        self.world = Box2D.b2World(gravity=(0, 0))

        # ------------------------------------------------------------------ #
        # ACTION SPACE
        #   action[0]: RCS thrust along body x (-1..1) * max thrust per axis
        #   action[1]: RCS thrust along body y (-1..1) * max thrust per axis
        #   action[2]: reaction-wheel torque (-1..1) * max wheel torque
        # ------------------------------------------------------------------ #
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0]),
            high=np.array([1.0, 1.0, 1.0]),
            dtype=np.float64,
        )

        # ------------------------------------------------------------------ #
        # OBSERVATION SPACE -- full 12-element state (see State enum)
        # positions are bounded by the world, everything else unbounded
        # ------------------------------------------------------------------ #
        w, h = self.cfg.world_width, self.cfg.world_height
        self.observation_space = spaces.Box(
            low=np.array(
                [0, 0, -np.inf, -np.inf, -np.inf, -np.inf,
                 0, 0, -np.inf, -np.inf, -np.inf, -np.inf]
            ),
            high=np.array(
                [w, h, np.inf, np.inf, np.inf, np.inf,
                 w, h, np.inf, np.inf, np.inf, np.inf]
            ),
            dtype=np.float64,
        )

        # bodies
        self.sail = None
        self.asteroid = None
        self.particles = []
        self.drawlist = []

        # rendering
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        self.canvas = None
        self.stars = []

        # state variables
        self.state = []
        self.previous_state = None
        self.game_over = False
        self.contact_flag = False
        self.prev_shaping = None
        self.step_count = 0

        # cached derived physical quantities (filled at reset)
        self._sail_mass = None
        self._asteroid_mass = None

        self.reset()

    def reset(self, seed=None, options=None):
        """Reset the environment."""
        super().reset(seed=seed)
        np.random.seed(seed=seed)

        self._destroy()
        self.world.contactListener = ContactDetector(self)

        # state variables
        self.state = []
        self.previous_state = None
        self.game_over = False
        self.contact_flag = False
        self.prev_shaping = None
        self.step_count = 0
        self.drawlist = []

        # ---------------------------------------------------------------- #
        # SAIL initial position (centre-ish of the world by default)
        # ---------------------------------------------------------------- #
        sail_pos = [0.65 * self.cfg.world_width, 0.5 * self.cfg.world_height]
        sail_new = None

        if self._args.initial_state is not None:
            assert (
                len(self._args.initial_state) == 6
            ), "initial_state must be length 6"
            if self._args.random_initial_position:
                warnings.warn(
                    YELLOW
                    + "WARN: initial_state set but position will be randomized"
                    + ENDL
                )
            sail_pos[0] = self._args.initial_state[0] * self.cfg.world_width
            sail_pos[1] = self._args.initial_state[1] * self.cfg.world_height
            sail_new = {
                "x_dot": self._args.initial_state[2],
                "y_dot": self._args.initial_state[3],
                "theta": self._args.initial_state[4],
                "theta_dot": self._args.initial_state[5],
            }
        elif self._args.initial_position is not None:
            assert (
                len(self._args.initial_position) == 3
            ), "initial_position must be length 3 (x, y, theta)"
            sail_pos[0] = self._args.initial_position[0] * self.cfg.world_width
            sail_pos[1] = self._args.initial_position[1] * self.cfg.world_height
            sail_new = {"theta": self._args.initial_position[2]}

        if self._args.random_initial_position:
            sail_pos[0] = np.random.uniform(0.55, 0.85) * self.cfg.world_width
            sail_pos[1] = np.random.uniform(0.3, 0.7) * self.cfg.world_height
            sail_new = {"theta": np.random.uniform(-np.pi, np.pi)}

        # ---------------------------------------------------------------- #
        # ASTEROID initial position (left of centre by default)
        # ---------------------------------------------------------------- #
        ast_pos = [0.30 * self.cfg.world_width, 0.5 * self.cfg.world_height]
        if self._args.initial_asteroid_position is not None:
            assert (
                len(self._args.initial_asteroid_position) == 3
            ), "initial_asteroid_position must be length 3 (x, y, theta)"
            ast_pos[0] = (
                self._args.initial_asteroid_position[0] * self.cfg.world_width
            )
            ast_pos[1] = (
                self._args.initial_asteroid_position[1] * self.cfg.world_height
            )

        # create bodies
        self._create_asteroid(ast_pos[0], ast_pos[1])
        self._create_sail(sail_pos[0], sail_pos[1])
        self.drawlist = [self.asteroid, self.sail]

        # cache masses (after density-based creation)
        self._sail_mass = self.sail.mass
        self._asteroid_mass = self.asteroid.mass

        # asteroid initial spin (rotation period) and any user velocity
        ast_spin = 2 * np.pi / self.cfg.asteroid_rotation_period
        self.asteroid.angularVelocity = ast_spin
        if self._args.initial_asteroid_position is not None:
            self.asteroid.angle = self._args.initial_asteroid_position[2]
        if self._args.initial_asteroid_velocity is not None:
            av = self._args.initial_asteroid_velocity
            self.asteroid.linearVelocity = (av[0], av[1])
            self.asteroid.angularVelocity = av[2]

        # apply sail initial dynamics
        if sail_new is not None:
            self.adjust_dynamics(self.sail, **sail_new)

        # rendering scenery
        self._create_stars()

        obs, _, _, _, info = self.step(np.array([0.0, 0.0, 0.0]))
        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Simulate the environment forward by one step.

        Args:
            action (np.ndarray): [rcs_x, rcs_y, wheel_torque], each in [-1, 1]

        Returns:
            Tuple: observation, reward, done, truncated, info
        """
        if isinstance(action, list):
            action = np.array(action)

        assert (
            action.shape == self.action_space.shape
        ), f"Incorrect action shape, expected {self.action_space.shape} got {action.shape}"

        action = np.clip(action, self.action_space.low, self.action_space.high)

        # apply controls
        self.__apply_rcs_thrust(action)
        self.__apply_reaction_wheel(action)

        # apply natural forces
        self.__apply_mutual_gravity()
        self.__apply_radiation_pressure()

        # step physics
        self.previous_state = self.state
        self.state = self.__generate_state()
        self._update_particles()

        # NOTE: this env computes NO reward. All reward terms -- shaped AND
        # terminal -- live exclusively in solarfocuser.rewards.REWARDS and are
        # evaluated by the vectorized SolarFocuser task. This env only reports
        # the physics state and the crash/capture termination flags.
        reward = 0.0

        # termination conditions
        truncated = False

        sx, sy = self.state[XX], self.state[YY]
        m = self.cfg.bounds_margin
        out_of_bounds = (
            sx < -m * self.cfg.world_width
            or sx > (1 + m) * self.cfg.world_width
            or sy < -m * self.cfg.world_height
            or sy > (1 + m) * self.cfg.world_height
        )

        crash = bool(self.game_over or out_of_bounds)
        # success: close to the asteroid surface at low relative speed
        capture = (not crash) and self.__check_capture(self.state)
        done = crash or capture

        self.step_count += 1
        if self.step_count >= self.cfg.max_episode_steps:
            truncated = True

        if self.render_mode == "human":
            self._render_frame()

        info = self.__build_info()
        info["crash"] = crash
        info["capture"] = capture

        return (
            np.array(self.state),
            reward,
            done,
            truncated,
            info,
        )

    def close(self):
        super().close()
        self._destroy()
        if self.window is not None:
            pygame.display.quit()

    ## QUERY FUNCTIONS

    def get_mass_properties(self) -> Tuple[float, float, float, float]:
        """Get mass and inertia of the sail and the asteroid.

        Returns:
            Tuple[float, float, float, float]:
                sail_mass, sail_inertia, asteroid_mass, asteroid_inertia
        """
        return (
            self.sail.mass / self._args.mass_correction_factor,
            self.sail.inertia / self._args.mass_correction_factor,
            self.asteroid.mass,
            self.asteroid.inertia,
        )

    def get_dimensional_properties(self) -> Tuple[float, float]:
        """Get the characteristic radii of the two physical bodies.

        Returns:
            Tuple[float, float]: sail_reflector_radius, asteroid_radius (m)
        """
        return self.cfg.sail_reflector_radius, self.cfg.asteroid_radius

    def get_asteroid_state(self) -> Tuple[float, float, float, float, float, float]:
        """Get the asteroid state (x, y, x_dot, y_dot, theta, theta_dot)."""
        assert self.asteroid is not None, "Please call reset() first!"
        p = self.asteroid.position
        v = self.asteroid.linearVelocity
        return (
            p.x,
            p.y,
            v.x,
            v.y,
            copy.deepcopy(self.asteroid.angle),
            self.asteroid.angularVelocity,
        )

    def get_relative_state(self) -> Tuple[float, float, float, float]:
        """Sail position and velocity relative to the asteroid (rendezvous frame).

        Returns:
            Tuple: dx, dy, dvx, dvy  (sail minus asteroid)
        """
        return (
            self.state[XX] - self.state[AST_X],
            self.state[YY] - self.state[AST_Y],
            self.state[X_DOT] - self.state[AST_X_DOT],
            self.state[Y_DOT] - self.state[AST_Y_DOT],
        )

    def get_force_budget(self) -> dict:
        """Report the natural and control force magnitudes for the agent."""
        dx = self.state[XX] - self.state[AST_X]
        dy = self.state[YY] - self.state[AST_Y]
        r = np.hypot(dx, dy)
        f_grav = (
            self.cfg.G * self._sail_mass * self._asteroid_mass / max(r, 1e-6) ** 2
        )
        f_srp = self.__srp_max_force()
        return {
            "separation_m": r,
            "gravity_N": f_grav,
            "max_srp_N": f_srp,
            "max_rcs_per_axis_N": self._actual_rcs_max_thrust,
            "max_wheel_torque_Nm": self._actual_wheel_torque,
        }

    ## ENVIRONMENT HELPER FUNCTIONS

    def _destroy(self):
        if self.asteroid is None:
            return
        self.world.contactListener = None
        self._clean_particles(True)
        if self.sail is not None:
            self.world.DestroyBody(self.sail)
            self.sail = None
        if self.asteroid is not None:
            self.world.DestroyBody(self.asteroid)
            self.asteroid = None
        self.drawlist = []

    def _parse_user_args(self):
        """Parse UserArgs into derived config values.

        Concerned with hardware scaling / malfunction parameters and the
        physically-derived sail and asteroid masses.
        """
        _SailWorld._validate_float_in_range(
            self._args.rcs_thruster_range, "rcs_thruster_range"
        )
        _SailWorld._validate_float_in_range(
            self._args.reaction_wheel_range, "reaction_wheel_range"
        )
        _SailWorld._validate_float_in_range(
            self._args.mass_correction_factor, "mass_correction_factor",
            min_value=0.1, max_value=10.0,
        )
        _SailWorld._validate_float_in_range(
            self._args.reflectivity_range, "reflectivity_range"
        )

        # control hardware actuals
        self._actual_rcs_max_thrust = (
            self._args.rcs_thruster_range * self.cfg.rcs_max_thrust_per_axis
        )
        self._actual_wheel_torque = (
            self._args.reaction_wheel_range
            * self.cfg.reaction_wheel_torque
            * self.cfg.reaction_wheels
        )
        self._actual_reflectivity = (
            self._args.reflectivity_range * self.cfg.sail_reflectivity
        )

        # physically-derived masses
        self._nominal_sail_mass = (
            self.cfg.sail_total_mass * self._args.mass_correction_factor
        )
        ast_volume = (4.0 / 3.0) * np.pi * self.cfg.asteroid_radius**3
        self._nominal_asteroid_mass = self.cfg.asteroid_density * ast_volume

        # sun direction as a normalised vector
        s = np.array(self.cfg.sun_direction, dtype=float)
        self._sun_unit = s / (np.linalg.norm(s) + 1e-12)

    @staticmethod
    def _validate_float_in_range(value, name, min_value=0.0, max_value=1.0):
        """Helper: validate that value is numeric and within [min, max]."""
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise TypeError(f"{name} must be numeric, got {type(value).__name__}")
        if not (min_value <= value <= max_value):
            raise ValueError(
                f"{name} must be between {min_value} and {max_value}, got {value}"
            )
        return True

    def __srp_max_force(self) -> float:
        """Maximum solar radiation pressure force (sail facing the sun).

        F = (1 + reflectivity) * (flux / c) * Area, scaled to the configured
        heliocentric distance by the inverse-square law.
        """
        if not self._args.enable_radiation_pressure:
            return 0.0
        area = np.pi * self.cfg.sail_reflector_radius**2
        flux = self.cfg.solar_flux_1au * (
            self.cfg.AU / self.cfg.heliocentric_distance
        ) ** 2
        pressure = flux / self.cfg.c_light
        return (1.0 + self._actual_reflectivity) * pressure * area

    def __check_capture(self, state: list) -> bool:
        """Success when the sail is just above the asteroid surface and slow.

        The surface is referenced from the maximum jittered radius so that the
        capture window is reached before the collision (crash) detector fires.
        """
        dx = state[XX] - state[AST_X]
        dy = state[YY] - state[AST_Y]
        max_radius = self.cfg.asteroid_radius * 1.20  # jitter envelope
        surface_dist = np.hypot(dx, dy) - max_radius

        dvx = state[X_DOT] - state[AST_X_DOT]
        dvy = state[Y_DOT] - state[AST_Y_DOT]
        rel_speed = np.hypot(dvx, dvy)

        return (
            0.0 < surface_dist < self.cfg.capture_distance
            and rel_speed < self.cfg.capture_speed
        )

    ## FORCE COMPUTATIONS

    def __apply_rcs_thrust(self, action):
        """Apply RCS translational thrust in the sail body frame.

        Thrust is applied at the centre of mass (net translation); the RCS
        torque authority is handled separately via the reaction-wheel channel
        and the rim-mounted geometry is reflected in the torque arm config.
        """
        fx_cmd = float(action[0]) * self._actual_rcs_max_thrust
        fy_cmd = float(action[1]) * self._actual_rcs_max_thrust

        if fx_cmd == 0.0 and fy_cmd == 0.0:
            return

        # rotate body-frame thrust into world frame using the sail attitude
        ang = self.sail.angle
        cos, sin = np.cos(ang), np.sin(ang)
        fx_world = cos * fx_cmd - sin * fy_cmd
        fy_world = sin * fx_cmd + cos * fy_cmd

        self.sail.ApplyForceToCenter((fx_world, fy_world), True)

        # visual exhaust particles (opposite to thrust)
        mag = np.hypot(fx_world, fy_world)
        if mag > 0:
            self._spawn_thrust_particles(-fx_world / mag, -fy_world / mag, mag)

    def __apply_reaction_wheel(self, action):
        """Apply reaction-wheel torque about the sail centre of mass."""
        torque = float(action[2]) * self._actual_wheel_torque
        if torque != 0.0:
            self.sail.ApplyTorque(torque, True)

    def __apply_mutual_gravity(self):
        """Apply Newtonian mutual gravity between the sail and the asteroid."""
        if not self._args.enable_mutual_gravity:
            return

        ps = self.sail.position
        pa = self.asteroid.position
        dx = pa.x - ps.x
        dy = pa.y - ps.y
        r2 = dx * dx + dy * dy
        r = np.sqrt(r2)
        if r < 1e-6:
            return

        f = self.cfg.G * self._sail_mass * self._asteroid_mass / r2
        ux, uy = dx / r, dy / r

        # equal and opposite forces (Newton's third law)
        self.sail.ApplyForceToCenter((f * ux, f * uy), True)
        self.asteroid.ApplyForceToCenter((-f * ux, -f * uy), True)

    def __apply_radiation_pressure(self):
        """Apply solar radiation pressure to the sail.

        The sun is a point source in direction self._sun_unit. The sail's
        attitude (theta) defines its surface normal; the absorbed/reflected
        momentum depends on the cosine of the angle between the sail normal and
        the incoming sunlight. Force always pushes away from the sun (no
        thrust component pulling toward the sun).
        """
        if not self._args.enable_radiation_pressure:
            return

        f_max = self.__srp_max_force()
        if f_max <= 0.0:
            return

        # sail normal in world frame (theta measured from +x)
        n = np.array([np.cos(self.sail.angle), np.sin(self.sail.angle)])

        # incoming light travels from the sun toward the sail: direction = -sun_unit
        light_dir = -self._sun_unit

        # cos of incidence between sail normal and the sun direction
        cos_inc = np.dot(n, self._sun_unit)

        # only the illuminated face matters; magnitude scales with |cos|, and
        # the resulting force is directed along the light propagation,
        # projected by cos_inc (specular-reflection simplification).
        eff = abs(cos_inc)
        if eff <= 0.0:
            return

        force = f_max * eff * light_dir
        self.sail.ApplyForceToCenter((float(force[0]), float(force[1])), True)

    def __generate_state(self):
        """Step Box2D and return the new full state vector."""
        self.world.Step(
            self.cfg.sim_dt,
            self.cfg.box2d_vel_iters,
            self.cfg.box2d_pos_iters,
        )

        sp = self.sail.position
        sv = self.sail.linearVelocity
        ap = self.asteroid.position
        av = self.asteroid.linearVelocity

        return [
            sp.x,
            sp.y,
            sv.x,
            sv.y,
            self.sail.angle,
            self.sail.angularVelocity,
            ap.x,
            ap.y,
            av.x,
            av.y,
            self.asteroid.angle,
            self.asteroid.angularVelocity,
        ]

    def __build_info(self) -> dict:
        """Diagnostic info dict returned each step."""
        return {
            "sail_mass": self._sail_mass,
            "asteroid_mass": self._asteroid_mass,
            "contact": self.contact_flag,
        }

    ## BODY CREATION

    def _create_sail(self, pos_x: float, pos_y: float):
        """Create the dynamic solar-sail body as a thin disk (n-gon)."""
        n = self.cfg.sail_render_segments
        r = self.cfg.sail_reflector_radius
        verts = [
            (r * np.cos(2 * np.pi * i / n), r * np.sin(2 * np.pi * i / n))
            for i in range(n)
        ]

        # back-solve a density that yields the configured total mass for this
        # polygon area: density = mass / area; Box2D computes mass = density*area
        area = np.pi * r**2  # disk approximation
        density = self._nominal_sail_mass / area

        self.sail = self.world.CreateDynamicBody(
            position=(pos_x, pos_y),
            angle=0.0,
            fixtures=fixtureDef(
                shape=polygonShape(vertices=verts),
                density=density,
                friction=0.2,
                restitution=0.0,
                categoryBits=0x0010,
                maskBits=0x0001,  # collide with the asteroid only
            ),
        )
        self.sail.color1 = self.cfg.sail_body_color
        self.sail.color2 = self.cfg.sail_edge_color
        # very low linear/angular damping (deep space)
        self.sail.linearDamping = 0.0
        self.sail.angularDamping = 0.0
        # microgravity forces are tiny; never let the body sleep or it freezes
        self.sail.sleepingAllowed = False

    def _create_asteroid(self, pos_x: float, pos_y: float):
        """Create the dynamic asteroid body as an irregular polygon."""
        n = self.cfg.asteroid_render_segments
        r = self.cfg.asteroid_radius
        # slightly irregular rock outline (deterministic per reset via np seed)
        verts = []
        for i in range(n):
            ang = 2 * np.pi * i / n
            jitter = 1.0 + 0.12 * np.sin(3 * ang) + 0.06 * np.cos(5 * ang)
            verts.append((r * jitter * np.cos(ang), r * jitter * np.sin(ang)))

        area = np.pi * r**2
        density = self._nominal_asteroid_mass / area

        self.asteroid = self.world.CreateDynamicBody(
            position=(pos_x, pos_y),
            angle=0.0,
            fixtures=fixtureDef(
                shape=polygonShape(vertices=verts),
                density=density,
                friction=0.5,
                restitution=0.0,
                categoryBits=0x0001,
                maskBits=0x0010,  # collide with the sail only
            ),
        )
        self.asteroid.color1 = self.cfg.asteroid_body_color
        self.asteroid.color2 = self.cfg.asteroid_edge_color
        self.asteroid.linearDamping = 0.0
        self.asteroid.angularDamping = 0.0
        # keep the asteroid awake so its spin and drift persist
        self.asteroid.sleepingAllowed = False

    ## PARTICLES (visual RCS exhaust)

    def _spawn_thrust_particles(self, ux, uy, mag):
        """Spawn a small exhaust particle in the (ux, uy) world direction."""
        sp = self.sail.position
        offset = self.cfg.sail_reflector_radius * 0.6
        px = sp.x + ux * offset
        py = sp.y + uy * offset
        particle = self._create_particle(px, py, ttl=0.6, radius=8.0)
        # small kick along exhaust direction (purely aesthetic)
        particle.linearVelocity = (ux * mag, uy * mag)

    def _create_particle(self, x, y, ttl, radius=8.0):
        p = self.world.CreateDynamicBody(
            position=(x, y),
            angle=0.0,
            fixtures=fixtureDef(
                shape=circleShape(radius=radius, pos=(0, 0)),
                density=1.0,
                friction=0.0,
                restitution=0.0,
                categoryBits=0x0100,
                maskBits=0x0000,  # collide with nothing
            ),
        )
        p.ttl = ttl
        self.particles.append(p)
        self._clean_particles(False)
        return p

    def _update_particles(self):
        for obj in self.particles:
            obj.ttl -= 0.1
            c = min(255, max(50, int(50 + 255 * obj.ttl)))
            obj.color1 = (255, c, max(40, c // 2))
            obj.color2 = (255, c, max(40, c // 2))
        self._clean_particles(False)

    def _clean_particles(self, all_particles):
        while self.particles and (all_particles or self.particles[0].ttl < 0):
            self.world.DestroyBody(self.particles.pop(0))

    ## RENDERING

    def _create_stars(self):
        self.stars = []
        if not self.cfg.stars:
            return
        for _ in range(self.cfg.n_stars):
            self.stars.append(
                (
                    np.random.uniform(0, self.cfg.viewport_width),
                    np.random.uniform(0, self.cfg.viewport_height),
                    np.random.uniform(0.5, 1.8),
                )
            )

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.cfg.viewport_width, self.cfg.viewport_height)
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        self.canvas = pygame.Surface(
            (self.cfg.viewport_width, self.cfg.viewport_height)
        )
        self.canvas.fill(self.cfg.background_color)

        self._render_stars()
        self._render_sun()
        self._render_bodies()
        self._render_markers()

        # flip so +y is up
        self.canvas = pygame.transform.flip(self.canvas, False, True)

        if self.render_mode == "human":
            self.window.blit(self.canvas, self.canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.canvas)), axes=(1, 0, 2)
            )

    def _world_to_px(self, x, y):
        """Convert world metres to pixel coordinates."""
        return (x * self.cfg.scale, y * self.cfg.scale)

    def _render_stars(self):
        for sx, sy, sr in self.stars:
            pygame.draw.circle(self.canvas, (220, 220, 235), (sx, sy), sr)

    def _render_sun(self):
        if not self._args.render_sun_direction:
            return
        # draw the sun glyph at the edge of the window in the sun direction
        cx = self.cfg.viewport_width / 2
        cy = self.cfg.viewport_height / 2
        half = self.cfg.sun_render_distance * min(
            self.cfg.viewport_width, self.cfg.viewport_height
        )
        sun_px = (
            cx + self._sun_unit[0] * half,
            cy + self._sun_unit[1] * half,
        )
        pygame.draw.circle(
            self.canvas, (255, 240, 180), sun_px, self.cfg.sun_radius_px
        )
        pygame.draw.circle(
            self.canvas, (255, 200, 90), sun_px, self.cfg.sun_radius_px, width=3
        )

    def _render_bodies(self):
        for obj in self.particles + self.drawlist:
            for f in obj.fixtures:
                trans = f.body.transform
                if isinstance(f.shape, circleShape):
                    center = self._world_to_px(*(trans * f.shape.pos))
                    pygame.draw.circle(
                        self.canvas, obj.color1, center,
                        f.shape.radius * self.cfg.scale,
                    )
                else:
                    path = [self._world_to_px(*(trans * v)) for v in f.shape.vertices]
                    pygame.draw.polygon(self.canvas, obj.color1, path)
                    pygame.draw.aalines(self.canvas, obj.color2, True, path)

    def _render_markers(self):
        if self._args.render_sail_center_position and self.sail is not None:
            self._draw_marker(
                self.sail.position.x, self.sail.position.y,
                self.sail.angle, (120, 200, 255),
            )
        if self._args.render_asteroid_center_position and self.asteroid is not None:
            self._draw_marker(
                self.asteroid.position.x, self.asteroid.position.y,
                self.asteroid.angle, (255, 120, 120),
            )

    def _draw_marker(self, x, y, theta, color=(255, 255, 255)):
        offset = self.cfg.world_width * 0.012
        cross_vert = [
            (x + offset * np.sin(theta), y - offset * np.cos(theta)),
            (x - offset * np.sin(theta), y + offset * np.cos(theta)),
        ]
        cross_horiz = [
            (x - offset * np.cos(theta), y - offset * np.sin(theta)),
            (x + offset * np.cos(theta), y + offset * np.sin(theta)),
        ]
        pygame.draw.lines(
            self.canvas, color, False,
            [self._world_to_px(*p) for p in cross_horiz], 2,
        )
        pygame.draw.lines(
            self.canvas, color, False,
            [self._world_to_px(*p) for p in cross_vert], 2,
        )

    ## DYNAMICS HELPER

    def adjust_dynamics(self, body, **kwargs):
        """Adjust dynamic parameters of a body (velocities and angle only)."""
        if kwargs.get("x_dot"):
            body.linearVelocity.x = kwargs["x_dot"]
        if kwargs.get("y_dot"):
            body.linearVelocity.y = kwargs["y_dot"]
        if kwargs.get("theta"):
            body.angle = kwargs["theta"]
        if kwargs.get("theta_dot"):
            body.angularVelocity = kwargs["theta_dot"]
        self.state = self.__generate_state()


## ===================================================================== ##
## VECTORIZED RL TASK -- SolarFocuser (register THIS class)
##
## FocuserTask runs cfg.env.num_envs independent SolarFocuser Box2D worlds
## and exposes the PPO runner contract:
##   step(actions) -> (obs, privileged_obs, rews, dones, infos)
##   infos['time_outs'] always present; episode_length_buf is a torch tensor.
##
## NOTE: _SailWorld (above) is the internal single-world physics env.
## SolarFocuser (below) is the public RL task that PPO consumes; it
## constructs _SailWorld instances internally.
## ===================================================================== ##

import torch
from solarfocuser.rewards.rewards import REWARDS

POS_SCALE = 1000.0  # m, normalization length scale for observations

class SolarFocuser:
    """THE RL task: vectorized over N internal _SailWorld Box2D worlds.

    This is the class to register in the task registry. It implements the
    PPO runner contract (num_envs, num_obs, step -> 5-tuple, time_outs...).
    """

    def __init__(self, args, render_mode=None):
        self.cfg = args
        self.device = getattr(args, "device", "cpu")

        # PPO-facing dimensions
        self.num_envs = args.env.num_envs
        self.num_obs = args.env.num_observations
        self.num_privileged_obs = args.env.num_privileged_obs
        self.num_actions = args.env.num_actions
        self.dt = args.sim_dt
        self.max_episode_length = args.env.max_episode_length

        # sanity: the underlying gym env must not truncate before the wrapper
        assert args.max_episode_steps > self.max_episode_length, (
            "cfg.max_episode_steps (inner env truncation) must exceed "
            "cfg.env.max_episode_length (wrapper timeout)"
        )

        # N independent Box2D worlds; only env 0 carries a render mode so that
        # fake_camera_img() / play scripts can draw without opening N windows
        self.envs = []
        for i in range(self.num_envs):
            rm = (render_mode if render_mode is not None else "rgb_array") if i == 0 else None
            self.envs.append(_SailWorld(args=args, render_mode=rm))

        # reward terms: prefer explicit cfg.rewards.terms list, fallback to
        # the names defined under cfg.rewards.scales
        self.reward_terms = []
        reward_names = getattr(args.rewards, "terms", None)
        if reward_names is None:
            reward_names = [n for n in vars(args.rewards.scales) if not n.startswith("_")]

        for name in reward_names:
            scale = getattr(args.rewards.scales, name)
            fn = getattr(REWARDS, "reward_" + name, None)
            assert fn is not None, f"REWARDS.reward_{name} is not implemented"
            self.reward_terms.append((name, scale, fn))

        # torch buffers (PPO contract)
        N = self.num_envs
        self.obs_buf = torch.zeros(N, self.num_obs, device=self.device)
        self.rew_buf = torch.zeros(N, device=self.device)
        self.reset_buf = torch.zeros(N, device=self.device)
        self.time_out_buf = torch.zeros(N, device=self.device)
        self.episode_length_buf = torch.zeros(N, dtype=torch.long, device=self.device)

        # physics mirrors used by REWARDS and obs construction
        self.raw_states = torch.zeros(N, 12, device=self.device)
        self.rel_pos = torch.zeros(N, 2, device=self.device)   # asteroid - sail
        self.rel_vel = torch.zeros(N, 2, device=self.device)
        self.sail_theta = torch.zeros(N, device=self.device)
        self.sail_theta_dot = torch.zeros(N, device=self.device)
        self.surface_dist = torch.zeros(N, device=self.device)
        self.actions = torch.zeros(N, self.num_actions, device=self.device)
        self.last_actions = torch.zeros(N, self.num_actions, device=self.device)
        self.crashed = torch.zeros(N, device=self.device)
        self.captured = torch.zeros(N, device=self.device)

        # sun angle from the configured direction (constant)
        sd = np.asarray(args.sun_direction, dtype=float)
        self.sun_angle = float(np.arctan2(sd[1], sd[0]))
        # surface reference: jitter envelope of the asteroid polygon
        self.surface_radius = args.asteroid_radius * 1.20

        # per-episode logging accumulators
        self.episode_sums = {
            name: torch.zeros(N, device=self.device) for name, _, _ in self.reward_terms
        }
        self.episode_sums["total"] = torch.zeros(N, device=self.device)

        self.extras = {}

        self.reset()

    # ------------------------------------------------------------------ #
    # PPO CONTRACT
    # ------------------------------------------------------------------ #

    def reset(self):
        """Reset all environments. Returns (obs, privileged_obs)."""
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        self._refresh_buffers()
        self._compute_observations()
        return self.obs_buf.clone(), self.get_privileged_observations()

    def get_observations(self):
        return self.obs_buf.clone()

    def get_privileged_observations(self):
        return None  # num_privileged_obs is None -> PPO critic uses actor obs

    def step(self, actions: torch.Tensor):
        """Step every Box2D world once.

        Returns:
            obs, privileged_obs, rewards, dones, infos
            infos always contains 'time_outs' (float tensor, num_envs).
        """
        self.last_actions[:] = self.actions
        self.actions = torch.clamp(
            actions.detach().to(self.device, dtype=torch.float32), -1.0, 1.0
        )
        acts_np = self.actions.cpu().numpy().astype(np.float64)

        crashed = torch.zeros(self.num_envs, device=self.device)
        captured = torch.zeros(self.num_envs, device=self.device)

        for i, env in enumerate(self.envs):
            state, _r_env, done, _trunc, info = env.step(acts_np[i])
            self.raw_states[i] = torch.from_numpy(np.asarray(state, dtype=np.float32))
            if info["crash"]:
                crashed[i] = 1.0
            elif info["capture"]:
                captured[i] = 1.0

        # expose terminal flags as task buffers so REWARDS terms (reward_crash,
        # reward_capture) can read them like any other physics quantity
        self.crashed = crashed
        self.captured = captured

        self.episode_length_buf += 1
        self._refresh_buffers()

        # timeouts (wrapper-controlled; PPO randomizes episode_length_buf at
        # the start of learn(), so these stagger automatically)
        self.time_out_buf = (
            self.episode_length_buf >= self.max_episode_length
        ).float()
        # a timeout coinciding with a terminal is a terminal, not a timeout
        self.time_out_buf = self.time_out_buf * (1.0 - crashed) * (1.0 - captured)

        self.reset_buf = torch.clamp(crashed + captured + self.time_out_buf, max=1.0)

        # rewards: EVERY term (shaped and terminal) comes from
        # solarfocuser.rewards.REWARDS -- no reward is computed anywhere else
        self.rew_buf[:] = 0.0
        for name, scale, fn in self.reward_terms:
            term = fn(self) * scale
            self.rew_buf += term
            self.episode_sums[name] += term
        self.episode_sums["total"] += self.rew_buf

        # build infos BEFORE resetting (so logged sums are the finished episodes')
        infos = {"time_outs": self.time_out_buf.clone()}
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        if len(env_ids) > 0:
            ep = {}
            for key, sums in self.episode_sums.items():
                ep["rew_" + key] = sums[env_ids].mean().item()
            ep["capture_rate"] = captured[env_ids].mean().item()
            ep["crash_rate"] = crashed[env_ids].mean().item()
            ep["episode_length"] = (
                self.episode_length_buf[env_ids].float().mean().item()
            )
            infos["episode"] = ep
            self.reset_idx(env_ids)
            self._refresh_buffers()

        self._compute_observations()

        return (
            self.obs_buf.clone(),
            self.get_privileged_observations(),
            self.rew_buf.clone(),
            self.reset_buf.clone(),
            infos,
        )

    def fake_camera_img(self):
        """RGB frame of env 0 for wandb gif logging (H, W, 3 uint8)."""
        frame = self.envs[0].render()
        if frame is None:  # human mode renders to the window; grab the canvas
            frame = np.zeros(
                (self.cfg.viewport_height, self.cfg.viewport_width, 3), dtype=np.uint8
            )
        return frame

    def close(self):
        for env in self.envs:
            env.close()

    # ------------------------------------------------------------------ #
    # INTERNALS
    # ------------------------------------------------------------------ #

    def reset_idx(self, env_ids):
        """Reset a subset of environments."""
        for i in env_ids.tolist() if torch.is_tensor(env_ids) else env_ids:
            obs, _ = self.envs[i].reset()
            self.raw_states[i] = torch.from_numpy(np.asarray(obs, dtype=np.float32))
            self.episode_length_buf[i] = 0
            self.actions[i] = 0.0
            self.last_actions[i] = 0.0
            for sums in self.episode_sums.values():
                sums[i] = 0.0

    def _refresh_buffers(self):
        """Recompute derived physics tensors from raw_states."""
        s = self.raw_states
        self.rel_pos = s[:, [AST_X, AST_Y]] - s[:, [XX, YY]]
        self.rel_vel = s[:, [AST_X_DOT, AST_Y_DOT]] - s[:, [X_DOT, Y_DOT]]
        self.sail_theta = s[:, THETA]
        self.sail_theta_dot = s[:, THETA_DOT]
        self.surface_dist = torch.norm(self.rel_pos, dim=1) - self.surface_radius

    def _compute_observations(self):
        self.obs_buf = torch.stack(
            [
                self.rel_pos[:, 0] / POS_SCALE,
                self.rel_pos[:, 1] / POS_SCALE,
                self.rel_vel[:, 0],
                self.rel_vel[:, 1],
                torch.sin(self.sail_theta),
                torch.cos(self.sail_theta),
                self.sail_theta_dot,
                self.surface_dist / POS_SCALE,
                torch.cos(self.sail_theta - self.sun_angle),
            ],
            dim=1,
        )


# compatibility alias: earlier integration snippets used this name
FocuserTask = SolarFocuser
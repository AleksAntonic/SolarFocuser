"""
Nominal model-based control (nominal MPC) for the SolarFocuser environment
(solar-sail asteroid mining around 101955 Bennu).

Dynamics (rotating body-fixed frame, spin Omega about +Z):
    r'' = a_grav(r) + a_SRP(r, H_hat) - 2 (w x r') - w x (w x r)

"""

from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple
 
import numpy as np
import scipy.optimize
 
from ..env.solarfocuser import SolarFocuser


class FocuserModel:
    def __init__(self, env: SolarFocuser):
        self.env = env
        # ! TODO input the parameters

        # mu = env.get_gravitational_parameter()
        # R_body = env.get_asteroid_radius()
        # r_SOI = env.get_sphere_of_influence_radius()
        # spin_rate = env.get_spin_rate()
        # self.mu = mu
        # self.R_body = R_body
        # self.r_SOI = r_SOI
        # self.spin_rate = spin_rate
        # self.omega_vec = np.array([0.0, 0.0, spin_rate])

        self.r_hover = None
        self.r_target = None

        self.Ad = None
        self.Bd = None
        self.sample_time = None

        self.sun_direction = np.array([-1.0, 0.0, 0.0])

        self.low_danger_radius = 2.3 * R_body
        self.high_danger_radius = 0.8 * r_SOI
        self.eclipse_radius = R_body

        self.horizon = 50
        self.dt = 1/60



    def get_control_references(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get the reference state and control trajectories for the MPC."""
        # ! TODO implement this method
        return self.r_hover, self.r_target
    def get_discrete_linear_system_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get the discrete-time linear system matrices (Ad, Bd) for the MPC."""
        return self.Ad, self.Bd
    def calculate_control(self, r_hover, r_target):
        if r_hover is None:
            r_hover = np.array([-2.0 * self.R_body, 0.0, 0.0])
        if r_target is None:
            r_target = -self.R_body * (r_hover / np.linalg.norm(r_hover))
 
        self.r_hover = np.asarray(r_hover, dtype=float)
        self.r_target = np.asarray(r_target, dtype=float)
    
    # def in_eclipse(self, r):
    #     """Check if the spacecraft is in eclipse (i.e., behind the asteroid)."""
    #     return np.linalg.norm(r) < self.eclipse_radius
    def _acceleration(
        self, r: np.ndarray, v: np.ndarray, h_hat: np.ndarray
    ) -> np.ndarray:
        """Total acceleration of the nominal SH model in the rotating frame:
        SH gravity + SRP (zero in eclipse) + Coriolis + centrifugal [km/s^2]."""
        a_grav = np.asarray(self.env.gravitational_acceleration(r))
        # if self._in_eclipse(r):
        #     a_srp = np.zeros(3)
        # else:
        a_srp = np.asarray(self.env.srp_acceleration(r, h_hat))
        a_coriolis = -2.0 * np.cross(self.omega_vec, v)
        a_centrifugal = -np.cross(self.omega_vec, np.cross(self.omega_vec, r))
        return a_grav + a_srp + a_coriolis + a_centrifugal
 
    def _rk4_step(
        self, state: np.ndarray, h_hat: np.ndarray, dt: float
    ) -> np.ndarray:
        """One RK4 step of the nominal model holding heading fixed."""
        def deriv(s):
            return np.concatenate([s[3:], self._acceleration(s[:3], s[3:], h_hat)])
        k1 = deriv(state)
        k2 = deriv(state + 0.5 * dt * k1)
        k3 = deriv(state + 0.5 * dt * k2)
        k4 = deriv(state + dt * k3)
        return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    @staticmethod
    def _angles_to_heading(yaw: float, pitch: float) -> np.ndarray:
        cp = np.cos(pitch)
        return np.array([cp * np.cos(yaw), cp * np.sin(yaw), np.sin(pitch)])
 
    @staticmethod
    def _heading_to_angles(h: np.ndarray) -> Tuple[float, float]:
        h = h / (np.linalg.norm(h) + 1e-15)
        pitch = np.arcsin(np.clip(h[2], -1.0, 1.0))
        yaw = np.arctan2(h[1], h[0])
        return yaw, pitch

    def _rollout_cost(
        self, angle_seq: np.ndarray, state0: np.ndarray, h_prev: np.ndarray
    ) -> float:
        """Roll the nominal model forward under a heading sequence; return the
        multi-term cost (soft tracking + aiming, hard safety + reslew limits).
 
        angle_seq is flat [yaw0, pitch0, ...] of length 2*horizon.
        """
        seq = angle_seq.reshape(self.horizon, 2)
        state = state0.copy()
        h_last = h_prev.copy()
        cost = 0.0
        sub_dt = self.dt / self.substeps
 
        for k in range(self.horizon):
            h_hat = self._angles_to_heading(seq[k, 0], seq[k, 1])
 
            dtheta = np.arccos(np.clip(np.dot(h_hat, h_last), -1.0, 1.0))
            if dtheta > self.max_reslew_per_step:
                cost += self.constraint_penalty
            cost += self.w_ctrl * dtheta * dtheta
            h_last = h_hat
 
            for _ in range(self.substeps):
                state = self._rk4_step(state, h_hat, sub_dt)
 
            r = state[:3]
            radius = np.linalg.norm(r)
 
            if radius <= self.R_body * self.crash_margin:
                cost += self.constraint_penalty
            if radius >= 2.0 * self.r_SOI:
                cost += self.constraint_penalty
 
            err = r - self.r_hover
            cost += self.w_track * float(err @ err)
 
            to_target = self.r_target - r
            n_tt = np.linalg.norm(to_target)
            if n_tt > 1e-9:
                to_target /= n_tt
                cost += self.w_aim * (1.0 - np.dot(h_hat, to_target))
 
        return cost

    def _plan(
        self, state0: np.ndarray, h_prev: np.ndarray, warm_start: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Optimize the heading sequence (derivative-free, robust to the eclipse
        discontinuity); return (first_heading, full_angle_seq)."""
        if warm_start is not None:
            x0 = warm_start.copy()
        else:
            y0, p0 = self._heading_to_angles(h_prev)
            x0 = np.tile([y0, p0], self.horizon)
 
        res = scipy.optimize.minimize(
            self._rollout_cost,
            x0,
            args=(state0, h_prev),
            method="Nelder-Mead",
            options={"maxiter": self.max_iter, "xatol": 1e-3, "fatol": 1e-3},
        )
        seq = res.x.reshape(self.horizon, 2)
        first_heading = self._angles_to_heading(seq[0, 0], seq[0, 1])
        return first_heading, res.x
    def discretize_system_matrices(self, sample_time: float) -> None:
        """Exact discretization (matrix exponential) of the SH nominal model
        LINEARIZED about the hover point. Diagnostic only -- it gives the local
        Ad, Bd for stability/observability checks; the MPC plans on the full
        nonlinear SH model, not on these matrices.
 
        Args:
            sample_time (float): discrete sampling time in seconds
 
        Raises:
            AttributeError: if calculate_control_law has not been called first
        """
 
        if self.r_hover is None:
            print("Note: control references not initialized. Using the (default) -X hover point")
            self.calculate_control_law()
 
        self.sample_time = sample_time
        r_eq = self.r_hover
        v_eq = np.zeros(3)
        u_eq = self.sun_direction.copy()
 
        # continuous A: numeric Jacobian of the SH acceleration balance
        eps_r = 1e-4 * self.R_body
        G = np.zeros((3, 3))
        for j in range(3):
            dr = np.zeros(3)
            dr[j] = eps_r
            a_plus = self._acceleration(r_eq + dr, v_eq, u_eq)
            a_minus = self._acceleration(r_eq - dr, v_eq, u_eq)
            G[:, j] = (a_plus - a_minus) / (2.0 * eps_r)
 
        A = np.zeros((self.state_shape, self.state_shape))
        A[0:3, 3:6] = np.eye(3)
        A[3:6, 0:3] = G
        wx, wy, wz = self.omega_vec
        A[3:6, 3:6] = -2.0 * np.array(
            [[0.0, -wz, wy], [wz, 0.0, -wx], [-wy, wx, 0.0]]
        )
 
        # continuous B: numeric Jacobian of SRP accel w.r.t. heading angles
        eps_u = 1e-5
        y0, p0 = self._heading_to_angles(u_eq)
        Bsub = np.zeros((3, self.action_shape))
        for j in range(self.action_shape):
            da = np.zeros(self.action_shape)
            da[j] = eps_u
            h_plus = self._angles_to_heading(y0 + da[0], p0 + da[1])
            h_minus = self._angles_to_heading(y0 - da[0], p0 - da[1])
            a_plus = self._acceleration(r_eq, v_eq, h_plus)
            a_minus = self._acceleration(r_eq, v_eq, h_minus)
            Bsub[:, j] = (a_plus - a_minus) / (2.0 * eps_u)
        B = np.zeros((self.state_shape, self.action_shape))
        B[3:6, :] = Bsub
 
        # exact discretization using matrix exponential
        self.Ad = scipy.linalg.expm(A * sample_time)
 
        # integrate matrix exponential, multiply with B
        Ad_int, _ = scipy.integrate.quad_vec(
            lambda tau: scipy.linalg.expm(A * tau), 0, sample_time
        )
        self.Bd = Ad_int @ B
 
    def run(
        self,
        x0: np.ndarray,
        n_steps: int,
        truth_accel: Optional[Callable] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
        """Run the MPC closed-loop.
 
        Args:
            x0: initial state [x,y,z,vx,vy,vz] in the rotating frame.
            n_steps: number of control steps.
            truth_accel: optional (r,v,h)->accel truth plant. If None, the
                nominal SH model is also used as truth (perfect-model case).
 
        Returns:
            (times, states, headings, outcome) where outcome is one of
            "stable", "crash", "escape".
        """
        if self.r_hover is None:
            self.calculate_control_law()
        if truth_accel is None:
            truth_accel = self._acceleration
 
        state = np.asarray(x0, dtype=float)
        h_prev = self.sun_direction.copy()
        warm = None
        sub_dt = self.dt / self.substeps
 
        times = [0.0]
        states = [state.copy()]
        headings = [h_prev.copy()]
        outcome = "stable"
 
        for step in range(n_steps):
            h_cmd, warm = self._plan(state, h_prev, warm)
            warm = np.concatenate([warm[2:], warm[-2:]])  # receding-horizon shift
 
            for _ in range(self.substeps):
                def deriv(s):
                    return np.concatenate([s[3:], truth_accel(s[:3], s[3:], h_cmd)])
                k1 = deriv(state)
                k2 = deriv(state + 0.5 * sub_dt * k1)
                k3 = deriv(state + 0.5 * sub_dt * k2)
                k4 = deriv(state + sub_dt * k3)
                state = state + (sub_dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
 
            h_prev = h_cmd
            t = (step + 1) * self.dt
            times.append(t)
            states.append(state.copy())
            headings.append(h_cmd.copy())
 
            radius = np.linalg.norm(state[:3])
            if radius <= self.R_body:
                outcome = "crash"
                break
            if radius >= 2.0 * self.r_SOI:
                outcome = "escape"
                break
 
        return np.array(times), np.array(states), np.array(headings), outcome


"""
Reward terms for the SolarFocuser task.

Every method is a static function taking the vectorized task and returning a
per-env torch tensor of shape (num_envs,). The task multiplies each term by
its scale from cfg.rewards.scales and accumulates per-episode sums for
logging (Episode/rew_<name> in tensorboard/wandb).

Terminal rewards (crash / capture) are applied directly by the task using
cfg.rewards.termination_crash and cfg.rewards.capture_bonus.
"""

import torch


class REWARDS:

    @staticmethod
    def reward_track_position(task):
        """Exponential shaping toward the asteroid.

        ~1 when on top of the capture zone, ->0 a few sigma away.
        sigma = cfg.rewards.position_sigma (m).
        """
        dist = torch.clamp(task.surface_dist, min=0.0)
        sigma = task.cfg.rewards.position_sigma
        return torch.exp(-dist / sigma)

    @staticmethod
    def reward_track_orientation(task):
        """Keep the concentrator's normal pointed at the sun.

        cos(theta - sun_angle): +1 facing the sun, -1 facing away. This is
        what makes the craft a solar *focuser* -- attitude is part of the
        task, and it also maximizes SRP authority.
        """
        return torch.cos(task.sail_theta - task.sun_angle)

    @staticmethod
    def reward_base_velocity(task):
        """Penalize relative velocity w.r.t. the asteroid.

        Negative term: the scale in cfg should be positive; the sign lives
        here so the cfg reads as 'how much do I care'.
        """
        return -torch.norm(task.rel_vel, dim=1)

    @staticmethod
    def reward_action_rate(task):
        """Penalize rapid action changes (thruster chatter)."""
        return -torch.sum(torch.square(task.actions - task.last_actions), dim=1)
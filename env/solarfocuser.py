import copy
import warnings
from typing import Dict, List, Tuple

import Box2D
import gymnasium as gym
import numpy as np
import pygame
from Box2D.b2 import (
    circleShape,
    contactListener,
    edgeShape,
    fixtureDef,
    polygonShape,
    revoluteJointDef,
)
from gymnasium import spaces

from .env_cfg import EnvConfig, State

class SolarFocuser(gym.Env):
    def __init__(self, config: EnvConfig):
        self.config = config
        self.state = None
        self.world = None
        self.solar_panel = None
        self.sun = None
        self.joint = None
        self.viewer = None
        self.window = None
        self.reset()


    def reset(self):
    def step(self, action):
    def render(self):
    def get_mining_position(self):    
    def compute_reward(self, state, action):
    def get_state(self):
    def close(self):
        if self.window is not None:
            self.window.close()
            self.window = None
    def create_world(self):
    def create_solar_focuser(self):
    def create_sun(self):
    def create_asteroid(self):   
"""SolarFocuser package"""
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from solarfocuser.env.focuser_task import SolarFocuser 
from solarfocuser.cfg.focuser_cfg import EnvConfig, TrainConfig, State, UserArgs
from solarfocuser.utils.task_registry import task_registry

task_registry.register(
    name="focuser",                     
    task_class=SolarFocuser,
    env_cfg=EnvConfig(),
    train_cfg=TrainConfig()             
)

__all__ = ["ROOT_DIR", "SolarFocuser", "EnvConfig", "TrainConfig", "State", "UserArgs", "task_registry"]
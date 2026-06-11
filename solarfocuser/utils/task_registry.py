import json
import os
from datetime import datetime

from solarfocuser.utils.helpers import (
    update_env_cfg_from_args, 
    update_train_cfg_from_args, 
    class_to_dict, 
    get_load_path, 
    set_seed
)

from solarfocuser import ROOT_DIR
from solarfocuser.runners.algs.ppo import PPO



class TaskRegistry():
    def __init__(self):
        self.task_classes = {}
        self.env_cfgs = {}
        self.train_cfgs = {}

    def register(self, name: str, task_class, env_cfg, train_cfg):
        self.task_classes[name] = task_class
        self.env_cfgs[name] = env_cfg
        self.train_cfgs[name] = train_cfg

    def get_task_class(self, name: str):
        return self.task_classes[name]

    def get_cfgs(self, name):
        train_cfg = self.train_cfgs[name]
        env_cfg = self.env_cfgs[name]
        return env_cfg, train_cfg

    def make_env(self, name, args=None, env_cfg=None):
        if args is None:
            class DummyArgs: pass
            args = DummyArgs()

        if name in self.task_classes:
            task_class = self.get_task_class(name)
        else:
            raise ValueError(f"Task with name: {name} was not registered")
            
        if env_cfg is None:
            env_cfg, _ = self.get_cfgs(name)
            
        env_cfg = update_env_cfg_from_args(env_cfg, args)
        set_seed(env_cfg.seed)
        headless = getattr(args, 'headless', False)
        render_mode = "human" if getattr(env_cfg, "enable_viewer", True) else None
        
        env = task_class(args=env_cfg, render_mode=render_mode)
        
        return env, env_cfg

    def make_alg_runner(self, env, name=None, args=None, env_cfg=None, train_cfg=None):
        if args is None:
            class DummyArgs: pass
            args = DummyArgs()
            
        create_and_save = False
        if train_cfg is None:
            _, train_cfg = self.get_cfgs(name)
            create_and_save = not getattr(args, 'debug', False)
            
        train_cfg = update_train_cfg_from_args(train_cfg, args)
        
        # Set up logging directories
        log_root = os.path.join(ROOT_DIR, 'solarfocuser/models/', train_cfg.runner.experiment_name)
        log_dir = os.path.join(log_root, datetime.now().strftime('%Y-%m-%d_%H%M%S') + '_' + train_cfg.runner.run_name)

        if create_and_save:
            os.makedirs(log_dir, exist_ok=True)
            env_cfg_dict = class_to_dict(env_cfg)
            train_cfg_dict = class_to_dict(train_cfg)
            
            cfg = {
                "train_cfg": train_cfg_dict,
                "env_cfg": env_cfg_dict
            }

            with open(os.path.join(log_dir, f"{train_cfg.runner.experiment_name}.json"), 'w') as f:
                json.dump(cfg, f, indent=2)

        # Dynamic instantiation based on string value in config
        algorithm = eval(train_cfg.algorithm_name)
        
        # Extract execution device (CPU/GPU) safely from configuration or args
        device = getattr(args, 'device', 'cpu')
        
        runner = algorithm(
            env=env,
            train_cfg=train_cfg,
            log_dir=log_dir,
            device=device
        )

        # Handle checkpoint resuming if requested
        resume = getattr(train_cfg.runner, 'resume', False)
        if resume:
            resume_path = get_load_path(
                log_root, 
                load_run=getattr(train_cfg.runner, 'load_run', -1),
                checkpoint=getattr(train_cfg.runner, 'checkpoint', -1)
            )
            print(f"Loading model from: {resume_path}")
            runner.load(resume_path)
            
        return runner


# Instantiate global task registry
task_registry = TaskRegistry()
from solarfocuser.env.focuser_task import SolarFocuser
from solarfocuser.cfg.focuser_cfg import EnvConfig, TrainConfig
task_registry.register("focuser", SolarFocuser, EnvConfig(), TrainConfig())
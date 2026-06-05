class State:
    def __init__(self, position, velocity):
        self.position = position
        self.velocity = velocity
@dataclass
class EnvConfig:
    state_dim: int
    action_dim: int
    max_episode_steps: int
    reward_threshold: float
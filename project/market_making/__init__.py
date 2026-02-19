from .params import MarketParams
from .simulator import MarketMakingEnv
from .dp_solver import value_iteration, simulate_dp_policy


def __getattr__(name):
    if name == "DQNAgent":
        from .dqn_agent import DQNAgent
        return DQNAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

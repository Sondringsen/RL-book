from .params import MarketParams
from .simulator import MarketMakingEnv
from .dp_solver import (
    value_iteration,
    simulate_dp_policy,
    value_iteration_2d,
    simulate_dp_policy_2d,
    value_iteration_3d,
    simulate_dp_policy_3d,
)


def __getattr__(name):
    if name == "DQNAgent":
        from .dqn_agent import DQNAgent
        return DQNAgent
    if name == "TabularQAgent":
        from .tabular_q import TabularQAgent
        return TabularQAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

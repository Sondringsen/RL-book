# Outline for presentation:
## Frontpage (1 slide)
Title describing the project and our names: Sam, Siddhant, and Sondre

## Overview (~1-2 slides)
Describe what we do intuitively: we use RL to do market making (1-2 slides)

## Method (~7 slides):
Describe in mathematical terms what our base setup looks like. This should describe the MDP in detail with action space, state space, reward function, transition function. Also describe the environment we train in, i.e. fill probability, price evolution, volatility evolution etc. First explain the simple setup in experiment 1, then how we expand on this in experiment 2, then how we expand on this further in experiment 3.

## Results
One slide about general results:
- It's hard to learn for the RL algorithm due to low signal to noise ratio induced by volatility in the price process. Since the different Q-values are quite close in absolute value, this makes it hard for RL to differentiate between the states. This can also be seen from the PnL plots in that suboptimal policies achieves almost as good PnL as the optimal policy
- High inventory (incentivizes selling and not buying) -> low ask spread, high bid spread
- Low inventory (incetiveses buying and not selling) -> high ask spread, low bid spread 
- Lower price leads to lower absolute value of spreads as price movements are proportional to price
- High volatility leads to being more conservative in setting spreads as there is more risk for the market maker

Just insert a few placeholder slides for plots.

## Conclusion
One slide that summarizes the 3 experiments and 
One slide about questions 

## General Instructions
Emphasize the difference between RL and MDP. For instance in the second and third experiments we discretize the price to make it work for MDP. A point of this project is to see when MDPs fail.

# Goal  

- A trader needs to decide how many options on SPY and XLE to buy/sell for different strikes  

- All options maturities is assumed to be the same, say 1 week, or 1 day, and such period represents the length of the time step

- The goal is to replicate or maybe even outperform a benchmark composed of SPY (denoted by $X$) and XLE (denoted by $Y$)

- Specifically, we want to replicate with options on $X$ and $Y$ the payoff $\xi(X,Y) = \frac{n_XX}{n_XX+n_YY}X+\frac{n_YY}{n_XX+n_YY}Y$ where $n_X$ and $n_Y$ are the numbers of shares outstanding for $X$ and $Y$ respectively

- The idea is that when inflation is high, XLE provides a good hedge for it, while if inflation is low, one does not want to miss growth in the stock market and so one trades SPY

- Short selling options is allowed, but there is no risk free asset

# Basic example of gymnasium environment  

`import gymnasium as gym`
`env = gym.make("LunarLander-v2", render_mode="human")`
`observation, info = env.reset(seed=42)`
`for n in range(1000):`
   `action = env.action_space.sample()  # this is where you would insert your policy`
   `observation, reward, terminated, truncated, info = env.step(action)`
   `if terminated or truncated:`
         `observation, info = env.reset()`
`env.close()`

# Access site-packages location address

`from distutils.sysconfig import get_python_lib`
`print(get_python_lib())`

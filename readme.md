
# Optimal Derivative positioning via Hierarchical Reinforcement Learning

- We consider a continuous time application of Financial Finance, according to which:
  - Investors preferences are represented by a monetary utility function, that is:
    - They maximize the worst case scenario expected future cash flow
    - Scenarios are defined by distorting a probability measure $\mathbb{P}$ that represents the base case scenario, and is estimated from historical observations
    - Rebates are offered for less likely, but still not impossible scenarios
    - The idea is that, at least in the short term, the set of scenarios may substantially deviate from $\mathbb{P}$, but not too much, so a decision can still be made
  - There is no budget constraint, as short selling is allowed.
  - Risk neutral and statistical measure are not equivalent - arbitrages may exist when trading continuously, which is however physically impossible

- In this experiment, the goal is to *outperform* a benchmark composed of SPY and XLE using options on SPY/XLE
  - All options maturities is assumed to be the same, say 1 week, or 1 day, and such period represents the length of the time step
  - The overall timeframe is fixed to a period of 6 months
  - The benchmark is defined by $\xi(X,Y) = \frac{n_XX}{n_XX+n_YY}X+\frac{n_YY}{n_XX+n_YY}Y$ where $n_X$ and $n_Y$ are the numbers of shares outstanding for $X$ and $Y$ respectively
    - The idea is that when inflation is high, XLE provides a good hedge for it, while if inflation is low, one does not want to miss growth in the stock market and so one trades SPY

# Training a single agent

- A simulated environment is first considered:
  - XLE and SPY are (multivariate) GBM or BG processes with fixed parameters
  - A PPO agent maximizing an MUF is trained within a Gym environment
  - The PPO algorithm is based on `stable-baselines3` implementation
  - Reward shaping and different actor/critic policies are tested
  
- Succefully training such an agent is not new, for instance, one may consult
  - Jaimungal et al., *Robust Risk Aware Reinforcement Learning*, Siam Journal of Financial Mathematics, Vol 13, Issue 1, 2022
  - Chopra, *Reinforcement Learning Methods for Conic Finance*, PhD Thesis, 2020
  - However, agents of this kind are the building blocks of what comes next

# The Hierarchy of Agents

- In the literature, results on the performance of Reinforcement Learning agents trained on real data is not entirely satisfactory (see Hirsa et al. *Deep Reinforcement learning on a multi-asset environment for trading*, arxiv, 2021)
  - Of course, one main issue when we move to real data is that, even in the medium term, the base case measure $\mathbb{P}$ may be far from today's
  - This requires training on very large datasets, which is not feasible
  
- To overcome this issue, we proceed as follows:
  - We construct clusters of histortically observed $\mathbb{P}$, and train a team of lower level agents, each of which specialized in one such $\mathbb{P}$
  - A higher level agent then decides which mixture of low level agents should be used at each time step

- Some open questions:
  - We need to check the literature on HRL - obviously applications of HRL to inflation risk and Financial Finance in particular are yet to be explored, but we want to make sure we are not doing exactly the same thing as someone else
  - What should be the observation space of the higher level agent, considering our task of taming inflation risk?
    - Perhaps options bid-ask spreads, which have shown predicting power in anticipating Federal Reserve moves:
    - see Shirai, *Extreme measures in continuous time conic finance*, Frontiers of Mathematical Finance, Vol 3 Number 1 2024

# Info about this repo

- The folder ``BenchEnv`` contains the conda environment needed to run files in this repo

- To check imported libraries, you may run the following code

```bash
from distutils.sysconfig import get_python_lib
print(get_python_lib())
```

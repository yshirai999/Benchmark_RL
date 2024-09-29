
# Optimal Derivative positioning via Reinforcement Learning

- The goal is to maximize the value, in 10 days from now, of a portfolio of options written on SPY with maturity 15 days
- The only relevant measures are thus the 15-day risk neutral measures at which options are bought today, and the 5-day risk neutral measure in 10-day from now at which options are sold
- The advantage of this approach is that we do not need to estimate anything from time series of data, the only thing we look at are the options implied and forward looking risk neutral measures

- In reality, we should train our agent on risk neutral measures obtained from daily calibration to options vol surface
- We do not have that data, so in this experiment we assume risk neutral parameters follow a Markov chain with given transition matrix P
- To ensure good fitting to the options surface, the log price of SPY at 15 days is assumed to be the difference of two gamma variates with parameters (bp,cp) and (bn,cn)
- Similarly for the 5-day measure
  
- A PPO agent maximizing return from the sale of the option portfolio can then be trained within a Gym environment
- The PPO algorithm is based on `stable-baselines3` implementation
- Reward shaping and different actor/critic policies can be tested

- The question is then essentially whether the agent can be trained to learn the transition matrix P
- This question is new, I think. Other works on RL in portfolio construction do not focus on options trading, nor on risk neutral measures. For instance, one can look at:
  - Jaimungal et al., *Robust Risk Aware Reinforcement Learning*, Siam Journal of Financial Mathematics, Vol 13, Issue 1, 2022
  - Chopra, *Reinforcement Learning Methods for Conic Finance*, PhD Thesis, 2020
  - Hirsa et al. *Deep Reinforcement learning on a multi-asset environment for trading*, arxiv, 2021
    - This last work is particularly interesting as it shows that performance of Reinforcement Learning agents trained on real data is not satisfactory due to:
      - nonstationarity of time series
      - lack of data

# Info about this repo

- The folder ``BenchEnv`` contains the conda environment needed to run files in this repo

- To check imported libraries, you may run the following code

```bash
from distutils.sysconfig import get_python_lib
print(get_python_lib())
```

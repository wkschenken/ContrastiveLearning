import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.pyplot import imshow
import random
import cmath
from scipy import signal
from scipy.optimize import curve_fit
from matplotlib import cm
from matplotlib.colors import ListedColormap,LinearSegmentedColormap
from matplotlib import colors
from scipy.stats import norm
from scipy.stats import cauchy
import matplotlib.mlab as mlab
from scipy import stats


# arguments for the normal distribution go as (mean, std, size of array)
# N-bandit problem, trying the examples in Sutton and Bartow

N=20

NumEps = 20 # number of epsilons to compare

eps = np.logspace(-3, -0.1, NumEps)
EpsFinal = np.zeros(NumEps)
EpsFinalPerStep = np.zeros(NumEps)
MSEEps = np.zeros(NumEps) # mean squared error in estimating the actual mean of the bandits
choices = np.linspace(0, N-1, N).astype(int) # options for bandits when choosing randomly


# go through the epsilon-greedy search algorithm MM times
MM=2*10**3

NumEpsAvgs = 2*10**3 # number of different bandit pdfs to average over for a given epsilon

a = 0
figure, axis = plt.subplots(1, 2)

for epsilon in eps:
    print("%5.2f percent finished" % (100*a/NumEps))

    RewardEps = np.zeros(MM) # final average of the rewards for a given epsilon
    RewardPerStepEps = np.zeros(MM) # final average of the reward per step for a given epsilon

    # number of times to run, for a given distribution of bandits at a given epsilon
    for nn in range(0, NumEpsAvgs):

        reward = np.zeros(MM) # integrated reward for that run
        RwdPerStep = np.zeros(MM) # average reward per step for that run

        # define pdfs of the bandits
        means = np.random.normal(loc = 0.0, scale = 1.0, size = N)
        stdevs = np.random.normal(loc = 0.0, scale = 1.0, size = N)
        Q = np.zeros(N) #array of Q values for each bandit, updated at each step
        nQ = np.zeros(N) # number of times each option has been chosen

        # run MM choices for each distribution of bandits, at the given epsilon
        for ii in range(0, MM):

            rwd = 0 # reward after each decision
            num = random.random() # random number determines choice of greedy or not

            # with probability 1-epsilon, pick the option with the current highest Q, indexed by qq
            if num>epsilon:
                # find the index with the current highest Q, given by qq
                qq=0
                Qmax = 0
                for jj in range(0, N-1):
                    if Q[jj]>Qmax:
                        Qmax = Q[jj]
                        qq=jj
                nQ[qq]+=1 # update the number of times this option has been chosen
                rwd = np.random.normal(loc = means[qq], scale = np.abs(stdevs[qq])) # greedy reward for that choice
                reward[ii] = reward[ii-1] + rwd
                RwdPerStep[ii] = (1-1/(ii+1))*RwdPerStep[ii-1] + rwd/(ii+1)
                Q[qq] = (1-1/nQ[qq])*Q[qq] + rwd/nQ[qq] # update the new Q value
            else:
                p = np.int32(random.choice(choices))
                nQ[p]+=1
                rwd = np.random.normal(loc = means[p], scale = np.abs(stdevs[p]))
                reward[ii] = reward[ii-1] + rwd
                RwdPerStep[ii] = (1-1/(ii+1))*RwdPerStep[ii-1] + rwd/(ii+1)
                Q[p] = (1-1/nQ[p])*Q[p] + rwd/nQ[p]

        RewardEps = RewardEps + reward/NumEpsAvgs
        RewardPerStepEps = RewardPerStepEps + RwdPerStep/NumEpsAvgs

        error = means - Q
        for ii in range(0, N-1):
            MSEEps[a] = MSEEps[a] + error[ii]**2/NumEpsAvgs




    fig1 = plt.figure(1)
    axis[0].plot(RewardEps, label = "Epsilon = {EpsilonVal:.3f}".format(EpsilonVal = epsilon))
    axis[0].set_xlabel("Choice Number")
    axis[0].set_ylabel("Integrated Reward")

    axis[1].plot(RewardPerStepEps, label = "Epsilon = {EpsilonVal:.3f}".format(EpsilonVal = epsilon))
    axis[1].set_xlabel("Choice Number")
    axis[1].set_ylabel("Reward Per Step")
    EpsFinal[a] = RewardEps[MM-1]
    EpsFinalPerStep[a] = RewardPerStepEps[MM-1]
    a +=1
plt.legend()

fig2 = plt.figure(2)
fig, ax = plt.subplots()
ax2 = ax.twinx()
ax.semilogx(eps, EpsFinal, color = 'k')
ax2.semilogx(eps, EpsFinalPerStep, color = 'k')

ax.set_xlabel('Epsilon')
ax.set_ylabel('Final Integrated Reward', color = 'k')
ax2.set_ylabel('Final Reward Per Step', color = 'k')

# defining display layout
plt.tight_layout()

fig3 = plt.figure(4)
plt.loglog(eps, MSEEps)
plt.xlabel("Epsilon")
plt.ylabel("Mean Squared Error (Q, q)")


plt.show()

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
import matplotlib.pyplot as plt
import pandas as pd

# define number of Nodes - beginning with a Nr x Nc grid of Nodes in 2D
N_dim = 2

Nr = 4
Nc = Nr
N_Nodes = Nr**2

NodePos = np.zeros((N_Nodes, N_dim)) # physical degrees of freedom - the positions of the Nodes in the network
edges = np.zeros((N_Nodes, N_Nodes)) # symmetric matrix with zeros on diagonal; =1 if Node i (row i) is connected to Node j (column j), =0 otherwise
                                        # should this be antisymmetric when constructing the Hessian?
# SpringConst = np.zeros(N_edges) # learning degrees of freedom - the spring constants of the edges connecting two Nodes

# Initialize the positions of the Nodes, defining the equilibrium position

for ii in range(Nr):
    for jj in range(Nc):
        NodePos[Nr*ii + jj, 0] = ii
        NodePos[Nr*ii + jj, 1] = jj

# Visualize the Node positions
# plt.figure()
# for ii in range(N_Nodes):
#     plt.scatter(NodePos[ii, 0], NodePos[ii, 1], c = 'k')
# plt.show()

# Begin by defining a simple square lattice with nearest neighbors connected
# Use this as a starting point to get the contrastive learning algorithm to work

# define the incidence matrix Delta as described in the contrastive learning literature from Liu and Durian

N_edges = (Nr-1)*Nc + (Nc-1)*Nr + 2*(Nr-1)*(Nc-1) # number of edges for a square graph + diagonals
# N_edges = (Nr-1)*Nc + (Nc-1)*Nr# number of edges for a square graph
Delta = np.zeros((N_edges, N_Nodes))

kk=0 # index to count edges

# Horizontal
for ii in range(Nr-1):
    for jj in range(Nc):
        Delta[kk, Nr*ii + jj] = 1
        Delta[kk, Nr*(ii+1) + jj] = -1
        kk+=1

# vertical
for jj in range(Nc-1):
    for ii in range(Nr):
        Delta[kk, Nr*ii + jj] = 1
        Delta[kk, Nr*ii + jj + 1] = -1
        kk+=1

# diagonal, up and to the right
for ii in range(Nr-1):
    for jj in range(Nc-1):
        Delta[kk, Nr*ii + jj] = 1
        Delta[kk, Nr*(ii+1) + jj + 1] = -1
        kk+=1

# diagonal, up and to the left
for ii in range(Nr-1):
    for jj in range(1, Nc):
        Delta[kk, Nr*ii + jj] = 1
        Delta[kk, Nr*(ii+1) + jj - 1] = -1
        kk+=1

# visualize the incidence matrix
# plt.figure()
# plt.matshow(Delta)
# plt.title("Incidence Matrix")

# plt.show()

# define a diagonal matrix with random weights; here, for random results each time.
# w = np.zeros(N_edges)
# for vv in range(N_edges):
#     w[vv] = 1+np.random.rand()

# generate random floating point values
from numpy.random import seed
from numpy.random import rand
# seed random number generator for repeatable results
seed(10)
wVec = rand(N_edges) + 1/2
w = np.diag(wVec)

# visualize the graph with edges
# plt.figure()
# for nn in range(N_edges):
#     i1 = np.where(Delta[nn, :]==1)
#     i2 = np.where(Delta[nn, :]==-1)
#     i1 = i1[0]
#     i2 = i2[0]
#     point1 = [NodePos[i1, 0], NodePos[i1, 1]]
#     point2 = [NodePos[i2, 0], NodePos[i2, 1]]
#     x_values = [point1[0], point2[0]]
#     y_values = [point1[1], point2[1]]
#     plt.plot(x_values, y_values, 'ko', linestyle = '-', linewidth = w[nn, nn])
# plt.title("Randomly initialized weights on a square graph")
# plt.show()

# calculate the Hessian from Delta and w
Hessian = np.matmul(np.transpose(Delta), np.matmul(w, Delta))

# visualize the Hessian
plt.figure()
plt.matshow(Hessian)
plt.title("Initial Hessian Matrix")
# plt.show()

# calculate the equilibrium voltages for an input voltage on the left hand side, with the bottom left designated as ground
# do this by a brute force gradient descent rather than by Kirchoff's laws, but not updating the input nodes

# This defines the "free network", to be used to calculate the voltage applied to the clamped network

epsilon = 1e-2 # learning rate for the energy gradient descent
eta = 0.1 # nudge parameter - nudges the clamped state output voltage from that of the free state towards the desired output
alpha = 0.1 # learning rate for the edge update rule

# inputs; arbitrary
Vg = 0
V1 = 7
V2 = 10

# Input indices
VgInd = 0
V1Ind = np.int64(Nr/2)*Nc
V2Ind = (Nr-1)*Nc
InputIndices = [Vg, V1Ind, V2Ind]
Inputs = [Vg, V1, V2]

# Desired outputs
VO1 = 5
VO2 = 3
VO3 = 2

# Output indices
VO1Ind = Nr*Nc - 1
VO2Ind = np.int64(Nr/2 + 1)*Nc - 1
VO3Ind = Nc - 1
OutputIndices = [VO1Ind, VO2Ind, VO3Ind]
Outputs = [VO1, VO2, VO3]

RelaxationIterations = 10**4 # how many iterations in gradient descent to minimize the energy of the network for a new input voltage and new weights
LearningIterations = 100 # iterations in the contrastive learning scheme; how many times to update the edges


# -------------------------------------------------------------------------------------------------------------------------------------#
# -------------------------------------------------------------------------------------------------------------------------------------#
# -------------------------------------------------------------------------------------------------------------------------------------#
# This section shows the operation of a single iteration of contrastive learning, to be put in a for loop at the end of the code
# Comment this section out when running the learning algorithm
# Used for construction, for trial and error, and to check that the energy minimization is working as expected
# -------------------------------------------------------------------------------------------------------------------------------------#
# -------------------------------------------------------------------------------------------------------------------------------------#
# -------------------------------------------------------------------------------------------------------------------------------------#

# NodeValsFree = np.zeros(N_Nodes) # array to hold e.g. node voltages in a resistor network
# NodeValsClamped = np.zeros(N_Nodes) # array to hold e.g. node voltages in a resistor network
#
# NodeValsFree[0] = Vg
# NodeValsFree[Nc] = V1
# NodeValsFree[2*Nc] = V2
#
# energyFree = np.zeros(RelaxationIterations)
# energyFree[0] = np.matmul(np.transpose(NodeValsFree), np.matmul(Hessian, NodeValsFree))
#
# for ii in range(1, RelaxationIterations):
#     # update voltages
#     NodeValsFree = NodeValsFree - epsilon*np.matmul(Hessian, NodeValsFree)
#     # reset clamped voltages
#     NodeValsFree[0] = Vg
#     NodeValsFree[Nc] = V1
#     NodeValsFree[2*Nc] = V2
#     # recalculate the energy
#     energyFree[ii] = np.matmul(np.transpose(NodeValsFree), np.matmul(Hessian, NodeValsFree))
#
# plt.figure()
# plt.plot(energyFree)
# plt.title("Energy minimization for free state")
# plt.yscale("log")
# plt.xscale("log")
# plt.ylabel("Energy (a.u.)")
# plt.xlabel("Iteration")
# # plt.show()
#
#
#
# # visualize the graph with edges at its equilibrium voltage
# plt.figure()
# for nn in range(N_edges):
#     i1 = np.where(Delta[nn, :]==1)
#     i2 = np.where(Delta[nn, :]==-1)
#     i1 = i1[0]
#     i2 = i2[0]
#     point1 = [NodePos[i1, 0], NodePos[i1, 1]]
#     point2 = [NodePos[i2, 0], NodePos[i2, 1]]
#     x_values = [point1[0], point2[0]]
#     y_values = [point1[1], point2[1]]
#     plt.plot(x_values, y_values, 'k', linestyle = '-', linewidth = w[nn, nn])
# for ii in range(N_Nodes):
#     plt.scatter(NodePos[ii, 0], NodePos[ii, 1], 100*NodeValsFree[ii], c = 'k')
#
# plt.title("Graph with markersize $\propto$ node voltages after optimization in free network")
#
# # plt.show()
#
# print(NodeValsFree)
# print(energyFree[-1])
#
#
#
# # Repeat the same thing with the clamped network, designating the upper two nodes as the output nodes
#
# NodeValsClamped[0] = Vg
# NodeValsClamped[Nc] = V1
# NodeValsClamped[2*Nc] = V2
#
# NodeValsClamped[Nr + Nc - 1] = eta*VO1 + (1-eta)*NodeValsFree[Nr + Nc - 1]
# NodeValsClamped[2*Nr + Nc - 1] = eta*VO2 + (1-eta)*NodeValsFree[2*Nr + Nc - 1]
#
# energyClamped = np.zeros(RelaxationIterations)
# energyClamped[0] = np.matmul(np.transpose(NodeValsClamped), np.matmul(Hessian, NodeValsFree))
#
# for ii in range(1, RelaxationIterations):
#     # update voltages
#     NodeValsClamped = NodeValsClamped - epsilon*np.matmul(Hessian, NodeValsClamped)
#     # reset clamped voltages
#     NodeValsClamped[0] = Vg
#     NodeValsClamped[Nc] = V1
#     NodeValsClamped[2*Nc] = V2
#     NodeValsClamped[Nr + Nc - 1] = eta*VO1 + (1-eta)*NodeValsFree[Nr + Nc - 1]
#     NodeValsClamped[2*Nr + Nc - 1] = eta*VO2 + (1-eta)*NodeValsFree[2*Nr + Nc - 1]
#     # recalculate the energy
#     energyClamped[ii] = np.matmul(np.transpose(NodeValsClamped), np.matmul(Hessian, NodeValsClamped))
#
# plt.figure()
# plt.plot(energyClamped)
# plt.title("Energy minimization for clamped state")
# plt.yscale("log")
# plt.xscale("log")
# plt.ylabel("Energy (a.u.)")
# plt.xlabel("Iteration")
#
# # visualize the graph with edges at its equilibrium voltage
# plt.figure()
# for nn in range(N_edges):
#     i1 = np.where(Delta[nn, :]==1)
#     i2 = np.where(Delta[nn, :]==-1)
#     i1 = i1[0]
#     i2 = i2[0]
#     point1 = [NodePos[i1, 0], NodePos[i1, 1]]
#     point2 = [NodePos[i2, 0], NodePos[i2, 1]]
#     x_values = [point1[0], point2[0]]
#     y_values = [point1[1], point2[1]]
#     plt.plot(x_values, y_values, 'k', linestyle = '-', linewidth = w[nn, nn])
# for ii in range(N_Nodes):
#     plt.scatter(NodePos[ii, 0], NodePos[ii, 1], 100*NodeValsClamped[ii], c = 'k')
#
# plt.title("Graph with markersize $\propto$ node voltages after optimization in clamped network")
#
# plt.show()
#
# print(NodeValsClamped)
# print(energyClamped[-1])



# -------------------------------------------------------------------------------------------------------------------------------------#
# -------------------------------------------------------------------------------------------------------------------------------------#
# -------------------------------------------------------------------------------------------------------------------------------------#



# In the for loop, we must add the change in weights after calculating the equilibrium voltages of all nodes in each network

# Define node input and output voltages
NodeValsFree = np.zeros(N_Nodes) # array to hold node voltages in the free state
NodeValsFreeInit = np.zeros(N_Nodes) # array to hold node voltages in the free state at the starting point
NodeValsClamped = np.zeros(N_Nodes) # array to hold node voltages in the clamped state

for vii in InputIndices:
    NodeValsFree[vii] = Inputs[list(InputIndices).index(vii)]
    NodeValsClamped[vii] = Inputs[list(InputIndices).index(vii)]

# Follow the error in the learning process
err_RMS = np.zeros(LearningIterations)

# make sure energy is minimized to the same order of magnitude after each update
e_Free = np.zeros(LearningIterations)
e_Clamped = np.zeros(LearningIterations)

# note the starting point for the edges
wInit = w
wVecInit = wVec

# Follow the RMS change in edge conductances at each step
# *** change this to a "while error>threshold" statement ***
dw_RMS = np.zeros(LearningIterations)

# First calculate the equilibrium of the free state
for m in range(LearningIterations):
    print(m/LearningIterations)
    for ii in range(1, RelaxationIterations):
        # update voltages
        NodeValsFree = NodeValsFree - epsilon*np.matmul(Hessian, NodeValsFree)
        # reset inputs
        for vii in InputIndices:
            NodeValsFree[vii] = Inputs[list(InputIndices).index(vii)]

    # Note the initial equilibrium voltages
    if m==0:
        NodeValsFreeInit = NodeValsFree

    # Note the energy of the free state at the end of the relaxation
    e_Free[m] = np.matmul(np.transpose(NodeValsFree), np.matmul(Hessian, NodeValsFree))/2

    # Repeat the same thing with the clamped network, using the outputs of the free network, designating the upper two nodes as the output nodes
    for voi in OutputIndices:
        NodeValsClamped[voi] = eta*Outputs[list(OutputIndices).index(voi)] + (1-eta)*NodeValsFree[voi]

    for ii in range(1, RelaxationIterations):
        # update voltages
        NodeValsClamped = NodeValsClamped - epsilon*np.matmul(Hessian, NodeValsClamped)
        # reset clamped voltages
        for vii in InputIndices:
            NodeValsClamped[vii] = Inputs[list(InputIndices).index(vii)]
        for voi in OutputIndices:
            NodeValsClamped[voi] = eta*Outputs[list(OutputIndices).index(voi)] + (1-eta)*NodeValsFree[voi]

    # Note the energy of the free state at the end of the relaxation
    e_Clamped[m] = np.matmul(np.transpose(NodeValsClamped), np.matmul(Hessian, NodeValsClamped))/2

    # Calculate \Delta V ^2 across all edges in each network and use that in the contrastive learning rule to update the edges w
    dV2F = np.square(np.matmul(Delta, NodeValsFree))
    dV2C = np.square(np.matmul(Delta, NodeValsClamped))

    wVec = wVec - (1/2)*(alpha/eta)*(dV2C - dV2F)
    w = np.diag(wVec)

    # recalculate the Hessian
    Hessian = np.matmul(np.transpose(Delta), np.matmul(w, Delta))

    # note the RMS error between the output nodes and the desired outputs
    err_RMS[m] = (1/2)*np.sqrt((NodeValsFree[VO1Ind] - VO1)**2 + (NodeValsFree[VO2Ind] - VO2)**2)

    # note the RMS change in conductances at this step
    dw_RMS[m] = (alpha/eta)*np.sqrt(np.dot((dV2C - dV2F), (dV2C - dV2F)))/(2*N_edges)


# Compare initial Hessian to the final Hessian matrix
plt.figure()
plt.matshow(Hessian)
plt.title("Final Hessian matrix")

# Plot error vs iteration number
plt.figure()

LearningIterationsArray = np.linspace(0, LearningIterations, LearningIterations)
plt.plot(LearningIterationsArray, err_RMS, label = "Numerical error")

# Fit the latter half of the error curve to an exponential decay
err_RMS_2ndHalf = err_RMS[np.int64(0.7*LearningIterations)-1:LearningIterations]
minErr = np.min(err_RMS_2ndHalf)
err_RMS_2ndHalf = err_RMS[np.int64(0.7*LearningIterations)-1:LearningIterations]/minErr
iterations_2ndHalf = np.linspace(np.int64(0.7*LearningIterations), LearningIterations, np.int64(LearningIterations - np.int64(0.7*LearningIterations))+1)
def ExpDecay(it, T, A):
    return A*np.exp(-(it/T))
pars_err_RMS, cov_err_RMS = curve_fit(ExpDecay, iterations_2ndHalf, err_RMS_2ndHalf, p0=[50, 3e-1], bounds=(0, np.inf))
T_fit = np.round(pars_err_RMS[0],decimals = 3)
A_fit = np.round(pars_err_RMS[1],decimals = 3)
err_RMS_fit = A_fit*np.exp(-LearningIterationsArray/T_fit)*minErr
plt.plot(LearningIterationsArray, err_RMS_fit, linestyle='--', linewidth=2, color='blue',label='Decay constant = {}'.format(T_fit))
plt.title("Error in output voltage vs iterations number")
plt.ylabel("RMS Error [V]")
plt.xlabel("Iteration")
plt.yscale("log")
plt.legend()

# Plot energy after relaxation at each iteration
plt.figure()
plt.plot(e_Clamped, label = "Final energy in the free state at this iteration")
plt.plot(e_Clamped, label = "Final energy in the clamped state at this iteration")
plt.title("Energy in the circuit as it updates")
plt.ylabel("Energy (a.u.)")
plt.xlabel("Iteration")
plt.legend()

# Plot RMS change in conductances after each iteration
plt.figure()
plt.plot(dw_RMS)
plt.title("RMS change in conductances at each iteration")
plt.ylabel("$\sqrt{<dw^2>}$")
plt.xlabel("Iteration")
plt.yscale("log")


# visualize the graph with edges at its equilibrium voltage before optimization
plt.figure()
for nn in range(N_edges):
    i1 = np.where(Delta[nn, :]==1)
    i2 = np.where(Delta[nn, :]==-1)
    i1 = i1[0]
    i2 = i2[0]
    point1 = [NodePos[i1, 0], NodePos[i1, 1]]
    point2 = [NodePos[i2, 0], NodePos[i2, 1]]
    x_values = [point1[0], point2[0]]
    y_values = [point1[1], point2[1]]
    plt.plot(x_values, y_values, 'k', linestyle = '-', linewidth = wInit[nn, nn])
for ii in range(N_Nodes):
    plt.scatter(NodePos[ii, 0], NodePos[ii, 1], 100*NodeValsFreeInit[ii], c = 'k', alpha = 0.5)
    if ii in InputIndices:
        plt.scatter(NodePos[ii, 0], NodePos[ii, 1], 20, c = 'b')
        plt.scatter(NodePos[ii, 0], NodePos[ii, 1], 100*Inputs[list(InputIndices).index(ii)], c = 'b', alpha = 0.2)
    if ii in OutputIndices:
        plt.scatter(NodePos[ii, 0], NodePos[ii, 1], 20, c = 'r')
        plt.scatter(NodePos[ii, 0], NodePos[ii, 1], 100*Outputs[list(OutputIndices).index(ii)], c = 'r', alpha = 0.2)

plt.suptitle("Graph with markersize $\propto$ node voltages before optimization")
plt.title("Blue = Inputs, Red = Outputs")


# visualize the final graph with edges at its equilibrium voltage after optimization
plt.figure()
for nn in range(N_edges):
    i1 = np.where(Delta[nn, :]==1)
    i2 = np.where(Delta[nn, :]==-1)
    i1 = i1[0]
    i2 = i2[0]
    point1 = [NodePos[i1, 0], NodePos[i1, 1]]
    point2 = [NodePos[i2, 0], NodePos[i2, 1]]
    x_values = [point1[0], point2[0]]
    y_values = [point1[1], point2[1]]
    plt.plot(x_values, y_values, 'k', linestyle = '-', linewidth = w[nn, nn])
for ii in range(N_Nodes):
    plt.scatter(NodePos[ii, 0], NodePos[ii, 1], 100*NodeValsFree[ii], c = 'k', alpha = 0.5)
    if ii in InputIndices:
        plt.scatter(NodePos[ii, 0], NodePos[ii, 1], 20, c = 'b')
        plt.scatter(NodePos[ii, 0], NodePos[ii, 1], 100*Inputs[list(InputIndices).index(ii)], c = 'b', alpha = 0.2)
    if ii in OutputIndices:
        plt.scatter(NodePos[ii, 0], NodePos[ii, 1], 20, c = 'r')
        plt.scatter(NodePos[ii, 0], NodePos[ii, 1], 100*Outputs[list(OutputIndices).index(ii)], c = 'r', alpha = 0.2)
plt.suptitle("Graph with markersize $\propto$ node voltages after optimization")
plt.title("Blue = Inputs, Red = Outputs")

print(NodeValsFreeInit)
print(NodeValsFree)

print(np.sqrt(np.dot(wVec, wVec)))



plt.show()

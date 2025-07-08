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
import networkx as nx

# ---------------------------------------------------------------------------------------------------------------------------
# Defining a graph by hand --- later, using NetworkX to define graphs more simply and exporting the incidence matrix
# ---------------------------------------------------------------------------------------------------------------------------
#
# # define number of Nodes - beginning with a Nr x Nc grid of Nodes in 2D
# N_dim = 2
#
# Nr = 5
# Nc = Nr
# N_Nodes = Nr**2
#
# NodePos = np.zeros((N_Nodes, N_dim)) # physical degrees of freedom - the positions of the Nodes in the network
#
# # Initialize the positions of the Nodes
# # add a small random displacement so that it's easier to distinguish edges that are connected to vs crossing a given node
# from numpy.random import seed
# from numpy.random import rand
# from numpy.random import randn
# # seed random number generator for repeatable results
# # seed(1)
# for ii in range(Nr):
#     for jj in range(Nc):
#         NodePos[Nr*ii + jj, 0] = ii + 0.05*randn()
#         NodePos[Nr*ii + jj, 1] = jj + 0.05*randn()
#
#
# # define the incidence matrix Delta
# # = +1 or -1 for incoming/outgoing edges (but the graph here is undirected, so labels are arbitrary)
#
# #N_edges = (Nr-1)*Nc + (Nc-1)*Nr + (Nc-1)*(Nr-1) # number of edges for a square graph + one-way diagonals
# #N_edges = (Nr-1)*Nc + (Nc-1)*Nr + 2*(Nc-1)*(Nr-1) # number of edges for a square graph + both diagonals
# # N_edges = (Nr-1)*Nc + (Nc-1)*Nr# number of edges for a square graph
# N_edges = (Nr-1)*Nc + (Nc-1)*Nr + 2*(Nc-1)*(Nr-1) + 3 # number of edges for a square graph + both diagonals + a few random connections
#
# Delta = np.zeros((N_edges, N_Nodes))
# DeltaT = np.zeros((N_edges, N_Nodes))
# DeltaGmT = np.zeros((N_edges, N_Nodes))
#
# kk=0 # index to count edges
#
# # Horizontal
# for ii in range(Nr-1):
#     for jj in range(Nc):
#         Delta[kk, Nr*ii + jj] = 1
#         Delta[kk, Nr*(ii+1) + jj] = -1
#         DeltaGmT[kk, Nr*ii + jj] = 1
#         DeltaGmT[kk, Nr*(ii+1) + jj] = -1
#         kk+=1
#
# # vertical
# for jj in range(Nc-1):
#     for ii in range(Nr):
#         Delta[kk, Nr*ii + jj] = 1
#         Delta[kk, Nr*ii + jj + 1] = -1
#         DeltaGmT[kk, Nr*ii + jj] = 1
#         DeltaGmT[kk, Nr*ii + jj + 1] = -1
#         kk+=1
#
# # diagonal, up and to the right
# for ii in range(Nr-1):
#     for jj in range(Nc-1):
#         Delta[kk, Nr*ii + jj] = 1
#         Delta[kk, Nr*(ii+1) + jj + 1] = -1
#         DeltaGmT[kk, Nr*ii + jj] = 1
#         DeltaGmT[kk, Nr*(ii+1) + jj + 1] = -1
#         kk+=1
#
# # diagonal, up and to the left
# for ii in range(Nr-1):
#     for jj in range(1, Nc):
#         Delta[kk, Nr*ii + jj] = 1
#         Delta[kk, Nr*(ii+1) + jj - 1] = -1
#         DeltaGmT[kk, Nr*ii + jj] = 1
#         DeltaGmT[kk, Nr*(ii+1) + jj - 1] = -1
#         kk+=1
#
# Delta[kk, 0] = 1
# Delta[kk, 2*Nr-1] = -1
# DeltaGmT[kk, 0] = 1
# DeltaGmT[kk, 2*Nr-1] = -1
# kk+=1
# Delta[kk, 1] = 1
# Delta[kk, 2*Nr-1] = -1
# DeltaGmT[kk, 1] = 1
# DeltaGmT[kk, 2*Nr-1] = -1
# kk+=1
# Delta[kk, 2] = 1
# Delta[kk, 2*Nr-1] = -1
# DeltaGmT[kk, 2] = 1
# DeltaGmT[kk, 2*Nr-1] = -1
#
# # The weights of the edges will be unimportant for the initial tests
# # initialize all edges to have an arbitrary weight = 1
# wVec = np.ones(N_edges)
# w = np.diag(wVec)



# ---------------------------------------------------------------------------------------------------------------------------
# Define a graph using NetworkX and export the incidence matrix
# ---------------------------------------------------------------------------------------------------------------------------

Nr = 5
Nc = 5
N_Nodes = Nr*Nc
p = .1

check = 0
while check == 0:
    G = nx.erdos_renyi_graph(N_Nodes, p)
    if np.size(list(nx.connected_components(G))) == 1:
        check = 1
    print("Generating a connected ER graph...")

N_edges = G.number_of_edges()

N_dim = 2

NodePos = np.zeros((N_Nodes, N_dim)) # physical degrees of freedom - the positions of the Nodes in the network




from numpy.random import seed
from numpy.random import rand
from numpy.random import randn

for ii in range(Nr):
    for jj in range(Nc):
        NodePos[Nr*ii + jj, 0] = ii + 0.05*randn()
        NodePos[Nr*ii + jj, 1] = jj + 0.05*randn()

wVec = np.ones(N_edges)
w = np.diag(wVec)

# Incidence matrix in the form rows = edges, columns = vertices to match the above format
def IM_From_G(G):
    Delta_FromNX = nx.incidence_matrix(G, oriented = True)
    Delta = np.transpose(Delta_FromNX.toarray())
    return Delta, Delta_FromNX

Delta, Delta_FromNX = IM_From_G(G)

print(np.shape(Delta))

DeltaT = np.zeros(np.shape(Delta))
DeltaGmT = np.zeros(np.shape(Delta))



# visualize the graph with edges
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
    plt.plot(x_values, y_values, 'ko', linestyle = '-', linewidth = w[nn, nn])
plt.title("Graph structure")
# plt.show()


# algorithm for finding a feedback vertex set (not the optimum)
# go through the graph and find a tree; take the symmetric difference of the tree with the graph;
# find a vertex cover of the symmetric difference (hard?); that should be a feedback vertex set

# to find a tree, initialize at a random starting point (the root)
# find all vertices connected to the root from the incidence matrix and delete those edges

root = np.random.randint(0, N_Nodes)
root_vec = np.zeros(N_Nodes)
root_vec[root] = 1
plt.scatter(NodePos[root, 0], NodePos[root, 1], s = 200, c = 'b')

# # visualize the graph with edges connected to the root
# # plt.figure()
# for nn in range(N_edges):
#     if Delta[nn, root]==1:
#         i1 = root
#         i2 = np.where(Delta[nn, :]==-1)
#         i2 = i2[0]
#         point1 = [NodePos[i1, 0], NodePos[i1, 1]]
#         point2 = [NodePos[i2, 0], NodePos[i2, 1]]
#         x_values = [point1[0], point2[0]]
#         y_values = [point1[1], point2[1]]
#         plt.plot(x_values, y_values, 'r', linestyle = '--', linewidth = 2*w[nn, nn])
#     if Delta[nn, root]==-1:
#         i1 = root
#         i2 = np.where(Delta[nn, :]==1)
#         i2 = i2[0]
#         point1 = [NodePos[i1, 0], NodePos[i1, 1]]
#         point2 = [NodePos[i2, 0], NodePos[i2, 1]]
#         x_values = [point1[0], point2[0]]
#         y_values = [point1[1], point2[1]]
#         plt.plot(x_values, y_values, 'r', linestyle = '--', linewidth = 2*w[nn, nn])
# plt.title("Graph structure with edges near root")
#
# plt.show()



NewBranches = []
PrevBranches = [root]

VT = 1 # count the number of vertices on the tree

while VT<N_Nodes:
    if VT==1:
        Branches = [root]
    else:
        NewBranches = []

    # print(Branches)
    for branch in Branches:

        branch = np.int64(branch)

        for nn in range(N_edges):

            if DeltaGmT[nn, branch]==1:

                i2 = np.where(DeltaGmT[nn, :]==-1)
                i2 = i2[0]


                if i2 not in Branches:
                    if i2 not in PrevBranches:
                        if i2 not in NewBranches:
                            i1 = branch
                            point1 = [NodePos[i1, 0], NodePos[i1, 1]]
                            point2 = [NodePos[i2, 0], NodePos[i2, 1]]
                            x_values = [point1[0], point2[0]]
                            y_values = [point1[1], point2[1]]
                            plt.plot(x_values, y_values, 'b', linestyle = '-', linewidth = 4*w[nn, nn])
                            DeltaGmT[nn, branch] = 0
                            DeltaGmT[nn, i2] = 0
                            DeltaT[nn, i1] = 1
                            DeltaT[nn, i2] = -1
                            NewBranches = np.append(NewBranches, i2)
                            VT+=1

            if DeltaGmT[nn, branch]==-1:

                i2 = np.where(DeltaGmT[nn, :]==1)
                i2 = i2[0]

                if i2 not in Branches:
                    if i2 not in PrevBranches:
                        if i2 not in NewBranches:
                            i1 = branch
                            point1 = [NodePos[i1, 0], NodePos[i1, 1]]
                            point2 = [NodePos[i2, 0], NodePos[i2, 1]]
                            x_values = [point1[0], point2[0]]
                            y_values = [point1[1], point2[1]]
                            plt.plot(x_values, y_values, 'b', linestyle = '-', linewidth = 4*w[nn, nn])
                            DeltaGmT[nn, branch] = 0
                            DeltaGmT[nn, i2] = 0
                            DeltaT[nn, i1] = -1
                            DeltaT[nn, i2] = 1
                            NewBranches = np.append(NewBranches, i2)
                            VT+=1

    # print(PrevBranches)
    PrevBranches = np.append(PrevBranches, Branches)

    Branches = NewBranches

points = np.append(PrevBranches, Branches)
# print(points)


plt.title("One instance of a tree (blue) for a given graph G (black)")

# plt.show()

# DeltaT = np.abs(Delta)-np.abs(DeltaGmT)

plt.figure()
plt.matshow(np.transpose(DeltaT))
plt.title("Incidence matrix for T")

plt.figure()
plt.matshow(np.transpose(DeltaGmT))
plt.title("Incidence matrix for G - T")

plt.figure()
plt.matshow(np.transpose(Delta))
plt.title("Incidence matrix for G")

#
#for ii in points:
#    ii = np.int64(ii)
#    for nn in range(N_edges):
#        DeltaGmT[nn, ii]=0


# visualize only the spanning tree
plt.figure()
for nn in range(N_edges):
    i1 = np.where(DeltaT[nn, :]==1)
    i2 = np.where(DeltaT[nn, :]==-1)
    i1 = i1[0]
    i2 = i2[0]
    point1 = [NodePos[i1, 0], NodePos[i1, 1]]
    point2 = [NodePos[i2, 0], NodePos[i2, 1]]
    x_values = [point1[0], point2[0]]
    y_values = [point1[1], point2[1]]
    plt.plot(x_values, y_values, 'ko', linestyle = '-', linewidth = w[nn, nn])
plt.title("Acyclic graph structure")
# plt.show()

# visualize G minus the spanning tree
plt.figure()
for nn in range(N_edges):
    i1 = np.where(DeltaGmT[nn, :]==1)
    i2 = np.where(DeltaGmT[nn, :]==-1)
    i1 = i1[0]
    i2 = i2[0]
    point1 = [NodePos[i1, 0], NodePos[i1, 1]]
    point2 = [NodePos[i2, 0], NodePos[i2, 1]]
    x_values = [point1[0], point2[0]]
    y_values = [point1[1], point2[1]]
    plt.plot(x_values, y_values, 'ko', linestyle = '-', linewidth = w[nn, nn])
plt.title("G-T")
# plt.show()




# ----------------------------------------------------------------------------------------------
# Now take out the FVS from the above algorithm, using the randomly selected spanning tree
# ----------------------------------------------------------------------------------------------

ConnGmT = np.matmul(np.transpose(DeltaGmT), DeltaGmT)

GmT_Degrees = ConnGmT.diagonal()

# print(GmT_Degrees)


while np.sum(np.abs((DeltaGmT)))>0:

    iMax = np.argmax(GmT_Degrees)

    for ii in range(np.shape(Delta)[0]):
        if Delta[ii, iMax] != 0:
            Delta[ii, :] = np.zeros(np.size(Delta[ii, :]))
            DeltaGmT[ii, :] = np.zeros(np.size(DeltaGmT[ii, :]))

    ConnGmT = np.matmul(np.transpose(DeltaGmT), DeltaGmT)
    GmT_Degrees = ConnGmT.diagonal()


FVS = N_Nodes - np.sum(np.abs(Delta))/2
# visualize G after removing the FVS

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
    plt.plot(x_values, y_values, 'ko', linestyle = '-', linewidth = w[nn, nn])
plt.title("G after removing the FVS of size {}".format(FVS))


plt.show()

import numpy as np
import math
import scipy
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

Nr = 4
Nc = 4
N_Nodes = Nr*Nc
p = .2

# G = nx.complete_graph(N_Nodes)

# pWS = 0.3
# kWS = 4
# G = nx.watts_strogatz_graph(N_Nodes, kWS, pWS)

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
        NodePos[Nr*ii + jj, 0] = ii + 0.1*randn()
        NodePos[Nr*ii + jj, 1] = jj + 0.1*randn()

wVec = np.ones(N_edges)
w = np.diag(wVec)

# Incidence matrix in the form rows = edges, columns = vertices to match the above format
def IM_From_G(G):
    Delta_FromNX = nx.incidence_matrix(G, oriented = True)
    Delta = np.transpose(Delta_FromNX.toarray())
    return Delta, Delta_FromNX

Delta, Delta_FromNX = IM_From_G(G)


NS_Delta = scipy.linalg.null_space(np.transpose(Delta))
print("Cycles in G from the null space of the incidence matrix: {}".format(np.shape(NS_Delta)[1]))

# print(np.shape(Delta))
# print(Delta)

DeltaT = np.zeros((N_edges, N_Nodes))
DeltaGmT = Delta.copy()



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


# check the cycle space of the tree
# First define a new incidence matrix that deletes columns with all zeros
DeltaT_Reduced = DeltaT.copy()
DeleteTheseIndices = 0
while DeleteTheseIndices<np.shape(DeltaT_Reduced)[0]:
    if np.sum(np.abs(DeltaT_Reduced[DeleteTheseIndices, :]))<1e-10:
        DeltaT_Reduced = np.delete(DeltaT_Reduced, DeleteTheseIndices, axis = 0)
    else:
        DeleteTheseIndices+=1
    print(ii)

NS_DeltaT = scipy.linalg.null_space(np.transpose(DeltaT_Reduced))
print(NS_DeltaT)
print("Cycles in T from the null space of the incidence matrix: {}".format(np.shape(NS_DeltaT)[1]))


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


# Check that the graph is acyclic by looking at the null space of the incidence matrix, which should be empty




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


FVS = N_Nodes - np.int64(np.sum(np.abs(Delta))/2) - 1
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


# plt.show()



# Now convert the incidence matrix Delta back to a graph defined in NX


# Create a graph
G_new = nx.Graph()
Tree = nx.Graph()

# for ii in range(N_Nodes):
#     G_new.add_node(ii)
#     Tree.add_node(ii)

Delta = np.transpose(Delta) # to fit the format of NX
DeltaT = np.transpose(DeltaT)
# print(np.shape(DeltaT))
# print(np.shape(Delta))
# print(Delta)

# Loop through columns (edges)
num_edges = Delta.shape[1]
# print(num_edges)
# print(N_edges)
for edge_idx in range(num_edges):
    col = Delta[:, edge_idx]
    source = np.where(col == -1)[0]
    target = np.where(col == 1)[0]
    # print(source)
    # print(target)
    # print(edge_idx)

    if len(source) == 1 and len(target) == 1:
        G_new.add_edge(source[0], target[0])


for edge_idx in range(num_edges):
    col = DeltaT[:, edge_idx]
    source = np.where(col == -1)[0]
    target = np.where(col == 1)[0]
    # print(source)
    # print(target)
    # print(edge_idx)

    if len(source) == 1 and len(target) == 1:
        Tree.add_edge(source[0], target[0])

plt.figure()
nx.draw(G_new, with_labels=True)

plt.figure()
nx.draw(Tree, with_labels=True)


print(Tree.number_of_nodes())
print(N_Nodes)


plt.show()









# --------------------------------------------------------------------------------------------------
# Now iterate to find an optimal FVS by looking at many different random spanning trees
# --------------------------------------------------------------------------------------------------

# Nr = 7
# Nc = 7
# N_Nodes = Nr*Nc
# p = .25
#
# # G = nx.complete_graph(N_Nodes)
#
# check = 0
# while check == 0:
#     G = nx.erdos_renyi_graph(N_Nodes, p)
#     if np.size(list(nx.connected_components(G))) == 1:
#         check = 1
#     print("Generating a connected ER graph...")

N_edges = G.number_of_edges()

N_dim = 2

NodePos = np.zeros((N_Nodes, N_dim)) # physical degrees of freedom - the positions of the Nodes in the network



NumTrees = 100 # number of spanning trees to check
FVS_Val = np.zeros(NumTrees)


for Tree_Index in range(NumTrees):

    from numpy.random import seed
    from numpy.random import rand
    from numpy.random import randn

    for ii in range(Nr):
        for jj in range(Nc):
            NodePos[Nr*ii + jj, 0] = ii + 0.1*randn(1)
            NodePos[Nr*ii + jj, 1] = jj + 0.1*randn(1)

    wVec = np.ones(N_edges)
    w = np.diag(wVec)

    # Incidence matrix in the form rows = edges, columns = vertices to match the above format
    def IM_From_G(G):
        Delta_FromNX = nx.incidence_matrix(G, oriented = True)
        Delta = np.transpose(Delta_FromNX.toarray())
        return Delta, Delta_FromNX

    Delta, Delta_FromNX = IM_From_G(G)

    # print(np.shape(Delta))
    # print(Delta)

    DeltaT = np.zeros((N_edges, N_Nodes))
    DeltaGmT = Delta.copy()



    print(Tree_Index/NumTrees)

    root = np.random.randint(0, N_Nodes)
    root_vec = np.zeros(N_Nodes)
    root_vec[root] = 1
    # plt.scatter(NodePos[root, 0], NodePos[root, 1], s = 200, c = 'b')


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
                                # plt.plot(x_values, y_values, 'b', linestyle = '-', linewidth = 4*w[nn, nn])
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
                                # plt.plot(x_values, y_values, 'b', linestyle = '-', linewidth = 4*w[nn, nn])
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
    #
    #for ii in points:
    #    ii = np.int64(ii)
    #    for nn in range(N_edges):
    #        DeltaGmT[nn, ii]=0


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


    FVS_Val[Tree_Index] = N_Nodes - np.int64(np.sum(np.abs(Delta))/2) - 1


# Show that the FVS size depends on the initial spanning tree
plt.figure()
plt.plot(FVS_Val)
plt.xlabel("Iteration")
plt.ylabel("FVS Size")
plt.show()

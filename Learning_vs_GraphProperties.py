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
# ---------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------
# The first half of the code uses NetworkX to define a graph and find a near-optimal feedback vertex set by a random search algorithm.
# The second half performs the contrastive learning algorithm, noting the cardinality of |G|, ||G||, and the FVS,
# ---------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------





# ---------------------------------------------------------------------------------------------------------------------------
# Define a graph using NetworkX and export the incidence matrix
# ---------------------------------------------------------------------------------------------------------------------------

Nr = 10
Nc = 10
N_Nodes = Nr*Nc
p = .05

# G = nx.complete_graph(N_Nodes)

N_Graphs = 100 # number of graphs to check

FVS_Vals_g = [] # Note the size of the FVS for each graph
N_Edges_g = [] # Note the edges for each graph
LearnTime_g = [] # Note the error decay time for each graph
ErrorFinal_g = [] # Note the final error (easier and more reliable to extract than the learning rate)
C_g = [] # Size of the basis for the cycle space


for iteration in range(N_Graphs):

    print("Simulating graph {}/{}".format(iteration, N_Graphs))

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

    # print(np.shape(Delta))
    # print(Delta)

    DeltaT = np.zeros((N_edges, N_Nodes))
    DeltaGmT = Delta.copy()
    Delta_Learning = Delta.copy()




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
    #     # plt.plot(x_values, y_values, 'ko', linestyle = '-', linewidth = w[nn, nn])
    # plt.title("Graph structure")
    # plt.show()


    # algorithm for finding a feedback vertex set (not the optimum)
    # go through the graph and find a tree; take the symmetric difference of the tree with the graph;
    # find a vertex cover of the symmetric difference (hard?); that should be a feedback vertex set

    # to find a tree, initialize at a random starting point (the root)
    # find all vertices connected to the root from the incidence matrix and delete those edges

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
    # print(points)


    # plt.title("One instance of a tree (blue) for a given graph G (black)")

    # plt.show()

    # DeltaT = np.abs(Delta)-np.abs(DeltaGmT)

    # plt.figure()
    # plt.matshow(np.transpose(DeltaT))
    # plt.title("Incidence matrix for T")
    #
    # plt.figure()
    # plt.matshow(np.transpose(DeltaGmT))
    # plt.title("Incidence matrix for G - T")
    #
    # plt.figure()
    # plt.matshow(np.transpose(Delta))
    # plt.title("Incidence matrix for G")

    #
    #for ii in points:
    #    ii = np.int64(ii)
    #    for nn in range(N_edges):
    #        DeltaGmT[nn, ii]=0


    # visualize only the spanning tree
    # plt.figure()
    for nn in range(N_edges):
        i1 = np.where(DeltaT[nn, :]==1)
        i2 = np.where(DeltaT[nn, :]==-1)
        i1 = i1[0]
        i2 = i2[0]
        point1 = [NodePos[i1, 0], NodePos[i1, 1]]
        point2 = [NodePos[i2, 0], NodePos[i2, 1]]
        x_values = [point1[0], point2[0]]
        y_values = [point1[1], point2[1]]
    #     plt.plot(x_values, y_values, 'ko', linestyle = '-', linewidth = w[nn, nn])
    # plt.title("Acyclic graph structure")
    # plt.show()

    # visualize G minus the spanning tree
    # plt.figure()
    for nn in range(N_edges):
        i1 = np.where(DeltaGmT[nn, :]==1)
        i2 = np.where(DeltaGmT[nn, :]==-1)
        i1 = i1[0]
        i2 = i2[0]
        point1 = [NodePos[i1, 0], NodePos[i1, 1]]
        point2 = [NodePos[i2, 0], NodePos[i2, 1]]
        x_values = [point1[0], point2[0]]
        y_values = [point1[1], point2[1]]
    #     plt.plot(x_values, y_values, 'ko', linestyle = '-', linewidth = w[nn, nn])
    # plt.title("G-T")
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


    FVS = N_Nodes - np.int64(np.sum(np.abs(Delta))/2) - 1
    # visualize G after removing the FVS

    # plt.figure()
    for nn in range(N_edges):
        i1 = np.where(Delta[nn, :]==1)
        i2 = np.where(Delta[nn, :]==-1)
        i1 = i1[0]
        i2 = i2[0]
        point1 = [NodePos[i1, 0], NodePos[i1, 1]]
        point2 = [NodePos[i2, 0], NodePos[i2, 1]]
        x_values = [point1[0], point2[0]]
        y_values = [point1[1], point2[1]]
    #     plt.plot(x_values, y_values, 'ko', linestyle = '-', linewidth = w[nn, nn])
    # plt.title("G after removing the FVS of size {}".format(FVS))


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

    # plt.figure()
    # nx.draw(G_new, with_labels=True)
    #
    # plt.figure()
    # nx.draw(Tree, with_labels=True)
    #
    #
    # print(Tree.number_of_nodes())
    # print(N_Nodes)


    # plt.show()









    # --------------------------------------------------------------------------------------------------
    # Iterate to find an optimal FVS by looking at many different random spanning trees
    # --------------------------------------------------------------------------------------------------



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



        print("Searching spanning trees in graph {}/{}......{}".format(iteration, N_Graphs, Tree_Index/NumTrees))

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
        # Take out the FVS from the above algorithm, using the randomly selected spanning tree
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
    # plt.figure()
    # plt.plot(FVS_Val)
    # plt.xlabel("Trial")
    # plt.ylabel("FVS Size")
    # plt.show()

    FVS_Opt = np.min(FVS_Val)




    # -----------------------------------------------------------------------------
    # Perform the learning algorithm on the graph above.
    # -----------------------------------------------------------------------------


    # generate random floating point values
    from numpy.random import seed
    from numpy.random import rand
    # seed random number generator for repeatable results
    seed(10)
    wVec = rand(N_edges) + 1/2
    w = np.diag(wVec)

    print(np.shape(Delta_Learning))
    print(np.shape(w))


    # calculate the Hessian from Delta_Learning and w
    Hessian = np.matmul(np.transpose(Delta_Learning), np.matmul(w, Delta_Learning))

    # visualize the Hessian
    # plt.figure()
    # plt.matshow(Hessian)
    # plt.title("Initial Hessian Matrix")
    # plt.show()

    # calculate the equilibrium voltages for an input voltage on the left hand side, with the bottom left designated as ground
    # do this by a brute force gradient descent rather than by Kirchoff's laws, but not updating the input nodes

    # This defines the "free network", to be used to calculate the voltage applied to the clamped network

    epsilon = 1e-2 # learning rate for the energy gradient descent
    eta = 0.05 # nudge parameter - nudges the clamped state output voltage from that of the free state towards the desired output
    alpha = 0.05 # learning rate for the edge update rule

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

    # Desired outputs; arbitrary as long as min(inputs)<Outputs<max(inputs)
    VO1 = 6
    VO2 = 3
    VO3 = 0.5

    # Output indices
    VO1Ind = Nr*Nc - 1
    VO2Ind = np.int64(Nr/2 + 1)*Nc - 1
    VO3Ind = Nc - 1
    OutputIndices = [VO1Ind, VO2Ind, VO3Ind]
    Outputs = [VO1, VO2, VO3]

    RelaxationIterations = 10**4 # how many iterations in gradient descent to minimize the energy of the network for a new input voltage and new weights
    LearningIterations = 200 # iterations in the contrastive learning scheme; how many times to update the edges


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
    skip_outer = True
    for m in range(LearningIterations):
        if np.isnan(np.sum(err_RMS)):
            skip_outer = False
            break
        print("Learning in graph {}/{}.....{}".format(iteration, N_Graphs, m/LearningIterations))
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

        # Calculate \Delta_Learning V ^2 across all edges in each network and use that in the contrastive learning rule to update the edges w
        dV2F = np.square(np.matmul(Delta_Learning, NodeValsFree))
        dV2C = np.square(np.matmul(Delta_Learning, NodeValsClamped))

        wVec = wVec - (1/2)*(alpha/eta)*(dV2C - dV2F)
        w = np.diag(wVec)

        # recalculate the Hessian
        Hessian = np.matmul(np.transpose(Delta_Learning), np.matmul(w, Delta_Learning))

        # note the RMS error between the output nodes and the desired outputs
        err_RMS[m] = (1/2)*np.sqrt((NodeValsFree[VO1Ind] - VO1)**2 + (NodeValsFree[VO2Ind] - VO2)**2)

        # note the RMS change in conductances at this step
        dw_RMS[m] = (alpha/eta)*np.sqrt(np.dot((dV2C - dV2F), (dV2C - dV2F)))/(2*N_edges)

        if err_RMS[m]<1e-10:
            err_RMS = err_RMS[0:m-1]
            e_Free = e_Free[0:m-1]
            e_Clamped = e_Clamped[0:m-1]
            dw_RMS = dw_RMS[0:m-1]
            break

    if skip_outer:
        # Compare initial Hessian to the final Hessian matrix
        # plt.figure()
        # plt.matshow(Hessian)
        # plt.title("Final Hessian matrix")

        # Plot error vs iteration number
        # plt.figure()

        LearningIterationsArray = np.linspace(0, np.size(err_RMS), np.size(err_RMS))
        # plt.plot(LearningIterationsArray, err_RMS, label = "Numerical error")

        # Fit the latter half of the error curve to an exponential decay
        err_RMS_2ndHalf = err_RMS[np.int64(0.7*np.size(err_RMS))-1:np.size(err_RMS)]
        minErr = np.min(err_RMS_2ndHalf)
        err_RMS_2ndHalf = err_RMS[np.int64(0.7*np.size(err_RMS))-1:np.size(err_RMS)]/minErr
        iterations_2ndHalf = np.linspace(np.int64(0.7*np.size(err_RMS)), np.size(err_RMS), np.int64(np.size(err_RMS) - np.int64(0.7*np.size(err_RMS)))+1)
        def ExpDecay(it, T, A):
            return A*np.exp(-(it/T))

        fit_okay = False
        try:
            pars_err_RMS, cov_err_RMS = curve_fit(ExpDecay, iterations_2ndHalf, err_RMS_2ndHalf, p0=[50, 3e-1], bounds=(0, np.inf))
            pars_err_RMS, cov_err_RMS = curve_fit(ExpDecay, iterations_2ndHalf, err_RMS_2ndHalf, p0=[50, 3e-1], bounds=(0, np.inf))
            T_fit = np.round(pars_err_RMS[0],decimals = 3)
            A_fit = np.round(pars_err_RMS[1],decimals = 3)
            err_RMS_fit = A_fit*np.exp(-LearningIterationsArray/T_fit)*minErr
            fit_okay = True
        except (RuntimeError, ValueError) as e:

            print(f"Fit failed at iteration {iteration}")

            continue



        # plt.plot(LearningIterationsArray, err_RMS_fit, linestyle='--', linewidth=2, color='blue',label='Decay constant = {}'.format(T_fit))
        # plt.suptitle("Error in output voltage vs iterations number")
        # plt.title("Blue = Inputs, Red = Outputs; FVS = {}, Nodes = {}, Edges = {}".format(FVS_Opt,N_Nodes,N_edges))
        # plt.ylabel("RMS Error [V]")
        # plt.xlabel("Iteration")
        # plt.yscale("log")
        # plt.legend()

        # Plot energy after relaxation at each iteration
        # plt.figure()
        # plt.plot(e_Clamped, label = "Final energy in the free state at this iteration")
        # plt.plot(e_Clamped, label = "Final energy in the clamped state at this iteration")
        # plt.title("Energy in the circuit as it updates")
        # plt.ylabel("Energy (a.u.)")
        # plt.xlabel("Iteration")
        # plt.legend()

        # Plot RMS change in conductances after each iteration
        # plt.figure()
        # plt.plot(dw_RMS)
        # plt.title("RMS change in conductances at each iteration")
        # plt.ylabel("$\sqrt{<dw^2>}$")
        # plt.xlabel("Iteration")
        # plt.yscale("log")


        # visualize the graph with edges at its equilibrium voltage before optimization
        # plt.figure()
        # for nn in range(N_edges):
        #     i1 = np.where(Delta_Learning[nn, :]==1)
        #     i2 = np.where(Delta_Learning[nn, :]==-1)
        #     i1 = i1[0]
        #     i2 = i2[0]
        #     point1 = [NodePos[i1, 0], NodePos[i1, 1]]
        #     point2 = [NodePos[i2, 0], NodePos[i2, 1]]
        #     x_values = [point1[0], point2[0]]
        #     y_values = [point1[1], point2[1]]
            # plt.plot(x_values, y_values, 'k', linestyle = '-', linewidth = wInit[nn, nn])
        # for ii in range(N_Nodes):
        #     # plt.scatter(NodePos[ii, 0], NodePos[ii, 1], 100*NodeValsFreeInit[ii], c = 'k', alpha = 0.5)
        #     if ii in InputIndices:
        #         # plt.scatter(NodePos[ii, 0], NodePos[ii, 1], 20, c = 'b')
        #         # plt.scatter(NodePos[ii, 0], NodePos[ii, 1], 100*Inputs[list(InputIndices).index(ii)], c = 'b', alpha = 0.2)
        #     if ii in OutputIndices:
                # plt.scatter(NodePos[ii, 0], NodePos[ii, 1], 20, c = 'r')
                # plt.scatter(NodePos[ii, 0], NodePos[ii, 1], 100*Outputs[list(OutputIndices).index(ii)], c = 'r', alpha = 0.2)

        # plt.suptitle("Graph with markersize $\propto$ node voltages before optimization")
        # plt.title("Blue = Inputs, Red = Outputs; FVS = {}, Nodes = {}, Edges = {}".format(FVS_Opt,N_Nodes,N_edges))


        # visualize the final graph with edges at its equilibrium voltage after optimization
        # plt.figure()
        # for nn in range(N_edges):
        #     i1 = np.where(Delta_Learning[nn, :]==1)
        #     i2 = np.where(Delta_Learning[nn, :]==-1)
        #     i1 = i1[0]
        #     i2 = i2[0]
        #     point1 = [NodePos[i1, 0], NodePos[i1, 1]]
        #     point2 = [NodePos[i2, 0], NodePos[i2, 1]]
        #     x_values = [point1[0], point2[0]]
        #     y_values = [point1[1], point2[1]]
            # plt.plot(x_values, y_values, 'k', linestyle = '-', linewidth = w[nn, nn])
        # for ii in range(N_Nodes):
        #     # plt.scatter(NodePos[ii, 0], NodePos[ii, 1], 100*NodeValsFree[ii], c = 'k', alpha = 0.5)
        #     if ii in InputIndices:
        #         # plt.scatter(NodePos[ii, 0], NodePos[ii, 1], 20, c = 'b')
        #         # plt.scatter(NodePos[ii, 0], NodePos[ii, 1], 100*Inputs[list(InputIndices).index(ii)], c = 'b', alpha = 0.2)
        #     if ii in OutputIndices:
                # plt.scatter(NodePos[ii, 0], NodePos[ii, 1], 20, c = 'r')
                # plt.scatter(NodePos[ii, 0], NodePos[ii, 1], 100*Outputs[list(OutputIndices).index(ii)], c = 'r', alpha = 0.2)
        # plt.suptitle("Graph with markersize $\propto$ node voltages after optimization")
        # plt.title("Blue = Inputs, Red = Outputs; FVS = {}, Nodes = {}, Edges = {}".format(FVS_Opt,N_Nodes,N_edges))

        # print(NodeValsFreeInit)
        # print(NodeValsFree)

        # print(np.sqrt(np.dot(wVec, wVec)))

        # plt.show()
        if fit_okay:
            if T_fit<1e5 and T_fit>0:
                FVS_Vals_g = np.append(FVS_Vals_g, [FVS_Opt])
                N_Edges_g = np.append(N_Edges_g, [N_edges])
                LearnTime_g = np.append(LearnTime_g, [T_fit])
                ErrorFinal_g = np.append(ErrorFinal_g, [err_RMS[-2]])

                # NS_DeltaT = scipy.linalg.null_space(np.transpose(DeltaT_Reduced))
                # print(NS_DeltaT)
                # print("Cycles in T from the null space of the incidence matrix: {}".format(np.shape(NS_DeltaT)[1]))
                C_g = np.append(C_g, len(nx.cycle_basis(G)))
                # print("Size of the basis of cycles of G: {}".format(C))
                # print("Edges: {}".format(N_edges))
                # print("Nodes: {}".format(N_Nodes))
                # print("Check Euler's formula: C+N-E = {}".format(C+N_Nodes-N_edges))





# Show a histogram of how the performance depends on the size of the FVS
plt.figure()

x = np.array(FVS_Vals_g)
y = np.array(ErrorFinal_g)


# Define bin edges
x_edges = np.linspace(np.min(x), np.max(x), 10)
y_edges = np.logspace(np.min(np.log10(y)), np.max(np.log(y)), 10)

# Compute 2D histogram
hist, x_edges, y_edges = np.histogram2d(x, y, bins=(x_edges, y_edges))

# Compute positions of bars
xpos, ypos = np.meshgrid(x_edges[:-1] + 0.25, y_edges[:-1] + 0.25, indexing="ij")
xpos = xpos.ravel()
ypos = ypos.ravel()
zpos = np.zeros_like(xpos)

# Heights and dimensions
dx = dy = 0.5 * np.ones_like(zpos)
dz = hist.ravel()

# Create 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color='skyblue', edgecolor='black')

ax.set_xlabel('|FVS|')
ax.set_ylabel('Final RMS Error')
ax.set_zlabel('Frequency')

# plt.show()


# Compare to how the performance depends on the number of edges

plt.figure()

x = np.array(N_Edges_g)
y = np.array(ErrorFinal_g)

# Define bin edges
x_edges = np.linspace(np.min(x), np.max(x), 10)
y_edges = np.logspace(np.min(np.log10(y)), np.max(np.log(y)), 10)

# Compute 2D histogram
hist, x_edges, y_edges = np.histogram2d(x, y, bins=(x_edges, y_edges))

# Compute positions of bars
xpos, ypos = np.meshgrid(x_edges[:-1] + 0.25, y_edges[:-1] + 0.25, indexing="ij")
xpos = xpos.ravel()
ypos = ypos.ravel()
zpos = np.zeros_like(xpos)

# Heights and dimensions
dx = dy = 0.5 * np.ones_like(zpos)
dz = hist.ravel()

# Create 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color='skyblue', edgecolor='black')

ax.set_xlabel('Number of edges')
ax.set_ylabel('Final RMS Error')
ax.set_zlabel('Frequency')


# Compare to how the performance depends on the size of the cycle space

plt.figure()

x = np.array(C_g)
y = np.array(ErrorFinal_g)

# Define bin edges
x_edges = np.linspace(np.min(x), np.max(x), 10)
y_edges = np.linspace(np.min(np.log10(y)), np.max(np.log10(y)), 10)

# Compute 2D histogram
hist, x_edges, y_edges = np.histogram2d(x, y, bins=(x_edges, y_edges))

# Compute positions of bars
xpos, ypos = np.meshgrid(x_edges[:-1] + 0.25, y_edges[:-1] + 0.25, indexing="ij")
xpos = xpos.ravel()
ypos = ypos.ravel()
zpos = np.zeros_like(xpos)

# Heights and dimensions
dx = dy = 0.5 * np.ones_like(zpos)
dz = hist.ravel()

# Create 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color='skyblue', edgecolor='black')

ax.set_xlabel('Size of the cycle space |C|')
ax.set_ylabel('Final RMS Error')
ax.set_zlabel('Frequency')


# Look at how the size of the cycle space compares to the size of the FVS

plt.figure()

x = np.array(C_g)
y = np.array(FVS_Vals_g)

# Define bin edges
x_edges = np.linspace(np.min(x), np.max(x), 10)
y_edges = np.linspace(np.min(y), np.max(y), 10)

# Compute 2D histogram
hist, x_edges, y_edges = np.histogram2d(x, y, bins=(x_edges, y_edges))

# Compute positions of bars
xpos, ypos = np.meshgrid(x_edges[:-1] + 0.25, y_edges[:-1] + 0.25, indexing="ij")
xpos = xpos.ravel()
ypos = ypos.ravel()
zpos = np.zeros_like(xpos)

# Heights and dimensions
dx = dy = 0.5 * np.ones_like(zpos)
dz = hist.ravel()

# Create 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color='skyblue', edgecolor='black')

ax.set_ylabel('|FVS|')
ax.set_xlabel('|C|')
ax.set_zlabel('Frequency')



plt.show()

# TODO: For a given set of vertices, make a histogram showing the learning rate (or some other performance metric) vs |FVS| and ||G||

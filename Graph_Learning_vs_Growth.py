""" This code was generated with Claude. 
GRAPH GROWTH VS. LEARNING PERFORMANCE (distant input/output placement, multiple graph families)
===============================================================================================================================
Generalization of the earlier BA-only version: set GRAPH_TYPE (below) to choose which random-graph family to test.
Everything else -- the three experiments, the distant I/O placement logic, the learning rule -- is unchanged and
works identically across graph types.

Supported graph types (see GRAPH_TYPES / GRAPH_PARAMS):
  "barabasi_albert"   -- preferential attachment (hub-dominated, the original version's default)
  "erdos_renyi"        -- edges independent at random, no degree bias
  "watts_strogatz"     -- ring lattice + random rewiring (local clustering + shortcuts)
  "random_geometric"   -- nodes at random 2D positions, edges iff within a fixed radius
  "waxman"              -- nodes at random 2D positions, edges probabilistic and distance-decaying (see the
                           standalone explanation of this model provided separately)
  "random_regular"     -- every node has exactly the same degree

Not every graph family has a natural "add one node at a time" growth process (see GROWTH_SUPPORTED below), so
Experiments 2 and 3 are only available for a subset of graph types -- Experiment 1 works for all of them.

EXPERIMENT 1 (ensemble-average design):
  For each graph size N in a sweep, sample many INDEPENDENT fresh graphs of that size (of the chosen GRAPH_TYPE),
  each with a freshly chosen distant input/output placement, run the learning rule, and average. Tells you how
  the *typical* distant-I/O graph of size N, from this family, learns.

EXPERIMENT 2 (growth-trajectory design):
  Grow a SINGLE graph incrementally (one node at a time), and re-run learning on the graph as it currently stands
  at several size checkpoints along the way. Repeat over several independent growth trajectories and average
  across trajectories at each checkpoint size. "Distant" is redefined at each checkpoint (see select_distant_io)
  since which nodes count as farthest apart can change as the graph grows. Should give the same results as Exp. 1,
  depending on the graph.

EXPERIMENT 3 (fixed node pair, tracked through growth):
  Like Experiment 2, but the input/output node ASSIGNMENT is chosen once, early in the trajectory, and never
  redefined -- every later checkpoint re-measures how much closer that SAME pair has become as the surrounding
  graph grows around them. E.g. two neurons whose positions are fixed while the rest of the brain wires up, or
  two ports whose locations are fixed while a city's road network fills in.

Distance between the input and output node sets is reported two ways: hop distance (used to SELECT the distant
placement, since it's what "far apart" means in the everyday sense) and effective resistance (the physically
appropriate distance measure for this resistor-network model, since it accounts for parallel paths -- reported for
comparison, but not used for selection, so the two experiments' results aren't circularly tied to it).
===============================================================================================================================
"""

import os

# See Learning_vs_GraphProperties_optimized.py for why this matters: pin each worker process to a single BLAS
# thread so process-level parallelism (below) isn't fighting itself for cores.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np
import networkx as nx
import pandas as pd
from scipy.optimize import curve_fit
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import matplotlib.pyplot as plt


# ===============================================================================================================================
# SECTION 0: CONFIGURATION
# ===============================================================================================================================

GRAPH_TYPE = "barabasi_albert"  # <-- change this to select which graph family to test (see list below)

GRAPH_TYPES = ["barabasi_albert", "erdos_renyi", "watts_strogatz", "random_geometric", "waxman", "random_regular"]

GRAPH_PARAMS = {
    # NOTE: density parameters below were empirically tuned to reliably yield a CONNECTED graph across the
    # default N_SIZES range (15-60) -- e.g. Erdos-Renyi needs p above roughly ln(N)/N to be connected with
    # high probability, which is why p=0.3 rather than a sparser-looking value. If you widen N_SIZES to
    # include smaller N, or otherwise change these, low connectivity rates will show up as slow runs (lots
    # of retries) or a RuntimeError from sample_graph_once / grow_graph_trajectory -- both are informative
    # about needing a higher density, not a bug.
    "barabasi_albert": {"m": 2},                                    # edges added per new node
    "erdos_renyi":      {"p": 0.3},                                  # per-pair edge probability
    "watts_strogatz":   {"k": 4, "p": 0.1},                          # ring-lattice degree, rewiring probability
    "random_geometric": {"radius": 0.45},                            # connection radius, unit square domain
    "waxman":            {"beta": 0.8, "alpha": 0.4, "L": float(np.sqrt(2))},  # decay-probability params, unit square domain
    "random_regular":   {"d": 4},                                    # exact degree of every node
}

# Graph families with a natural node-by-node growth process, usable by Experiments 2 and 3 (see Section 1 for why
# the others -- Watts-Strogatz, random-regular -- aren't included: they're defined by a GLOBAL structural
# constraint that adding one node at a time can't preserve).
GROWTH_SUPPORTED = {"barabasi_albert", "erdos_renyi", "random_geometric", "waxman"}

N_SIZES = list(range(15, 81, 5))  # graph sizes to compare: 15, 20, 25, ..., 60

EXP1_GRAPHS_PER_SIZE = 50         # independent fresh graphs sampled at each size (Experiment 1)
EXP2_N_TRAJECTORIES = 50          # independent growth trajectories, each checkpointed at every size in N_SIZES (Experiments 2 & 3)

N_REPEATS = 5                     # independent learning realizations per graph (fresh task + weights each time)
LEARNING_ITERATIONS = 200
ETA = 0.05
ALPHA = 0.05
CONV_THRESHOLD = 5e-2

N_INPUTS = 3
N_OUTPUTS = 3
DISTANT_SEARCH_CANDIDATES = 300   # random input/output splits tried when searching for a maximally distant placement


# ===============================================================================================================================
# SECTION 1: GRAPH GENERATION (all six families -- single-shot sampling for Experiment 1, incremental growth for 2 & 3)
# ===============================================================================================================================

def sample_graph_once(graph_type, N, rng, params, max_tries=200):
    """
    Draw one fresh, independent, CONNECTED graph of the given type and size (used by Experiment 1). Retries
    with a new random seed if the draw comes out disconnected -- most of these models don't guarantee
    connectivity by construction (Barabasi-Albert is the one exception), so this keeps the downstream
    distance/learning machinery (which assumes a connected graph) valid.
    """
    for _ in range(max_tries):
        seed = int(rng.integers(1 << 31))
        if graph_type == "barabasi_albert":
            G = nx.barabasi_albert_graph(N, params["m"], seed=seed)
        elif graph_type == "erdos_renyi":
            G = nx.gnp_random_graph(N, params["p"], seed=seed)
        elif graph_type == "watts_strogatz":
            G = nx.connected_watts_strogatz_graph(N, params["k"], params["p"], tries=100, seed=seed)
        elif graph_type == "random_geometric":
            G = nx.random_geometric_graph(N, params["radius"], seed=seed)
        elif graph_type == "waxman":
            G = nx.waxman_graph(N, beta=params["beta"], alpha=params["alpha"], L=params.get("L"), seed=seed)
        elif graph_type == "random_regular":
            G = nx.random_regular_graph(params["d"], N, seed=seed)
        else:
            raise ValueError(f"unknown graph_type '{graph_type}' (expected one of {GRAPH_TYPES})")

        if nx.is_connected(G):
            return G

    raise RuntimeError(f"Could not sample a connected '{graph_type}' graph at N={N} after {max_tries} tries -- "
                        f"consider raising its density parameter in GRAPH_PARAMS.")


def _sample_positions(N, rng, domain=(0.0, 0.0, 1.0, 1.0)):
    """N i.i.d. uniform random positions in a rectangular domain, used by the spatial growth models below."""
    x_min, y_min, x_max, y_max = domain
    return {i: (rng.uniform(x_min, x_max), rng.uniform(y_min, y_max)) for i in range(N)}


def _grow_ba(N_final, checkpoint_sizes, rng, m):
    """
    Grow a Barabasi-Albert graph one node at a time via preferential attachment, implemented directly
    (rather than calling nx.barabasi_albert_graph repeatedly at different N) so we're guaranteed -- rather
    than relying on incidental implementation details -- that the graph at each checkpoint is a literal
    subgraph of the graph at every larger checkpoint.
    """
    # NOTE: this must match nx.barabasi_albert_graph's default seed exactly (a STAR graph on m+1 nodes,
    # not a complete graph -- an earlier version of this function used nx.complete_graph(m + 1) here,
    # which meant Experiment 1 (which calls nx.barabasi_albert_graph directly) and Experiments 2/3 (which
    # call this function) were silently drawing from two different generative processes. That's a real
    # confound worth avoiding: any Experiment-1-vs-2 divergence should come from genuine sampling
    # differences, not from comparing two different models by accident.
    G = nx.star_graph(m)
    repeated_nodes = [n for n in G.nodes() for _ in range(G.degree(n))]  # preferential-attachment sampling pool

    snapshots = {}
    if G.number_of_nodes() in checkpoint_sizes:
        snapshots[G.number_of_nodes()] = G.copy()

    source = G.number_of_nodes()
    while source < N_final:
        targets = set()
        while len(targets) < m:
            targets.add(repeated_nodes[rng.integers(len(repeated_nodes))])
        G.add_node(source)
        G.add_edges_from((source, t) for t in targets)
        repeated_nodes.extend(targets)
        repeated_nodes.extend([source] * m)
        source += 1
        if source in checkpoint_sizes:
            snapshots[source] = G.copy()

    return snapshots


def _grow_iid_pairwise(N_final, checkpoint_sizes, rng, edge_prob_fn):
    """
    Generic incremental-growth builder for any "independent pairwise edge" random graph model (Erdos-Renyi,
    random geometric, Waxman): each pair's edge decision depends only on properties fixed at node creation
    (each node's own random position, if any) and nothing about how many OTHER nodes exist or get added
    later -- so the graph restricted to the first N_c nodes is automatically a subgraph of the graph
    restricted to any larger N_c'. We therefore draw the full edge set once on N_final nodes and take
    induced subgraphs for each checkpoint, which is exactly equivalent to -- but much simpler than --
    growing the graph step by step.
    """
    G_full = nx.Graph()
    G_full.add_nodes_from(range(N_final))
    for i in range(N_final):
        for j in range(i + 1, N_final):
            if rng.random() < edge_prob_fn(i, j):
                G_full.add_edge(i, j)
    return {N_c: G_full.subgraph(range(N_c)).copy() for N_c in checkpoint_sizes}


def _grow_once(graph_type, N_final, checkpoint_sizes, rng, params):
    """Dispatches to the appropriate incremental-growth process for one attempt (see grow_graph_trajectory
    for the surrounding connectivity-retry loop)."""
    if graph_type == "barabasi_albert":
        return _grow_ba(N_final, checkpoint_sizes, rng, params["m"])

    if graph_type == "erdos_renyi":
        p = params["p"]
        return _grow_iid_pairwise(N_final, checkpoint_sizes, rng, edge_prob_fn=lambda i, j: p)

    if graph_type == "random_geometric":
        pos = _sample_positions(N_final, rng)
        radius = params["radius"]

        def prob(i, j):
            d = np.hypot(pos[i][0] - pos[j][0], pos[i][1] - pos[j][1])
            return 1.0 if d <= radius else 0.0

        return _grow_iid_pairwise(N_final, checkpoint_sizes, rng, edge_prob_fn=prob)

    if graph_type == "waxman":
        pos = _sample_positions(N_final, rng)
        beta, alpha, L = params["beta"], params["alpha"], params.get("L") or float(np.sqrt(2))

        def prob(i, j):
            d = np.hypot(pos[i][0] - pos[j][0], pos[i][1] - pos[j][1])
            return beta * np.exp(-d / (alpha * L))

        return _grow_iid_pairwise(N_final, checkpoint_sizes, rng, edge_prob_fn=prob)

    raise NotImplementedError(f"no growth process implemented for '{graph_type}'")


def grow_graph_trajectory(graph_type, N_final, checkpoint_sizes, rng, params, max_tries=50):
    """
    Public entry point for Experiments 2 & 3: grow one graph of the given type from scratch, snapshotting
    it at every size in checkpoint_sizes, retrying (regenerating the WHOLE trajectory, to preserve the
    "one continuously evolving graph" property) if any checkpoint comes out disconnected.

    Raises NotImplementedError immediately (no retry) if graph_type isn't in GROWTH_SUPPORTED -- Watts-
    Strogatz's ring-lattice structure and random-regular's exact-degree constraint are both GLOBAL
    properties of the whole graph that a node-by-node growth process can't preserve while adding one node
    at a time, so there's no well-defined growth trajectory for them here.
    """
    if graph_type not in GROWTH_SUPPORTED:
        raise NotImplementedError(
            f"'{graph_type}' doesn't have a natural node-by-node growth process usable here (it's defined by "
            f"a global structural constraint -- a ring lattice for Watts-Strogatz, exact vertex-degree "
            f"regularity for random-regular -- that adding one node at a time can't preserve). Experiments 2 "
            f"and 3 aren't available for it; use Experiment 1 (ensemble average) only, or pick a different "
            f"GRAPH_TYPE from {sorted(GROWTH_SUPPORTED)}."
        )

    checkpoint_sizes = sorted(set(checkpoint_sizes))
    for _ in range(max_tries):
        snapshots = _grow_once(graph_type, N_final, checkpoint_sizes, rng, params)
        if all(nx.is_connected(G) for G in snapshots.values()):
            return snapshots

    raise RuntimeError(f"Could not grow a fully-connected '{graph_type}' trajectory after {max_tries} attempts -- "
                        f"consider raising its density parameter in GRAPH_PARAMS.")


# ===============================================================================================================================
# SECTION 2: DISTANT INPUT/OUTPUT PLACEMENT
# ===============================================================================================================================

def select_distant_io(G, n_inputs=N_INPUTS, n_outputs=N_OUTPUTS, n_candidates=DISTANT_SEARCH_CANDIDATES,
                       rng=None, mode="max"):
    """
    Randomly search for a disjoint (input_nodes, output_nodes) split that is far apart, measured by mean
    shortest-path hop distance between every input node and every output node. This is a simple randomized
    search rather than an exact combinatorial optimum (exact would mean checking all splits, which is
    combinatorially infeasible) -- with a few hundred candidates it reliably finds a split close to the
    graph's diameter for graphs of this size.

    mode="max" (default) finds a distant placement; mode="min" finds a close one, which is useful as a
    control condition if you want to compare against "close together" rather than just "random."
    """
    if rng is None:
        rng = np.random.default_rng()
    nodes = list(G.nodes())
    dist = dict(nx.all_pairs_shortest_path_length(G))

    best_score = -np.inf if mode == "max" else np.inf
    best_split = None
    for _ in range(n_candidates):
        chosen = rng.choice(nodes, size=n_inputs + n_outputs, replace=False)
        input_idx, output_idx = chosen[:n_inputs], chosen[n_inputs:]
        score = np.mean([dist[i][o] for i in input_idx for o in output_idx])
        if (mode == "max" and score > best_score) or (mode == "min" and score < best_score):
            best_score, best_split = score, (input_idx, output_idx)

    return best_split[0], best_split[1], best_score


def mean_hop_distance(G, set_a, set_b):
    """Mean shortest-path hop distance between every node in set_a and every node in set_b, on graph G as
    it currently stands. Unlike select_distant_io, this doesn't search for anything -- it just measures the
    distance between a SPECIFIC given pair of node sets, which is what Experiment 3 needs: re-measuring the
    same fixed nodes' distance as the surrounding graph grows, rather than re-selecting which nodes count as
    "distant" at each checkpoint."""
    return float(np.mean([nx.shortest_path_length(G, a, b) for a in set_a for b in set_b]))


def effective_resistance_sets(G, set_a, set_b):
    """
    Mean pairwise effective resistance between every node in set_a and every node in set_b, computed from
    the Moore-Penrose pseudoinverse of the graph Laplacian: R_eff(i,j) = L+[i,i] + L+[j,j] - 2*L+[i,j].
    Unlike hop distance, this accounts for parallel paths (many independent routes between two nodes lowers
    their effective resistance even if each individual route is long) -- the more physically appropriate
    notion of "distance" here, since the learning dynamics literally are a resistor-network current flow.
    Reported for comparison only; placement selection above uses hop distance.
    """
    L = nx.laplacian_matrix(G).toarray().astype(float)
    L_pinv = np.linalg.pinv(L)
    vals = [L_pinv[a, a] + L_pinv[b, b] - 2 * L_pinv[a, b] for a in set_a for b in set_b]
    return float(np.mean(vals))


# ===============================================================================================================================
# SECTION 3: CONTRASTIVE LEARNING WITH EXPLICIT (NOT RANDOM) I/O PLACEMENT
# ===============================================================================================================================

def solve_equilibrium(Hessian, fixed_idx, fixed_vals, N_Nodes):
    """Direct linear solve for the node-voltage equilibrium given a set of clamped node values. See
    Learning_vs_GraphProperties_optimized.py for the full derivation."""
    fixed_idx = np.asarray(fixed_idx, dtype=int)
    fixed_vals = np.asarray(fixed_vals, dtype=float)
    free_idx = np.setdiff1d(np.arange(N_Nodes), fixed_idx)
    H_ff = Hessian[np.ix_(free_idx, free_idx)]
    H_fc = Hessian[np.ix_(free_idx, fixed_idx)]
    V = np.zeros(N_Nodes)
    V[fixed_idx] = fixed_vals
    V[free_idx] = np.linalg.solve(H_ff, -H_fc @ fixed_vals)
    return V


def run_single_task_fixed_io(Delta_Learning, N_Nodes, N_edges, input_idx, output_idx, rng,
                              learning_iterations=LEARNING_ITERATIONS, eta=ETA, alpha=ALPHA,
                              conv_threshold=CONV_THRESHOLD):
    """
    One contrastive-learning run with an explicitly specified (not randomly chosen) input/output node
    assignment -- everything else (fixed target function, fresh random training example each iteration,
    held-out generalization check) matches run_single_task in the main script.
    """
    n_in, n_out = len(input_idx), len(output_idx)

    target_matrix = rng.random((n_out, n_in))
    target_matrix /= target_matrix.sum(axis=1, keepdims=True)

    def sample_task():
        inputs = rng.random(n_in) * 10.0
        inputs[0] = 0.0  # first input pinned near 0 (ground-like reference)
        outputs = target_matrix @ inputs
        return inputs, outputs

    wVec = rng.random(N_edges) + 0.5
    Hessian = Delta_Learning.T @ np.diag(wVec) @ Delta_Learning

    err_RMS = np.full(learning_iterations, np.nan)
    T_conv = np.nan

    for m_it in range(learning_iterations):
        inputs, outputs = sample_task()

        NodeValsFree = solve_equilibrium(Hessian, input_idx, inputs, N_Nodes)

        clamped_targets = eta * outputs + (1 - eta) * NodeValsFree[output_idx]
        clamped_idx = np.concatenate([input_idx, output_idx])
        clamped_vals = np.concatenate([inputs, clamped_targets])
        NodeValsClamped = solve_equilibrium(Hessian, clamped_idx, clamped_vals, N_Nodes)

        dV2F = (Delta_Learning @ NodeValsFree) ** 2
        dV2C = (Delta_Learning @ NodeValsClamped) ** 2
        wVec = wVec - 0.5 * (alpha / eta) * (dV2C - dV2F)
        Hessian = Delta_Learning.T @ np.diag(wVec) @ Delta_Learning

        err_RMS[m_it] = np.sqrt(np.mean((NodeValsFree[output_idx] - outputs) ** 2))
        if np.isnan(err_RMS[m_it]) or np.isinf(err_RMS[m_it]):
            return None

        if np.isnan(T_conv) and err_RMS[m_it] < conv_threshold:
            T_conv = m_it

    T_fit = np.nan
    n = len(err_RMS)
    tail = err_RMS[int(0.7 * n):]
    tail_idx = np.arange(int(0.7 * n), n)
    if np.min(tail) > 0:
        min_err = np.min(tail)
        try:
            def exp_decay(it, T, A):
                return A * np.exp(-it / T)
            pars, _ = curve_fit(exp_decay, tail_idx, tail / min_err, p0=[50, 0.3], bounds=(0, np.inf))
            T_fit = pars[0]
        except (RuntimeError, ValueError):
            pass

    return {"err_final": err_RMS[-1], "T_fit": T_fit, "T_conv": T_conv}


def _aggregate(realizations, key):
    """Mean and std of `key` across realizations, ignoring NaNs."""
    vals = np.array([r[key] for r in realizations], dtype=float)
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        return np.nan, np.nan
    return float(np.mean(vals)), (float(np.std(vals)) if len(vals) > 1 else 0.0)


def _run_learning_with_io(G, N_nodes, input_idx, output_idx, rng, hop_distance=None):
    """Core per-graph pipeline given an EXPLICIT input/output placement: run N_REPEATS learning
    realizations and aggregate, plus report both distance measures for this placement on this graph.
    hop_distance can be passed in if already computed (e.g. by select_distant_io, which computes it as a
    side effect of searching) to avoid recomputing it; otherwise it's measured fresh via mean_hop_distance."""
    if hop_distance is None:
        hop_distance = mean_hop_distance(G, input_idx, output_idx)
    eff_resistance = effective_resistance_sets(G, input_idx, output_idx)

    Delta_Learning = nx.incidence_matrix(G, oriented=True).toarray().T
    N_edges = G.number_of_edges()

    realizations = [
        r for r in (run_single_task_fixed_io(Delta_Learning, N_nodes, N_edges, input_idx, output_idx, rng)
                    for _ in range(N_REPEATS))
        if r is not None
    ]
    if not realizations:
        return None

    err_mean, err_std = _aggregate(realizations, "err_final")
    Tfit_mean, Tfit_std = _aggregate(realizations, "T_fit")
    Tconv_mean, Tconv_std = _aggregate(realizations, "T_conv")

    return {
        "N": N_nodes,
        "n_edges": N_edges,
        "hop_distance": hop_distance,
        "eff_resistance": eff_resistance,
        "err_final_mean": err_mean, "err_final_std": err_std,
        "T_fit_mean": Tfit_mean, "T_fit_std": Tfit_std,
        "T_conv_mean": Tconv_mean, "T_conv_std": Tconv_std,
        "n_valid_repeats": len(realizations),
    }


def _run_learning_on_graph(G, N_nodes, rng):
    """Wrapper used by Experiments 1 and 2: SELECTS a distant I/O placement fresh on this graph, then
    delegates to _run_learning_with_io. (Experiment 3 instead reuses a placement chosen once on an
    earlier, smaller snapshot -- that's the whole difference between "redefine distant every checkpoint"
    and "track the same fixed pair.")"""
    input_idx, output_idx, hop_distance = select_distant_io(G, rng=rng)
    return _run_learning_with_io(G, N_nodes, input_idx, output_idx, rng, hop_distance=hop_distance)


# ===============================================================================================================================
# SECTION 4: EXPERIMENT 1 -- ENSEMBLE-AVERAGE DESIGN (fresh independent graph per size, per trial)
# ===============================================================================================================================

def run_ensemble_trial(N_nodes, trial_index, graph_type=GRAPH_TYPE, params=None):
    """One (graph size, trial) unit of work for Experiment 1: sample one fresh graph of size N_nodes,
    place I/O far apart, run the learning realizations. Designed to be called independently so trials can
    be parallelized across processes."""
    params = params if params is not None else GRAPH_PARAMS[graph_type]
    rng = np.random.default_rng()
    G = sample_graph_once(graph_type, N_nodes, rng, params)
    result = _run_learning_on_graph(G, N_nodes, rng)
    if result is not None:
        result["trial_index"] = trial_index
    return result


# ===============================================================================================================================
# SECTION 5: EXPERIMENT 2 -- GROWTH-TRAJECTORY DESIGN (one evolving graph, checkpointed at each size)
# ===============================================================================================================================

def run_growth_trajectory(trajectory_index, graph_type=GRAPH_TYPE, params=None, checkpoint_sizes=tuple(N_SIZES)):
    """One full growth trajectory for Experiment 2: grow a single graph, and at every checkpoint size,
    place I/O far apart (redefined fresh at each checkpoint) and run the learning realizations. Returns a
    list of per-checkpoint result dicts. Designed to be called independently so trajectories can be
    parallelized across processes."""
    params = params if params is not None else GRAPH_PARAMS[graph_type]
    rng = np.random.default_rng()
    snapshots = grow_graph_trajectory(graph_type, checkpoint_sizes[-1], checkpoint_sizes, rng, params)

    results = []
    for N_c in checkpoint_sizes:
        result = _run_learning_on_graph(snapshots[N_c], N_c, rng)
        if result is not None:
            result["trajectory_index"] = trajectory_index
            results.append(result)
    return results


# ===============================================================================================================================
# SECTION 5b: EXPERIMENT 3 -- FIXED NODE PAIR (same two node sets tracked as the surrounding graph grows)
# ===============================================================================================================================
#
# Unlike Experiment 2, where "distant" is re-searched for at every checkpoint (so the specific nodes serving as
# input/output can change from one checkpoint to the next), this pins one input/output node ASSIGNMENT once,
# early in the trajectory, and never changes it again. Every later checkpoint re-measures (via
# mean_hop_distance / effective_resistance_sets) how close those SAME nodes have become as new nodes and edges
# accumulate around them -- e.g. two neurons whose identities/positions are fixed while the rest of the brain
# wires up around them, or two ports whose locations are fixed while a city's road network fills in and links
# them via more and shorter routes. Since node IDs are only ever added (never renumbered) as the trajectory
# grows, indices chosen in the smallest snapshot remain valid node references in every later one.

def run_growth_trajectory_fixed_pair(trajectory_index, graph_type=GRAPH_TYPE, params=None,
                                      checkpoint_sizes=tuple(N_SIZES)):
    """One full growth trajectory for Experiment 3. Designed to be called independently so trajectories
    can be parallelized across processes, same as Experiment 2."""
    params = params if params is not None else GRAPH_PARAMS[graph_type]
    rng = np.random.default_rng()
    checkpoint_sizes = tuple(sorted(checkpoint_sizes))
    assert checkpoint_sizes[0] >= N_INPUTS + N_OUTPUTS, (
        "smallest checkpoint must have enough nodes to select the initial fixed I/O pair from"
    )
    snapshots = grow_graph_trajectory(graph_type, checkpoint_sizes[-1], checkpoint_sizes, rng, params)

    # Select the fixed I/O node assignment ONCE, on the smallest available snapshot -- held fixed for
    # every later checkpoint below.
    G0 = snapshots[checkpoint_sizes[0]]
    input_idx, output_idx, _ = select_distant_io(G0, rng=rng)

    results = []
    for N_c in checkpoint_sizes:
        G = snapshots[N_c]
        # hop_distance is NOT passed in here (unlike Experiment 2's use of _run_learning_on_graph) --
        # it's deliberately re-measured fresh on the current (larger) graph for this same fixed pair,
        # which is the whole point: watching that number change as the graph grows around them.
        result = _run_learning_with_io(G, N_c, input_idx, output_idx, rng)
        if result is not None:
            result["trajectory_index"] = trajectory_index
            results.append(result)
    return results


# ===============================================================================================================================
# SECTION 6: RUN THE EXPERIMENTS (parallelized) AND AGGREGATE
# ===============================================================================================================================

if __name__ == "__main__":

    growth_supported = GRAPH_TYPE in GROWTH_SUPPORTED
    print(f"Graph type: {GRAPH_TYPE}  (params: {GRAPH_PARAMS[GRAPH_TYPE]})")
    if not growth_supported:
        print(f"NOTE: '{GRAPH_TYPE}' has no defined growth process, so Experiments 2 and 3 will be skipped "
              f"-- only Experiment 1 (ensemble average) runs for this graph type.\n")

    # --- Experiment 1: ensemble average (always runs, for every graph type) ------------------------------------------------
    exp1_tasks = [(N, trial) for N in N_SIZES for trial in range(EXP1_GRAPHS_PER_SIZE)]
    print(f"Experiment 1 (ensemble average): {len(N_SIZES)} sizes x {EXP1_GRAPHS_PER_SIZE} independent graphs each...")
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(run_ensemble_trial, N, trial) for N, trial in exp1_tasks]
        exp1_raw = [
            f.result() for f in tqdm(as_completed(futures), total=len(futures), desc="Experiment 1", unit="graph")
        ]
    df_exp1 = pd.DataFrame([r for r in exp1_raw if r is not None])
    df_exp1.to_csv(f"{GRAPH_TYPE}_experiment1_ensemble.csv", index=False)

    df_exp2 = df_exp3 = None
    if growth_supported:
        # --- Experiment 2: growth trajectories, distant I/O re-selected each checkpoint ------------------------------------
        print(f"\nExperiment 2 (growth trajectories): {EXP2_N_TRAJECTORIES} independent trajectories x "
              f"{len(N_SIZES)} checkpoints each...")
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(run_growth_trajectory, traj) for traj in range(EXP2_N_TRAJECTORIES)]
            exp2_raw = []
            for f in tqdm(as_completed(futures), total=len(futures), desc="Experiment 2", unit="trajectory"):
                exp2_raw.extend(f.result())
        df_exp2 = pd.DataFrame(exp2_raw)
        df_exp2.to_csv(f"{GRAPH_TYPE}_experiment2_growth_trajectory.csv", index=False)

        # --- Experiment 3: fixed node pair, tracked through growth ----------------------------------------------------------
        print(f"\nExperiment 3 (fixed node pair): {EXP2_N_TRAJECTORIES} independent trajectories x "
              f"{len(N_SIZES)} checkpoints each...")
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(run_growth_trajectory_fixed_pair, traj) for traj in range(EXP2_N_TRAJECTORIES)]
            exp3_raw = []
            for f in tqdm(as_completed(futures), total=len(futures), desc="Experiment 3", unit="trajectory"):
                exp3_raw.extend(f.result())
        df_exp3 = pd.DataFrame(exp3_raw)
        df_exp3.to_csv(f"{GRAPH_TYPE}_experiment3_fixed_pair.csv", index=False)

    # ---------------------------------------------------------------------------------------------------------------------
    # AGGREGATE ACROSS TRIALS/TRAJECTORIES, PER GRAPH SIZE (mean and standard error of the mean)
    # ---------------------------------------------------------------------------------------------------------------------
    def summarize(df, group_col="N"):
        g = df.groupby(group_col)
        n = g.size()
        summary = pd.DataFrame({
            "n_samples": n,
            "T_conv_mean": g["T_conv_mean"].mean(),
            "T_conv_sem": g["T_conv_mean"].std() / np.sqrt(n),
            "err_final_mean": g["err_final_mean"].mean(),
            "err_final_sem": g["err_final_mean"].std() / np.sqrt(n),
            "hop_distance_mean": g["hop_distance"].mean(),
            "eff_resistance_mean": g["eff_resistance"].mean(),
        })
        return summary.reset_index()

    summary_exp1 = summarize(df_exp1)
    summary_exp1.to_csv(f"{GRAPH_TYPE}_experiment1_summary_by_size.csv", index=False)
    print("\nExperiment 1 (ensemble) summary by N:")
    print(summary_exp1.to_string(index=False))

    summary_exp2 = summary_exp3 = None
    if growth_supported:
        summary_exp2 = summarize(df_exp2)
        summary_exp3 = summarize(df_exp3)
        summary_exp2.to_csv(f"{GRAPH_TYPE}_experiment2_summary_by_size.csv", index=False)
        summary_exp3.to_csv(f"{GRAPH_TYPE}_experiment3_summary_by_size.csv", index=False)
        print("\nExperiment 2 (growth trajectory, distant I/O re-selected each checkpoint) summary by N:")
        print(summary_exp2.to_string(index=False))
        print("\nExperiment 3 (growth trajectory, SAME fixed I/O pair every checkpoint) summary by N:")
        print(summary_exp3.to_string(index=False))

    # ===========================================================================================================================
    # SECTION 7: PLOTS
    # ===========================================================================================================================

    def plot_series(ax_ylabel_key, title, ylabel, yerr_key=None, yscale=None):
        """Small helper: plots summary_exp1 (always) plus summary_exp2/3 (if growth was supported) on one figure."""
        plt.figure()
        series = [(summary_exp1, 'o', "Experiment 1: ensemble average")]
        if growth_supported:
            series += [
                (summary_exp2, 's', "Experiment 2: I/O re-selected each checkpoint"),
                (summary_exp3, '^', "Experiment 3: same fixed I/O pair"),
            ]
        for summary, marker, label in series:
            if yerr_key:
                plt.errorbar(summary["N"], summary[ax_ylabel_key], yerr=summary[yerr_key],
                             marker=marker, capsize=3, label=label)
            else:
                plt.plot(summary["N"], summary[ax_ylabel_key], marker=marker, label=label)
        if yscale:
            plt.yscale(yscale)
        plt.xlabel("Number of nodes N")
        plt.ylabel(ylabel)
        plt.title(f"{title}  ({GRAPH_TYPE})")
        plt.legend()

    plot_series("T_conv_mean", "Learning time vs. graph size", "Mean convergence time T_conv (iterations)",
                yerr_key="T_conv_sem")
    plot_series("err_final_mean", "Final learning error vs. graph size", "Mean final RMS error (log scale)",
                yerr_key="err_final_sem", yscale="log")
    plot_series("hop_distance_mean", "I/O separation vs. graph size", "Mean I/O hop distance")
    plot_series("eff_resistance_mean", "Effective resistance vs. graph size", "Mean I/O effective resistance")

    plt.show()

    saved = [f"{GRAPH_TYPE}_experiment1_ensemble.csv", f"{GRAPH_TYPE}_experiment1_summary_by_size.csv"]
    if growth_supported:
        saved += [f"{GRAPH_TYPE}_experiment2_growth_trajectory.csv", f"{GRAPH_TYPE}_experiment3_fixed_pair.csv",
                  f"{GRAPH_TYPE}_experiment2_summary_by_size.csv", f"{GRAPH_TYPE}_experiment3_summary_by_size.csv"]
    print("\nSaved: " + ", ".join(saved))

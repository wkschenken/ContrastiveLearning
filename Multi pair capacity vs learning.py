"""
MULTI-PAIR CAPACITY VS. GRAPH SIZE
===============================================================================================================================
Extends the single-input-output-pair experiments (Graph_Growth_vs_Learning.py) to K independent input/output
PAIRS sharing one graph, to test the hypothesis that a single pair may not stress a network's capacity regardless
of size (there's already "enough" pathway redundancy even at small N), while several independent pairs competing
for the same shared edges should make graph size matter more.

TRAINING PROCEDURE -- round-robin, NOT simultaneous joint clamping:
  Rather than clamping all K pairs' inputs/outputs at once, training cycles through the K pairs one at a time:
  apply ONE pair's input, run STEPS_PER_PAIR (default 5) contrastive-learning updates using only that pair's
  input/output nodes (updating the SHARED edge-weight vector wVec), then switch to the next pair. This repeats,
  cycling through every pair, for as many rounds as needed.

  Because all K pairs share the same underlying weights, training pair B can disturb pair A's already-learned
  mapping. To catch that interference rather than mask it: EVERY pair is retrained every round (never skipped
  once it first crosses the error threshold), and convergence is checked via a periodic EVALUATION SWEEP -- after
  each full round, every pair's current error is measured (with NO further training that step) and "converged"
  means ALL K pairs are simultaneously below threshold in that same sweep, not just each pair individually at
  some point in its own history.

  The reported metric, T_conv_total, is the total number of individual weight-update STEPS elapsed (summed
  across every pair's mini-training-blocks, across every round) until that joint convergence is reached -- or
  NaN if the step budget (MAX_TOTAL_STEPS_FACTOR * K * LEARNING_ITERATIONS) is exhausted first.

NODE PLACEMENT: K pairs are placed sequentially (see select_k_distant_clustered_pairs), each built the same way
as the single-pair experiments' select_distant_clustered_io (a distant seed pair, then a compact cluster grown
around each seed) -- but with distances always measured on the FULL graph (so placement reflects true physical
distance, since current can flow through any node including ones already claimed by an earlier pair), while the
CANDIDATE POOL of nodes eligible to be newly claimed shrinks as each pair is placed, so all K pairs' node sets end
up mutually disjoint. Pairs are placed sequentially, not with an explicit "keep every pair away from every other
pair" objective -- this tends to spread them out incidentally, but isn't optimized for; that's a further
refinement worth revisiting if inter-pair proximity (as opposed to pure shared-edge capacity) turns out to matter.

GRAPH_TYPE selects which random-graph family to sample from (see GRAPH_TYPES / GRAPH_PARAMS) -- same six options
as Graph_Growth_vs_Learning.py. Only single-shot sampling is implemented here (no growth trajectories), so all
six types work uniformly; there's no GROWTH_SUPPORTED subset to worry about like in that script.
===============================================================================================================================
"""

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np
import networkx as nx
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import matplotlib.pyplot as plt


# ===============================================================================================================================
# SECTION 0: CONFIGURATION
# ===============================================================================================================================

GRAPH_TYPE = "barabasi_albert"  # <-- change this to select which graph family to test

GRAPH_TYPES = ["barabasi_albert", "erdos_renyi", "watts_strogatz", "random_geometric", "waxman", "random_regular"]

GRAPH_PARAMS = {
    # NOTE: density parameters tuned (in Graph_Growth_vs_Learning.py) to reliably yield a CONNECTED graph
    # across that script's N range (15-60); this script's N_SIZES runs higher (30-110 by default), where
    # connectivity is if anything easier to achieve, so these same values carry over safely.
    "barabasi_albert": {"m": 2},
    "erdos_renyi":      {"p": 0.3},
    "watts_strogatz":   {"k": 4, "p": 0.1},
    "random_geometric": {"radius": 0.45},
    "waxman":            {"beta": 0.8, "alpha": 0.4, "L": float(np.sqrt(2))},
    "random_regular":   {"d": 4},
}

# NOTE: N_SIZES starts much higher than the single-pair scripts (15) -- with 2+2=4 dedicated nodes per pair,
# K=5 pairs already claims 20 nodes just for I/O, so small N leaves little or no relay capacity, which would
# confound the very capacity effect this experiment is trying to isolate. Tune this range based on the largest
# K you actually test (see K_VALUES) and how much "spare" relay capacity you want guaranteed at the smallest N.
N_SIZES = [30, 50, 70, 90, 110]

K_VALUES = [1, 2, 3, 4, 5]   # number of simultaneous independent I/O pairs to compare

N_INPUTS = 2
N_OUTPUTS = 2
DISTANT_SEARCH_CANDIDATES = 300

EXP_GRAPHS_PER_SIZE = 20   # independent fresh graphs sampled at each (K, N) combination
N_REPEATS = 5              # independent round-robin training runs per graph (fresh weights + fresh target functions)

STEPS_PER_PAIR = 5         # individual weight-update steps run per pair, per visit, before switching pairs
LEARNING_ITERATIONS = 200  # used only to set the step budget below (kept for continuity with the single-pair scripts)
MAX_TOTAL_STEPS_FACTOR = 1.0  # total step budget = MAX_TOTAL_STEPS_FACTOR * K * LEARNING_ITERATIONS
ETA = 0.05
ALPHA = 0.05
CONV_THRESHOLD = 5e-2
W_MIN = 1e-3  # minimum edge conductance -- see run_round_robin_multi_pair: without this floor, the
              # contrastive weight-update rule has no mechanism keeping conductances positive, and edge
              # weights can and do go deeply negative during training (observed as low as -136 in testing).
              # A negative conductance breaks the resistor-network assumption entirely -- it's what
              # actually guarantees an equilibrium output can never lie outside the input voltages' range,
              # not just how the target functions happen to be constructed. Flooring weights at a small
              # positive value keeps the network physically valid throughout training.

# Illustrative before/after-training plots of a handful of representative sample graphs (see Section 4b) --
# a fast, SEPARATE pass that does not feed into or reuse data from the main sweep below.
VISUALIZE_SAMPLES = True
VIS_N = None         # graph size to use for the visualization pass; None -> middle value of N_SIZES
VIS_K_VALUES = None  # which K values to visualize; None -> same as K_VALUES (can mean a lot of subplots --
                      # restrict to e.g. [1, 3, 5] for a smaller "handful" if K_VALUES itself is long)


# ===============================================================================================================================
# SECTION 1: GRAPH GENERATION (single-shot sampling only -- this experiment uses the ensemble-average design,
# not growth trajectories: "same fixed pair, graph grows around it" doesn't have an obvious K-pair analog, so
# it's left out here rather than guessed at.)
# ===============================================================================================================================

def sample_graph_once(graph_type, N, rng, params, max_tries=200):
    """Draw one fresh, independent, CONNECTED graph of the given type and size, retrying with a new seed
    if the draw comes out disconnected. Most of these models don't guarantee connectivity by construction
    (Barabasi-Albert is the one exception), so the retry matters."""
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


# ===============================================================================================================================
# SECTION 2: DISTANT, CLUSTERED I/O PLACEMENT -- FOR K PAIRS, DRAWN SEQUENTIALLY FROM A SHRINKING POOL
# ===============================================================================================================================

def select_distant_io(G, n_inputs, n_outputs, n_candidates, rng, mode="max", candidate_nodes=None):
    """
    Randomly search for a disjoint (input_nodes, output_nodes) split that is far apart, measured by mean
    shortest-path hop distance -- ALWAYS on the full graph G (so distance reflects true physical routing,
    even when candidate_nodes restricts which specific nodes are eligible to be chosen, e.g. because an
    earlier pair already claimed some).
    """
    nodes = list(candidate_nodes) if candidate_nodes is not None else list(G.nodes())
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


def grow_cluster(G, seed, cluster_size, excluded, rng):
    """Grow a compact cluster of `cluster_size` nodes around `seed` by taking the closest available nodes
    by hop distance (ties broken randomly), excluding anything in `excluded`. Distance is always measured
    on the full graph G."""
    dist_from_seed = nx.single_source_shortest_path_length(G, seed)
    candidates = [n for n in dist_from_seed if n != seed and n not in excluded]
    candidates.sort(key=lambda n: (dist_from_seed[n], rng.random()))
    return np.array([seed] + candidates[:cluster_size - 1])


def select_distant_clustered_io(G, n_inputs, n_outputs, n_candidates, rng, excluded_nodes=frozenset()):
    """Single distant, clustered input/output placement (as in Graph_Growth_vs_Learning.py), but with an
    excluded_nodes set threaded through so it can be called repeatedly to place several mutually disjoint
    pairs on the same graph -- see select_k_distant_clustered_pairs."""
    candidate_nodes = [n for n in G.nodes() if n not in excluded_nodes]
    seed_in, seed_out, _ = select_distant_io(G, 1, 1, n_candidates, rng, candidate_nodes=candidate_nodes)
    input_seed, output_seed = int(seed_in[0]), int(seed_out[0])

    input_idx = grow_cluster(G, input_seed, n_inputs, excluded=excluded_nodes | {output_seed}, rng=rng)
    output_idx = grow_cluster(G, output_seed, n_outputs, excluded=excluded_nodes | set(input_idx.tolist()), rng=rng)
    return input_idx, output_idx


def select_k_distant_clustered_pairs(G, K, n_inputs=N_INPUTS, n_outputs=N_OUTPUTS,
                                      n_candidates=DISTANT_SEARCH_CANDIDATES, rng=None):
    """Place K independent input/output pairs on the same graph, drawn sequentially so all K pairs' node
    sets are mutually disjoint. Raises a clear error (rather than a cryptic one deep inside grow_cluster)
    if the graph is too small to fit K pairs."""
    if rng is None:
        rng = np.random.default_rng()
    if G.number_of_nodes() < K * (n_inputs + n_outputs):
        raise RuntimeError(f"Graph has {G.number_of_nodes()} nodes, not enough to place {K} pairs of "
                            f"{n_inputs}+{n_outputs} nodes each ({K * (n_inputs + n_outputs)} needed). "
                            f"Increase N or reduce K.")
    used = set()
    pairs = []
    for _ in range(K):
        input_idx, output_idx = select_distant_clustered_io(G, n_inputs, n_outputs, n_candidates, rng,
                                                              excluded_nodes=frozenset(used))
        pairs.append((input_idx, output_idx))
        used.update(input_idx.tolist())
        used.update(output_idx.tolist())
    return pairs


# ===============================================================================================================================
# SECTION 3: ROUND-ROBIN MULTI-PAIR TRAINING
# ===============================================================================================================================

def solve_equilibrium(Hessian, fixed_idx, fixed_vals, N_Nodes):
    """Direct linear solve for the node-voltage equilibrium given a set of clamped node values."""
    fixed_idx = np.asarray(fixed_idx, dtype=int)
    fixed_vals = np.asarray(fixed_vals, dtype=float)
    free_idx = np.setdiff1d(np.arange(N_Nodes), fixed_idx)
    H_ff = Hessian[np.ix_(free_idx, free_idx)]
    H_fc = Hessian[np.ix_(free_idx, fixed_idx)]
    V = np.zeros(N_Nodes)
    V[fixed_idx] = fixed_vals
    V[free_idx] = np.linalg.solve(H_ff, -H_fc @ fixed_vals)
    return V


def run_round_robin_multi_pair(Delta_Learning, N_Nodes, N_edges, input_idx_list, output_idx_list, rng,
                                steps_per_pair=STEPS_PER_PAIR, eta=ETA, alpha=ALPHA,
                                conv_threshold=CONV_THRESHOLD, max_total_steps=None, return_weights=False):
    """
    One full round-robin training run across K pairs sharing one graph (see module docstring for the
    procedure). Returns a dict with T_conv_total (NaN if the step budget was exhausted first), whether it
    converged, how many rounds it took, and each pair's error at the final evaluation sweep.

    return_weights=True additionally includes "wVec_initial" and "wVec_final" in the result -- off by
    default (the main sweep doesn't need this and it would otherwise bloat every parallelized result), on
    for the sample-graph visualization pass in Section 4b, which wants to plot both.
    """
    K = len(input_idx_list)
    if max_total_steps is None:
        max_total_steps = int(MAX_TOTAL_STEPS_FACTOR * K * LEARNING_ITERATIONS)

    target_matrices = []
    for k in range(K):
        tm = rng.random((len(output_idx_list[k]), len(input_idx_list[k])))
        tm /= tm.sum(axis=1, keepdims=True)
        # Each row of tm is nonnegative and sums to 1, i.e. a convex combination -- this is what
        # guarantees every output = tm @ inputs lands within [min(inputs), max(inputs)] for THIS pair,
        # for any inputs whatsoever, so the target function can never ask for a physically unreachable
        # output. Asserted explicitly (cost: negligible, done once per pair, not per training step) so
        # this stays true under future edits rather than being an invariant a reader has to re-derive.
        assert np.all(tm >= 0) and np.allclose(tm.sum(axis=1), 1.0), \
            "target matrix rows must be nonnegative and sum to 1 (convex combination of inputs)"
        target_matrices.append(tm)

    def sample_task(k):
        n_in_k = len(input_idx_list[k])
        inputs = rng.random(n_in_k) * 10.0
        inputs[0] = 0.0  # first input of each pair pinned near 0 (ground-like), matching the single-pair convention
        outputs = target_matrices[k] @ inputs
        return inputs, outputs

    def evaluate_pair(k, Hessian):
        """Measure pair k's current error with NO further training: a fresh input example, solved at the
        current shared weights, error against that pair's own target function."""
        inputs, outputs = sample_task(k)
        NodeValsFree = solve_equilibrium(Hessian, input_idx_list[k], inputs, N_Nodes)
        return float(np.sqrt(np.mean((NodeValsFree[output_idx_list[k]] - outputs) ** 2)))

    wVec = rng.random(N_edges) + 0.5
    wVec_initial = wVec.copy()
    Hessian = Delta_Learning.T @ np.diag(wVec) @ Delta_Learning

    step_count = 0
    round_num = 0

    while step_count < max_total_steps:
        round_num += 1
        for k in range(K):
            for _ in range(steps_per_pair):
                inputs, outputs = sample_task(k)
                NodeValsFree = solve_equilibrium(Hessian, input_idx_list[k], inputs, N_Nodes)

                clamped_targets = eta * outputs + (1 - eta) * NodeValsFree[output_idx_list[k]]
                clamped_idx = np.concatenate([input_idx_list[k], output_idx_list[k]])
                clamped_vals = np.concatenate([inputs, clamped_targets])
                NodeValsClamped = solve_equilibrium(Hessian, clamped_idx, clamped_vals, N_Nodes)

                dV2F = (Delta_Learning @ NodeValsFree) ** 2
                dV2C = (Delta_Learning @ NodeValsClamped) ** 2
                wVec = wVec - 0.5 * (alpha / eta) * (dV2C - dV2F)
                wVec = np.maximum(wVec, W_MIN)  # keep conductances positive -- see W_MIN's definition above
                Hessian = Delta_Learning.T @ np.diag(wVec) @ Delta_Learning
                step_count += 1

                if not np.all(np.isfinite(wVec)):
                    result = {"T_conv_total": np.nan, "converged": False, "n_rounds": round_num, "blew_up": True,
                              "final_errors_per_pair": [np.nan] * K}
                    if return_weights:
                        result["wVec_initial"], result["wVec_final"] = wVec_initial, wVec.copy()
                    return result

                if step_count >= max_total_steps:
                    break
            if step_count >= max_total_steps:
                break

        # End of round: evaluation sweep -- every pair's error, measured with no further training this step.
        errors_this_round = [evaluate_pair(k, Hessian) for k in range(K)]
        if all(e < conv_threshold for e in errors_this_round):
            result = {"T_conv_total": step_count, "converged": True, "n_rounds": round_num, "blew_up": False,
                      "final_errors_per_pair": errors_this_round}
            if return_weights:
                result["wVec_initial"], result["wVec_final"] = wVec_initial, wVec.copy()
            return result

    final_errors = [evaluate_pair(k, Hessian) for k in range(K)]
    result = {"T_conv_total": np.nan, "converged": False, "n_rounds": round_num, "blew_up": False,
              "final_errors_per_pair": final_errors}
    if return_weights:
        result["wVec_initial"], result["wVec_final"] = wVec_initial, wVec.copy()
    return result


# ===============================================================================================================================
# SECTION 4: PER-GRAPH PIPELINE (sample graph -> place K pairs -> N_REPEATS round-robin training runs -> aggregate)
# ===============================================================================================================================

def run_multi_pair_trial(N_nodes, K, trial_index, graph_type=GRAPH_TYPE, params=None):
    """One (N, K, trial) unit of work: sample one fresh graph, place K pairs, run N_REPEATS independent
    round-robin training runs, and aggregate. Designed to be called independently so trials can be
    parallelized across processes."""
    params = params if params is not None else GRAPH_PARAMS[graph_type]
    rng = np.random.default_rng()

    G = sample_graph_once(graph_type, N_nodes, rng, params)
    pairs = select_k_distant_clustered_pairs(G, K, rng=rng)
    input_idx_list = [p[0] for p in pairs]
    output_idx_list = [p[1] for p in pairs]

    Delta_Learning = nx.incidence_matrix(G, oriented=True).toarray().T
    N_edges = G.number_of_edges()

    run_results = [
        run_round_robin_multi_pair(Delta_Learning, N_nodes, N_edges, input_idx_list, output_idx_list, rng)
        for _ in range(N_REPEATS)
    ]
    converged_runs = [r for r in run_results if r["converged"]]

    T_conv_vals = np.array([r["T_conv_total"] for r in converged_runs], dtype=float)
    T_conv_mean = float(np.mean(T_conv_vals)) if len(T_conv_vals) else np.nan
    T_conv_std = float(np.std(T_conv_vals)) if len(T_conv_vals) > 1 else 0.0

    return {
        "N": N_nodes,
        "K": K,
        "trial_index": trial_index,
        "n_edges": G.number_of_edges(),
        "T_conv_mean": T_conv_mean,
        "T_conv_std": T_conv_std,
        "convergence_rate": len(converged_runs) / N_REPEATS,  # fraction of repeats that converged within budget
        "n_converged_repeats": len(converged_runs),
    }


# ===============================================================================================================================
# SECTION 4b: SAMPLE GRAPH VISUALIZATION -- a handful of representative graphs, before/after training
# ===============================================================================================================================
# Purely illustrative: shows node roles (which nodes are inputs/outputs for which pair, vs. plain relay nodes)
# and edge weights (line thickness) both before and after round-robin training, for a small number of freshly
# sampled example graphs. This is a separate, fast pass -- it does NOT reuse or feed into the main sweep's data.

_PAIR_COLORS = plt.cm.tab10.colors  # up to 10 visually distinct colors -- plenty for typical K ranges


def _graph_layout(G, rng):
    """Use the graph's own spatial positions if it has them (random_geometric and waxman graphs carry a
    real 2D 'pos' per node), since that's a more meaningful layout than a generic force-directed one for
    those types; fall back to spring_layout otherwise."""
    if all("pos" in G.nodes[n] for n in G.nodes()):
        return {n: G.nodes[n]["pos"] for n in G.nodes()}
    return nx.spring_layout(G, seed=int(rng.integers(1 << 31)))


def plot_graph_with_io(G, input_idx_list, output_idx_list, wVec, pos, ax, title, w_min=None, w_max=None):
    """
    Draw one graph on the given axes: edges with line thickness proportional to their conductance weight,
    relay (non-I/O) nodes in light gray, and each pair's input/output nodes in a shared color (one color
    per pair, up to 10 pairs) -- circles for inputs, squares for outputs, so both pair membership (color)
    and role (shape) are visible at once.

    w_min/w_max set the normalization range for edge thickness; pass the SAME values in for a "before" and
    "after" pair of panels (see generate_sample_graph_visualizations) so thickness is directly comparable
    across them -- normalizing each panel independently would make an edge look "thicker after training"
    purely because the overall weight range shrank, even if that edge's actual weight barely changed.
    """
    w = np.asarray(wVec, dtype=float)
    w_min = w.min() if w_min is None else w_min
    w_max = w.max() if w_max is None else w_max
    w_range = w_max - w_min
    linewidths = 0.5 + 4.0 * (w - w_min) / w_range if w_range > 1e-12 else np.full_like(w, 1.5)

    nx.draw_networkx_edges(G, pos, ax=ax, edgelist=list(G.edges()), width=linewidths,
                            edge_color="gray", alpha=0.6)

    used_nodes = set()
    for k, (in_idx, out_idx) in enumerate(zip(input_idx_list, output_idx_list)):
        color = _PAIR_COLORS[k % len(_PAIR_COLORS)]
        nx.draw_networkx_nodes(G, pos, nodelist=list(in_idx), node_color=[color], node_shape='o',
                                node_size=220, ax=ax, edgecolors='black', linewidths=1.0)
        nx.draw_networkx_nodes(G, pos, nodelist=list(out_idx), node_color=[color], node_shape='s',
                                node_size=220, ax=ax, edgecolors='black', linewidths=1.0)
        used_nodes.update(int(n) for n in in_idx)
        used_nodes.update(int(n) for n in out_idx)

    relay_nodes = [n for n in G.nodes() if n not in used_nodes]
    nx.draw_networkx_nodes(G, pos, nodelist=relay_nodes, node_color="lightgray", node_size=80, ax=ax,
                            edgecolors="dimgray", linewidths=0.5)

    ax.set_title(title, fontsize=10)
    ax.set_axis_off()


def generate_sample_graph_visualizations(graph_type=GRAPH_TYPE, params=None, N=None, k_values=None, seed=None):
    """
    Produce one before/after-training figure per K in k_values: samples a fresh graph, places K pairs, runs
    one round-robin training pass, and plots the graph with its edge weights and I/O placement both before
    and after training side by side. Purely for visual sanity-checking -- doesn't feed into the main sweep.
    """
    params = params if params is not None else GRAPH_PARAMS[graph_type]
    N = N if N is not None else N_SIZES[len(N_SIZES) // 2]
    k_values = list(k_values) if k_values is not None else K_VALUES
    rng = np.random.default_rng(seed)

    fig, axes = plt.subplots(len(k_values), 2, figsize=(11, 5 * len(k_values)), squeeze=False)

    for row, K in enumerate(k_values):
        G = sample_graph_once(graph_type, N, rng, params)
        pairs = select_k_distant_clustered_pairs(G, K, rng=rng)
        input_idx_list = [p[0] for p in pairs]
        output_idx_list = [p[1] for p in pairs]

        Delta_Learning = nx.incidence_matrix(G, oriented=True).toarray().T
        N_edges = G.number_of_edges()
        result = run_round_robin_multi_pair(Delta_Learning, N, N_edges, input_idx_list, output_idx_list, rng,
                                             return_weights=True)

        pos = _graph_layout(G, rng)
        status = "converged" if result["converged"] else "did not converge"
        w_min = min(result["wVec_initial"].min(), result["wVec_final"].min())
        w_max = max(result["wVec_initial"].max(), result["wVec_final"].max())
        plot_graph_with_io(G, input_idx_list, output_idx_list, result["wVec_initial"], pos, axes[row][0],
                            f"K={K}, N={N} -- BEFORE training", w_min=w_min, w_max=w_max)
        plot_graph_with_io(G, input_idx_list, output_idx_list, result["wVec_final"], pos, axes[row][1],
                            f"K={K}, N={N} -- AFTER training ({status}, {result['n_rounds']} rounds)",
                            w_min=w_min, w_max=w_max)

    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markeredgecolor='black',
               label='Input node (color = pair)', markersize=9),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', markeredgecolor='black',
               label='Output node (color = pair)', markersize=9),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgray', markeredgecolor='dimgray',
               label='Relay node', markersize=7),
        Line2D([0], [0], color='gray', linewidth=3, label='Edge (thickness = weight)'),
    ]
    fig.legend(handles=legend_handles, loc='lower center', ncol=4, fontsize=9, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(f"Sample graphs ({graph_type})", fontsize=14)
    plt.tight_layout(rect=(0, 0.06, 1, 0.97))


# ===============================================================================================================================
# SECTION 5: RUN THE SWEEP (parallelized across (N, K, trial)) AND AGGREGATE
# ===============================================================================================================================

if __name__ == "__main__":

    if VISUALIZE_SAMPLES:
        print("Generating sample graph visualizations...")
        generate_sample_graph_visualizations(N=VIS_N, k_values=VIS_K_VALUES)

    tasks = [(N, K, trial) for K in K_VALUES for N in N_SIZES for trial in range(EXP_GRAPHS_PER_SIZE)]
    print(f"Running {len(K_VALUES)} K-values x {len(N_SIZES)} sizes x {EXP_GRAPHS_PER_SIZE} graphs each "
          f"= {len(tasks)} total graph trials...")

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(run_multi_pair_trial, N, K, trial) for N, K, trial in tasks]
        raw_results = [
            f.result() for f in tqdm(as_completed(futures), total=len(futures), desc="Graph trials", unit="graph")
        ]

    df = pd.DataFrame(raw_results)
    df.to_csv(f"{GRAPH_TYPE}_multipair_capacity_results.csv", index=False)

    # ---------------------------------------------------------------------------------------------------------------------
    # AGGREGATE ACROSS TRIALS, PER (N, K) -- mean and standard error of the mean, plus mean convergence rate
    # ---------------------------------------------------------------------------------------------------------------------
    def summarize(df):
        g = df.groupby(["K", "N"])
        n = g.size()
        summary = pd.DataFrame({
            "n_trials": n,
            "T_conv_mean": g["T_conv_mean"].mean(),
            "T_conv_sem": g["T_conv_mean"].std() / np.sqrt(n),
            "mean_convergence_rate": g["convergence_rate"].mean(),
        })
        return summary.reset_index()

    summary = summarize(df)
    summary.to_csv(f"{GRAPH_TYPE}_multipair_capacity_summary.csv", index=False)
    print("\nSummary by (K, N):")
    print(summary.to_string(index=False))

    # ===========================================================================================================================
    # SECTION 6: PLOTS
    # ===========================================================================================================================

    # --- 6a. The main requested plot: T_conv_total vs. N, one line per K, superposed ---------------------------------------
    plt.figure()
    for K in K_VALUES:
        s = summary[summary["K"] == K]
        plt.errorbar(s["N"], s["T_conv_mean"], yerr=s["T_conv_sem"], marker='o', capsize=3, label=f"K={K} pairs")
    plt.xlabel("Number of nodes N")
    plt.ylabel("Mean total steps to joint convergence (T_conv_total)")
    plt.title(f"Steps to convergence vs. graph size, by number of I/O pairs  ({GRAPH_TYPE})")
    plt.legend()

    # --- 6b. Convergence rate vs. N, one line per K -- IMPORTANT companion plot: T_conv_mean above is only computed
    # over runs that converged within budget, so if convergence_rate is low for some (K, N), that T_conv_mean is
    # survivorship-biased (only counting the "easy" successful attempts) and should be read with that in mind.
    plt.figure()
    for K in K_VALUES:
        s = summary[summary["K"] == K]
        plt.plot(s["N"], s["mean_convergence_rate"], marker='o', label=f"K={K} pairs")
    plt.xlabel("Number of nodes N")
    plt.ylabel("Fraction of runs that converged within the step budget")
    plt.title(f"Convergence rate vs. graph size, by number of I/O pairs  ({GRAPH_TYPE})")
    plt.ylim(-0.05, 1.05)
    plt.legend()

    plt.show()

    print(f"\nSaved: {GRAPH_TYPE}_multipair_capacity_results.csv, {GRAPH_TYPE}_multipair_capacity_summary.csv")

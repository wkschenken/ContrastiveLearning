""" This code was generated with Claude. 
LEARNING VS. GRAPH PROPERTIES
===============================================================================================================================
For each of N_GRAPHS randomly sampled connected graphs:
  1. Compute its (exact, when feasible) minimum Feedback Vertex Set (FVS).
  2. Run several independent realizations of a contrastive (equilibrium-propagation-style) learning rule on it,
     each with a fresh random input/output node assignment, a fresh random target function, and fresh weights.
  3. Aggregate learning-performance statistics (mean +/- std across realizations) per graph.
Finally, relate learning performance to graph structure (|FVS|, |E|, cycle-space size) via both single-variable
plots and a multivariate regression, across all sampled graphs. Graphs are processed in parallel (concurrent.futures).
===============================================================================================================================
"""

import os

# IMPORTANT: these must be set BEFORE numpy is imported (BLAS libraries read them at load time).
# By default, numpy's underlying linear-algebra library (OpenBLAS / MKL / Apple's Accelerate) will try
# to multithread every np.linalg.solve call across ALL available cores. That's fine in isolation, but
# once we're *also* running many worker processes in parallel (one per graph), each process's BLAS
# calls independently try to grab every core -- N worker processes x (BLAS threads per process) can
# vastly oversubscribe the CPU, with threads fighting each other for time instead of doing useful work.
# Pinning each process to 1 BLAS thread here means the process-level parallelism (below) is the only
# parallelism in play, which is what actually lets N_edges-per-graph work scale with core count.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import time
import warnings
import numpy as np
import networkx as nx
import pandas as pd
import pulp
from scipy.optimize import curve_fit
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Silences a known noisy NetworkX FutureWarning ("incidence_matrix will return a scipy.sparse array
# instead of a matrix...") -- purely a heads-up about a future internal representation change, and
# doesn't affect correctness here since we immediately convert to a dense np.array either way.
warnings.filterwarnings("ignore", message=r"incidence_matrix will return.*", category=FutureWarning)


# ===============================================================================================================================
# SECTION 0: GLOBAL CONFIGURATION
# ===============================================================================================================================

N_NODES = 25            # number of nodes per sampled graph
P_EDGE = 0.35            # Erdos-Renyi edge probability
N_GRAPHS = 250           # number of graphs to sample
N_REPEATS = 5            # independent task/weight-init realizations per graph (stats item #6)
LEARNING_ITERATIONS = 200
ETA = 0.05               # nudge parameter (free -> clamped state)
ALPHA = 0.05             # edge-conductance learning rate
CONV_THRESHOLD = 5e-2    # error threshold defining "converged" for the fit-independent T_conv metric (stats #4)

# Exact-FVS ILP settings (see min_fvs_exact below)
ILP_TIME_LIMIT_PER_SOLVE = 5     # seconds allotted to each individual CBC solve call
ILP_MAX_CUTS = 60                # safety cap on the number of cutting-plane rounds
ILP_MAX_TOTAL_TIME = 45          # seconds; if exceeded, abandon the exact solve and fall back to the heuristic

# Heuristic FVS fallback settings
HEURISTIC_NUM_TREES = 100


# ===============================================================================================================================
# SECTION 1: FEEDBACK VERTEX SET (FVS) ESTIMATION
# ===============================================================================================================================

def min_fvs_exact(G, time_limit_per_solve=ILP_TIME_LIMIT_PER_SOLVE, max_cuts=ILP_MAX_CUTS,
                   max_total_time=ILP_MAX_TOTAL_TIME):
    """
    Exact minimum Feedback Vertex Set via a cutting-plane ILP (branch-and-cut with lazily-added cycle
    constraints), solved with PuLP's bundled CBC solver.

    Formulation: binary x_v = 1 iff vertex v is removed; minimize sum(x_v) subject to "every cycle in G
    has at least one removed vertex" -- but a graph can have exponentially many cycles, so these
    constraints aren't listed up front. Instead:
      1. Solve the ILP with whatever cycle constraints have been added so far (initially none).
      2. Check whether the induced subgraph on the *un-removed* vertices is actually a forest.
      3. If yes -- done, this is the exact optimum. If no, every remaining cycle found this round gets
         its own constraint ("at least one of these vertices must be removed"), and we re-solve.
    This provably converges to the true minimum FVS (it's the same lazy-constraint-generation idea used
    for TSP subtour elimination), typically within a few dozen rounds for graphs of this size (~25 nodes).

    Returns (fvs_set, exact_flag). exact_flag is False if the time/iteration budget ran out before CBC
    could certify optimality -- callers should fall back to the heuristic search in that case.
    """
    nodes = list(G.nodes())
    prob = pulp.LpProblem("MinFVS", pulp.LpMinimize)
    x = {v: pulp.LpVariable(f"x_{v}", cat="Binary") for v in nodes}
    prob += pulp.lpSum(x.values())

    t_start = time.time()
    removed = set(nodes)  # conservative fallback if we bail before any solve completes
    for _ in range(max_cuts):
        if time.time() - t_start > max_total_time:
            return removed, False

        # threads=1: same reasoning as the BLAS thread-limiting above -- CBC can multithread
        # internally, which would compete with the process-level parallelism across graphs.
        prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=time_limit_per_solve, threads=1))
        status_optimal = pulp.LpStatus[prob.status] == "Optimal"
        removed = {v for v in nodes if x[v].value() is not None and x[v].value() > 0.5}
        remaining = [v for v in nodes if v not in removed]
        cycles = nx.cycle_basis(G.subgraph(remaining))

        if not cycles:
            return removed, status_optimal

        # Add a constraint for every violated cycle found this round (not just one) -- converges
        # much faster than adding a single cycle constraint per round.
        for cyc in cycles:
            prob += pulp.lpSum(x[v] for v in cyc) >= 1

    return removed, False


def heuristic_fvs_size(G, num_trees=HEURISTIC_NUM_TREES, rng=None):
    """
    Approximate FVS size (fallback for when the exact ILP solver doesn't converge in time): repeat
    [random spanning tree -> remove non-tree ("chord") edges' endpoints by iteratively deleting the
    highest chord-degree vertex] over many random spanning trees, and keep the smallest FVS found.

    Speed improvement vs. the original version: each vertex's "chord degree" (its degree within the
    non-tree-edge subgraph only) is updated incrementally as vertices are removed, rather than
    recomputing the full degree vector from scratch (an O(N^2) matrix product) after every single
    removal. Spanning trees are built with nx.bfs_tree rather than a hand-rolled Python BFS, since an
    adjacency-list representation is what makes the incremental updates below straightforward.
    """
    if rng is None:
        rng = np.random.default_rng()
    nodes = list(G.nodes())
    best_size = len(nodes)

    for _ in range(num_trees):
        root = rng.choice(nodes)
        tree_edges = set()
        for u, v in nx.bfs_tree(G, root).edges():
            tree_edges.add((u, v))
            tree_edges.add((v, u))
        non_tree_edges = [e for e in G.edges() if e not in tree_edges]

        degree = {v: 0 for v in nodes}
        chord_adj = {v: set() for v in nodes}
        for u, v in non_tree_edges:
            degree[u] += 1
            degree[v] += 1
            chord_adj[u].add(v)
            chord_adj[v].add(u)

        removed = set()
        active = set(nodes)
        remaining_chord_edges = len(non_tree_edges)
        while remaining_chord_edges > 0:
            v_max = max(active, key=lambda v: degree[v])
            removed.add(v_max)
            active.discard(v_max)
            for nb in list(chord_adj[v_max]):
                if nb in active:
                    degree[nb] -= 1
                    remaining_chord_edges -= 1
                chord_adj[nb].discard(v_max)
            chord_adj[v_max].clear()

        best_size = min(best_size, len(removed))

    return best_size


def get_fvs_size(G, rng=None):
    """Try the exact ILP solver first; fall back to the (sped-up) heuristic search if it can't
    certify optimality within its time/iteration budget. Returns (fvs_size, is_exact)."""
    removed, exact = min_fvs_exact(G)
    if exact:
        return len(removed), True
    return heuristic_fvs_size(G, rng=rng), False


# ===============================================================================================================================
# SECTION 2: CONTRASTIVE LEARNING ON A FIXED GRAPH
# ===============================================================================================================================

def solve_equilibrium(Hessian, fixed_idx, fixed_vals, N_Nodes):
    """
    Direct linear solve for the node-voltage equilibrium: the values that minimize the network's
    quadratic energy V^T Hessian V / 2 subject to a subset of nodes ("fixed_idx") being clamped to
    fixed_vals. Equivalent to Kirchhoff's current law holding at every non-clamped node -- solved
    exactly in one linear-algebra call rather than by iterative gradient descent.
    """
    fixed_idx = np.asarray(fixed_idx, dtype=int)
    fixed_vals = np.asarray(fixed_vals, dtype=float)
    free_idx = np.setdiff1d(np.arange(N_Nodes), fixed_idx)
    H_ff = Hessian[np.ix_(free_idx, free_idx)]
    H_fc = Hessian[np.ix_(free_idx, fixed_idx)]
    V = np.zeros(N_Nodes)
    V[fixed_idx] = fixed_vals
    V[free_idx] = np.linalg.solve(H_ff, -H_fc @ fixed_vals)
    return V


def run_single_task(Delta_Learning, N_Nodes, N_edges, rng,
                     learning_iterations=LEARNING_ITERATIONS, eta=ETA, alpha=ALPHA,
                     conv_threshold=CONV_THRESHOLD):
    """
    One full contrastive-learning run on a fixed graph (one "realization" -- stats item #6 calls this
    several times per graph with independent randomness and aggregates the results).

    Stats item #5 (memorization vs. generalization): rather than training toward one single fixed
    input/output voltage pair, the target is a genuine (fixed, random) FUNCTION of the inputs --
    desired_outputs = TargetMatrix @ inputs, with TargetMatrix's rows summing to 1 so outputs stay
    within [min(inputs), max(inputs)] as in the original design. A fresh random input example is drawn
    every learning iteration (so the network has to learn the mapping, not memorize one point), and a
    held-out input -- never seen during training -- is evaluated at the end with the final learned
    weights to give a genuine generalization-error metric.
    """
    nodes = np.arange(N_Nodes)

    # Fixed input/output node assignment for this realization (disjoint, so no node is both clamped
    # to an input value and nudged toward an output target)
    while True:
        input_idx = rng.choice(nodes, size=3, replace=False)
        output_idx = rng.choice(nodes, size=3, replace=False)
        if set(input_idx).isdisjoint(output_idx):
            break

    target_matrix = rng.random((3, 3))
    target_matrix /= target_matrix.sum(axis=1, keepdims=True)

    def sample_task():
        inputs = rng.random(3) * np.array([0.0, 10.0, 10.0])  # first input pinned near 0 (ground-like)
        outputs = target_matrix @ inputs
        return inputs, outputs

    wVec = rng.random(N_edges) + 0.5
    Hessian = Delta_Learning.T @ np.diag(wVec) @ Delta_Learning

    err_RMS = np.full(learning_iterations, np.nan)
    T_conv = np.nan

    for m in range(learning_iterations):
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

        # Stats item #2 fix: RMS error over all 3 outputs (the original only checked 2 of the 3)
        err_RMS[m] = np.sqrt(np.mean((NodeValsFree[output_idx] - outputs) ** 2))

        if np.isnan(err_RMS[m]) or np.isinf(err_RMS[m]):
            return None  # this realization blew up; caller discards it

        # Stats item #4: fit-independent convergence time -- first iteration where the error drops
        # below a fixed threshold, recorded regardless of whether the exponential-decay fit succeeds
        if np.isnan(T_conv) and err_RMS[m] < conv_threshold:
            T_conv = m

    # Exponential-decay fit over the last 30% of iterations (kept from before as a secondary metric)
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

    # Stats item #5: generalization check on a fresh held-out input, using the FINAL learned weights
    Hessian_final = Delta_Learning.T @ np.diag(wVec) @ Delta_Learning
    val_inputs, val_outputs = sample_task()
    NodeValsFree_val = solve_equilibrium(Hessian_final, input_idx, val_inputs, N_Nodes)
    generalization_error = np.sqrt(np.mean((NodeValsFree_val[output_idx] - val_outputs) ** 2))

    return {
        "err_final": err_RMS[-1],
        "T_fit": T_fit,
        "T_conv": T_conv,
        "generalization_error": generalization_error,
    }


# ===============================================================================================================================
# SECTION 3: PER-GRAPH PIPELINE (sample graph -> estimate FVS -> run several learning realizations -> aggregate)
# ===============================================================================================================================

def _aggregate(realizations, key):
    """Mean and std of `key` across realizations, ignoring NaNs (e.g. a failed exponential fit)."""
    vals = np.array([r[key] for r in realizations], dtype=float)
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        return np.nan, np.nan
    return float(np.mean(vals)), (float(np.std(vals)) if len(vals) > 1 else 0.0)


def run_one_graph(iteration, N_Nodes=N_NODES, p_edge=P_EDGE, n_repeats=N_REPEATS):
    """
    Full pipeline for one sampled graph: build a connected ER graph, estimate its FVS, run
    N_REPEATS independent learning realizations on it (stats item #6), and return aggregated
    per-graph statistics. Designed to be called independently per graph so the whole sweep can be
    parallelized (speed item #4) via ProcessPoolExecutor across graphs.

    Uses a fresh, independently-seeded RNG per call (rather than a fixed seed(10) as in the original
    script) -- this also removes the previously-flagged bug where initial weights were deterministic
    across all graphs, and is required anyway for the repeats in this function to be independent of
    each other.
    """
    rng = np.random.default_rng()

    while True:
        G = nx.gnp_random_graph(N_Nodes, p_edge, seed=int(rng.integers(1 << 31)))
        if nx.is_connected(G):
            break

    N_edges = G.number_of_edges()
    is_planar = nx.is_planar(G)
    cycle_size = len(nx.cycle_basis(G))  # |C| = |E| - |V| + 1 for a connected graph

    Delta_Learning = nx.incidence_matrix(G, oriented=True).toarray().T  # rows=edges, cols=vertices

    fvs_size, fvs_exact = get_fvs_size(G, rng=rng)

    realizations = [
        r for r in (run_single_task(Delta_Learning, N_Nodes, N_edges, rng) for _ in range(n_repeats))
        if r is not None
    ]
    if not realizations:
        return None

    err_mean, err_std = _aggregate(realizations, "err_final")
    Tfit_mean, Tfit_std = _aggregate(realizations, "T_fit")
    Tconv_mean, Tconv_std = _aggregate(realizations, "T_conv")
    gen_mean, gen_std = _aggregate(realizations, "generalization_error")

    return {
        "iteration": iteration,
        "fvs": fvs_size,
        "fvs_exact": fvs_exact,
        "n_edges": N_edges,
        "cycle_size": cycle_size,
        "planar": is_planar,
        "err_final_mean": err_mean, "err_final_std": err_std,
        "T_fit_mean": Tfit_mean, "T_fit_std": Tfit_std,
        "T_conv_mean": Tconv_mean, "T_conv_std": Tconv_std,
        "generalization_error_mean": gen_mean, "generalization_error_std": gen_std,
        "n_valid_repeats": len(realizations),
    }


# ===============================================================================================================================
# SECTION 4: RUN THE SWEEP (parallelized across graphs -- speed item #4) AND AGGREGATE
# ===============================================================================================================================

if __name__ == "__main__":

    print(f"Running {N_GRAPHS} graphs x {N_REPEATS} realizations each, in parallel across available cores...")
    # ProcessPoolExecutor + as_completed yields each graph's result as soon as it's done (regardless of
    # submission order), which is what lets tqdm advance the bar as work actually completes. This uses
    # only the standard library, so it doesn't depend on a specific joblib version being installed.
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(run_one_graph, i) for i in range(N_GRAPHS)]
        raw_results = [
            f.result() for f in tqdm(as_completed(futures), total=N_GRAPHS, desc="Graphs completed", unit="graph")
        ]
    results = [r for r in raw_results if r is not None]
    print(f"Completed {len(results)}/{N_GRAPHS} graphs with at least one valid learning realization.")

    df = pd.DataFrame(results)

    n_exact = int(df["fvs_exact"].sum())
    print(f"Exact FVS found for {n_exact}/{len(df)} graphs "
          f"({len(df) - n_exact} fell back to the heuristic search).")
    print(f"Percentage of graphs that are planar: {100 * df['planar'].mean():.1f}%")

    # ---------------------------------------------------------------------------------------------------------------------
    # Stats item #3: multivariate regression -- how much does each structural feature matter once the
    # others are held fixed? (Single-variable scatter/histograms below can't separate this, since |FVS|,
    # |E|, and cycle-space size are all mechanically correlated with each other.)
    # ---------------------------------------------------------------------------------------------------------------------
    valid = df["err_final_mean"] > 0
    X = df.loc[valid, ["fvs", "n_edges", "cycle_size"]].values
    y = np.log10(df.loc[valid, "err_final_mean"].values)

    scaler = StandardScaler()
    Xz = scaler.fit_transform(X)
    reg = LinearRegression().fit(Xz, y)
    r2 = reg.score(Xz, y)

    print("\nMultivariate regression: log10(mean final RMS error) ~ |FVS| + |E| + |C|  (standardized predictors)")
    for name, coef in zip(["|FVS|", "|E| (edges)", "|C| (cycle size)"], reg.coef_):
        print(f"  standardized coefficient [{name}]: {coef:+.4f}")
    print(f"  R^2 = {r2:.4f}  (n = {int(valid.sum())} graphs)")
    print("  Interpretation: the largest |coefficient| is the feature most associated with final error,")
    print("  after controlling for the other two -- unlike the single-variable plots below, which can't")
    print("  separate these since |FVS|, |E|, and |C| are all correlated with each other.\n")

    # ===========================================================================================================================
    # SECTION 5: VISUALIZE RELATIONSHIPS BETWEEN GRAPH STRUCTURE AND LEARNING PERFORMANCE
    # ===========================================================================================================================

    def plot_3d_histogram(x, y, xlabel, ylabel, title=None):
        """Reusable 2D-histogram-as-3D-bars plot (used for all comparisons below)."""
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]
        x_edges = np.linspace(x.min(), x.max(), 10)
        y_edges = np.linspace(y.min(), y.max(), 10)
        hist, x_edges, y_edges = np.histogram2d(x, y, bins=(x_edges, y_edges))
        xpos, ypos = np.meshgrid(x_edges[:-1] + 0.25, y_edges[:-1] + 0.25, indexing="ij")
        xpos, ypos = xpos.ravel(), ypos.ravel()
        zpos = np.zeros_like(xpos)
        dx = dy = 0.5 * np.ones_like(zpos)
        dz = hist.ravel()

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color='skyblue', edgecolor='black')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_zlabel('Frequency')
        if title:
            ax.set_title(title)

    log_err = np.log10(df["err_final_mean"].where(df["err_final_mean"] > 0))

    # --- 5a. Final RMS error vs. |FVS| --------------------------------------------------------------------------------------
    plot_3d_histogram(df["fvs"], log_err, '|FVS|', 'Log Final RMS Error (mean over repeats)')

    # --- 5b. Final RMS error vs. number of edges ----------------------------------------------------------------------------
    plot_3d_histogram(df["n_edges"], log_err, 'Number of edges', 'Log Final RMS Error (mean over repeats)')

    # --- 5c. Final RMS error vs. size of the cycle space |C| ----------------------------------------------------------------
    plot_3d_histogram(df["cycle_size"], log_err, 'Size of the cycle space |C|', 'Log Final RMS Error (mean over repeats)')

    # --- 5d. Size of the cycle space |C| vs. size of the FVS (structural relationship, independent of learning) ------------
    plot_3d_histogram(df["cycle_size"], df["fvs"], 'Size of the cycle space |C|', '|FVS|')

    # --- 5e. Generalization error vs. |FVS| (stats item #5 -- does structure predict held-out performance too?) -----------
    log_gen_err = np.log10(df["generalization_error_mean"].where(df["generalization_error_mean"] > 0))
    plot_3d_histogram(df["fvs"], log_gen_err, '|FVS|', 'Log Generalization RMS Error (mean over repeats)')

    plt.show()

    df.to_csv("graph_learning_results.csv", index=False)
    print("Full per-graph results saved to graph_learning_results.csv")

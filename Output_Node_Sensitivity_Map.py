# Generated with Claude. 
"""
OUTPUT-NODE CONVERGENCE-TIME SENSITIVITY MAP
===============================================================================================================================
Extracted from Multi_Pair_Capacity_vs_Learning.py's Section 4c into its own standalone script. Fixes 2 input nodes
and 1 output node on a single graph (distant, clustered placement, same convention used throughout the rest of
this project), then sweeps the SECOND output node's location across every other node in the graph, measuring mean
time-to-convergence at each candidate location. Produces a per-node map of "how learnable is this task, purely as
a function of where the second output sits" -- an empirically measured outcome per node, rather than an analytical
distance proxy.

RESISTOR MODEL: set RESISTOR_MODEL to choose the edge current-voltage law. "linear" is Ohm's law (I = w*deltaV),
solved with a single direct linear solve, same as the original scripts. The three nonlinear options each replace
that with I = w*g(deltaV) for a nonlinear g, solved via a convex-energy minimization (scipy trust-region Newton-CG)
since every g here is chosen to be the derivative of a convex potential -- guaranteeing a unique equilibrium to
find, not several to get stuck between:
  "cubic":       g(x) = x + gamma*x^3               (params: gamma; gamma<0 is NOT guaranteed convex -- see below)
  "saturating":  g(x) = tanh(x / V_sat)              (params: V_sat)
  "power_law":   g(x) = sign(x) * |x|^n              (params: n)
The weight-update rule is derived from the SAME potential in every case (see RESISTOR_MODELS / edge_potential_
per_weight below), not chosen separately by analogy -- for "linear" this is provably identical to the original
scripts' update rule, and for the other three it's the direct nonlinear generalization of it.

Caveat on "cubic": for gamma < 0 (a softening nonlinearity), the potential x^2/2 + gamma*x^4/4 stops being convex
once |x| exceeds sqrt(-1/(3*gamma)) -- the network can then have multiple equilibria, and nothing here guarantees
finding the same one twice. gamma >= 0 (hardening) is the well-posed, recommended range; gamma < 0 is left
available to experiment with, but treat any results from it cautiously.
===============================================================================================================================
"""

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np
import networkx as nx
from scipy.optimize import minimize
from scipy.spatial import Delaunay
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import matplotlib.pyplot as plt


# ===============================================================================================================================
# SECTION 0: CONFIGURATION
# ===============================================================================================================================

# --- Graph type ------------------------------------------------------------------------------------------------------------------
# IMPORTANT: this differs from Multi_Pair_Capacity_vs_Learning.py -- all graphs sampled here are required to
# be PLANAR (see sample_graph_once), which turned out to rule out half of that script's original six-type
# menu, not just require lower density parameters as first expected. Direct measurement (not just edge-count
# budgeting) is what caught this: erdos_renyi, random_geometric, and waxman all measured 0/100 connected-AND-
# planar hits at N=40 (the actual default size used here) even at densities barely above the connectivity
# threshold -- planarity is a far more fragile property than the textbook "at most 3N-6 edges" bound
# suggests, since forbidden K5/K3,3 minors show up with high probability well before that bound is
# approached, and this gets rapidly worse as N grows (a parameter combination that worked at N=20 can fail
# completely at N=40, which is exactly what happened here). Only graph families that are planar by
# STRUCTURAL GUARANTEE, not just by chance, remain practical:
#   barabasi_albert (m=1): a tree. Trees are always planar, so this is the only value of m that works
#     reliably here -- but a preferential-attachment TREE still has a much more skewed, hub-like degree
#     distribution than a uniformly random tree, so it's a real simplification, not a total loss of interest.
#   watts_strogatz (k=2): a ring lattice with light rewiring is almost always planar (measured 100/100).
#   random_regular (d=2): a union of cycles (measured ~22-31/100, handled fine by the retry loop).
#   delaunay: a NEW type, not in the original six-type menu -- the Delaunay triangulation of random 2D
#     points, which is planar BY CONSTRUCTION (no retry ever needed) rather than by chance. It also sits at
#     the opposite density extreme from the other three (a maximal planar triangulation, right at the 3N-6
#     edge bound), giving real structural diversity in this menu rather than four flavors of "sparse tree-
#     like network." Replaces random_geometric/waxman as this menu's "genuinely spatial" option.
GRAPH_TYPE = "delaunay"  # <-- change this to select which graph family to test

GRAPH_TYPES = ["barabasi_albert", "watts_strogatz", "random_regular", "delaunay"]

GRAPH_PARAMS = {
    "barabasi_albert": {"m": 1},
    "watts_strogatz":   {"k": 2, "p": 0.1},
    "random_regular":   {"d": 2},
    "delaunay":         {},  # no density parameter -- always a maximal planar triangulation
}

# --- I/O placement (unchanged) --------------------------------------------------------------------------------------------------
N_INPUTS = 2  # must be >= 2: the first input is always pinned to ground (0V, see train_single_pair's
              # sample_task), so with only 1 input every training example would be identical (inputs=[0.0]
              # always) and the task would degenerate to learning a constant zero output -- not meaningful.
              # Adjustable per-call via run_output_node_sensitivity_map(n_inputs=...) / show_representative_
              # maps(n_inputs=...), not just this default.
DISTANT_SEARCH_CANDIDATES = 300

# --- Learning parameters (unchanged) ----------------------------------------------------------------------------------------------
STEPS_PER_PAIR = 5
LEARNING_ITERATIONS = 200
MAX_TOTAL_STEPS_FACTOR = 1.0
ETA = 0.05
ALPHA = 0.05
CONV_THRESHOLD = 5e-2
W_MIN = 1e-3  # minimum edge conductance floor -- see train_single_pair

# --- Resistor model: linear or nonlinear, and which nonlinearity + its parameters ------------------------------------------------
RESISTOR_MODEL = "linear"  # options: "linear", "cubic", "saturating", "power_law"

RESISTOR_MODEL_PARAMS = {
    "linear": {},
    "cubic": {"gamma": 0.1},          # see module docstring: gamma < 0 is not guaranteed well-posed
    "saturating": {"V_sat": 1.0},
    "power_law": {"n": 1.5},
}

NEWTON_MAX_ITERS = 200   # generous headroom for strongly nonlinear regimes; costs nothing when convergence is
                          # fast, since scipy's trust-region solver stops as soon as it actually converges
NEWTON_TOL = 1e-9

# --- This analysis's own settings -------------------------------------------------------------------------------------------------
SENSITIVITY_MAP_N = 100          # size of the single graph this analysis is run on
SENSITIVITY_MAP_N_REPEATS = 8   # independent training runs per candidate second-output node

N_OUTPUTS_TOTAL = 1  # total number of output nodes in the task: N_OUTPUTS_TOTAL - 1 are held FIXED
                      # (selected once via distant-clustered placement, same as the inputs), and the
                      # remaining ONE output node is what gets swept across every candidate node for the
                      # sensitivity map. N_OUTPUTS_TOTAL=1 (the default) means zero fixed outputs -- a
                      # single, fully-swept output -- while N_OUTPUTS_TOTAL=2 recovers the original
                      # "1 fixed + 1 swept" setup this script started with.

# Representative-maps panel (see Section 5b): several maps side by side, all on one graph, each with a FRESH
# random I/O node placement -- shows how much the sensitivity map's shape depends on where the fixed nodes sit.
N_MAPS = 3

# Distance-vs-convergence-time analysis (see Section 5c): aggregates (distance-to-inputs, mean convergence
# time) across every candidate node in N_DISTANCE_ANALYSIS_GRAPHS independently sampled graphs, into one
# scatter plot -- a single graph's ~N candidate nodes isn't really enough points for a convincing trend.
N_DISTANCE_ANALYSIS_GRAPHS = 15


# ===============================================================================================================================
# SECTION 1: GRAPH GENERATION
# ===============================================================================================================================

def _delaunay_graph(N, rng):
    """Delaunay triangulation of N random 2D points -- planar BY CONSTRUCTION (the triangulation's edges
    never cross when drawn as the straight lines between the same points used to build it) and connected
    for any reasonable point set, so this needs no retry loop at all. Carries real 2D 'pos' per node (used
    directly by _graph_layout), same as the spatial types it replaces."""
    points = rng.random((N, 2))
    tri = Delaunay(points)
    G = nx.Graph()
    G.add_nodes_from(range(N))
    for simplex in tri.simplices:
        for i in range(3):
            for j in range(i + 1, 3):
                G.add_edge(int(simplex[i]), int(simplex[j]))
    for i in range(N):
        G.nodes[i]["pos"] = (float(points[i, 0]), float(points[i, 1]))
    return G


def sample_graph_once(graph_type, N, rng, params, max_tries=500, require_planar=True):
    """Draw one fresh, independent, CONNECTED graph of the given type and size, retrying with a new seed
    if the draw comes out disconnected. barabasi_albert (as a tree, see GRAPH_PARAMS), watts_strogatz, and
    random_regular don't guarantee connectivity by construction, so the retry matters for them; delaunay
    is connected (and planar) by construction and never needs a retry.

    require_planar=True (the default here) additionally rejects any non-planar draw. See the GRAPH_TYPES
    comment above for why three of the original six graph types (erdos_renyi, random_geometric, waxman)
    were dropped entirely rather than just given lower density parameters -- they measured 0/100
    connected-AND-planar at this script's actual default N, not merely "rare.\""""
    for _ in range(max_tries):
        seed = int(rng.integers(1 << 31))
        if graph_type == "barabasi_albert":
            G = nx.barabasi_albert_graph(N, params["m"], seed=seed)
        elif graph_type == "watts_strogatz":
            G = nx.connected_watts_strogatz_graph(N, params["k"], params["p"], tries=100, seed=seed)
        elif graph_type == "random_regular":
            G = nx.random_regular_graph(params["d"], N, seed=seed)
        elif graph_type == "delaunay":
            G = _delaunay_graph(N, np.random.default_rng(seed))
        else:
            raise ValueError(f"unknown graph_type '{graph_type}' (expected one of {GRAPH_TYPES})")

        if not nx.is_connected(G):
            continue
        if require_planar and not nx.is_planar(G):
            continue
        return G
    raise RuntimeError(f"Could not sample a connected{' + planar' if require_planar else ''} '{graph_type}' "
                        f"graph at N={N} after {max_tries} tries -- consider lowering its density parameter "
                        f"in GRAPH_PARAMS (the values here were tuned by directly measuring hit rate at "
                        f"N=40; a very different N may need re-tuning -- see the GRAPH_TYPES comment above "
                        f"for why this can fail badly rather than just becoming slow).")


def _graph_layout(G, rng):
    """Use the graph's own spatial positions if it has them (delaunay graphs carry a real 2D 'pos' per
    real 2D 'pos' per node); fall back to spring_layout otherwise."""
    if all("pos" in G.nodes[n] for n in G.nodes()):
        return {n: G.nodes[n]["pos"] for n in G.nodes()}
    return nx.spring_layout(G, seed=int(rng.integers(1 << 31)))


# ===============================================================================================================================
# SECTION 2: DISTANT, CLUSTERED I/O PLACEMENT
# ===============================================================================================================================

def select_distant_io(G, n_inputs, n_outputs, n_candidates, rng, mode="max", candidate_nodes=None):
    """Randomly search for a disjoint (input_nodes, output_nodes) split that is far apart, measured by
    mean shortest-path hop distance -- on the full graph G, so distance reflects true physical routing."""
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
    by hop distance (ties broken randomly), excluding anything in `excluded`."""
    dist_from_seed = nx.single_source_shortest_path_length(G, seed)
    candidates = [n for n in dist_from_seed if n != seed and n not in excluded]
    candidates.sort(key=lambda n: (dist_from_seed[n], rng.random()))
    return np.array([seed] + candidates[:cluster_size - 1])


def select_distant_clustered_io(G, n_inputs, n_outputs, n_candidates=DISTANT_SEARCH_CANDIDATES, rng=None,
                                 excluded_nodes=frozenset()):
    """Two-stage distant + clustered placement: pick a single (input_seed, output_seed) pair that's
    maximally far apart, then grow a compact cluster of n_inputs/n_outputs nodes around each seed.

    n_outputs=0 is handled as a special case (needed for N_OUTPUTS_TOTAL=1, i.e. zero fixed outputs):
    there's no output to search for a "distant" input placement against, so the inputs are just clustered
    around a random seed node instead -- still a compact group, just without a "far from anything"
    property, which isn't well-defined without something fixed to be far from.
    """
    if rng is None:
        rng = np.random.default_rng()
    candidate_nodes = [n for n in G.nodes() if n not in excluded_nodes]

    if n_outputs == 0:
        seed_in = int(rng.choice(candidate_nodes))
        input_idx = grow_cluster(G, seed_in, n_inputs, excluded=excluded_nodes, rng=rng)
        return input_idx, np.array([], dtype=int)

    seed_in, seed_out, _ = select_distant_io(G, 1, 1, n_candidates, rng, candidate_nodes=candidate_nodes)
    input_seed, output_seed = int(seed_in[0]), int(seed_out[0])

    input_idx = grow_cluster(G, input_seed, n_inputs, excluded=excluded_nodes | {output_seed}, rng=rng)
    output_idx = grow_cluster(G, output_seed, n_outputs, excluded=excluded_nodes | set(input_idx.tolist()), rng=rng)
    return input_idx, output_idx


# ===============================================================================================================================
# SECTION 3: PLUGGABLE RESISTOR PHYSICS
# ===============================================================================================================================
# Each model supplies three per-unit-weight functions of the edge voltage difference x = deltaV:
#   g(x)       -- current law: edge current = w * g(x)
#   g_prime(x) -- dg/dx, the local slope, used as the effective conductance in the equilibrium solver's Hessian
#   phi0(x)    -- antiderivative of g with phi0(0) = 0, i.e. this edge's stored energy per unit weight; this IS
#                 the quantity whose clamped-vs-free difference drives the weight update (dE/dw_e = phi0(deltaV_e)
#                 exactly, since edge energy is w_e * phi0(deltaV_e) and w_e drops out of its own derivative).
# "linear" recovers the original scripts' physics and update rule exactly (phi0(x) = x^2/2).

def _linear_g(x, p):
    return x

def _linear_g_prime(x, p):
    return np.ones_like(x)

def _linear_phi0(x, p):
    return 0.5 * x ** 2


def _cubic_g(x, p):
    return x + p["gamma"] * x ** 3

def _cubic_g_prime(x, p):
    return 1.0 + 3.0 * p["gamma"] * x ** 2

def _cubic_phi0(x, p):
    return 0.5 * x ** 2 + 0.25 * p["gamma"] * x ** 4


def _saturating_g(x, p):
    return np.tanh(x / p["V_sat"])

def _saturating_g_prime(x, p):
    t = np.tanh(x / p["V_sat"])
    return (1.0 - t ** 2) / p["V_sat"]

def _saturating_phi0(x, p):
    # Numerically stable log(cosh(z)): computing cosh(z) directly overflows for large z.
    V_sat = p["V_sat"]
    z = np.abs(x) / V_sat
    log_cosh = z + np.log1p(np.exp(-2 * z)) - np.log(2.0)
    return V_sat * log_cosh


def _power_law_g(x, p):
    return np.sign(x) * np.abs(x) ** p["n"]

def _power_law_g_prime(x, p):
    # Small-epsilon regularization: n*|x|^(n-1) is genuinely singular at x=0 for n<1 (infinite slope) and
    # exactly zero at x=0 for n>1 (which combined with several such edges can ill-condition the local
    # Hessian) -- (|x|+eps) keeps this finite and positive everywhere without materially changing the value
    # away from x=0.
    n = p["n"]
    eps = p.get("eps", 1e-6)
    return n * (np.abs(x) + eps) ** (n - 1)

def _power_law_phi0(x, p):
    return np.abs(x) ** (p["n"] + 1) / (p["n"] + 1)


RESISTOR_MODELS = {
    "linear":     {"g": _linear_g,     "g_prime": _linear_g_prime,     "phi0": _linear_phi0},
    "cubic":      {"g": _cubic_g,      "g_prime": _cubic_g_prime,      "phi0": _cubic_phi0},
    "saturating": {"g": _saturating_g, "g_prime": _saturating_g_prime, "phi0": _saturating_phi0},
    "power_law":  {"g": _power_law_g,  "g_prime": _power_law_g_prime, "phi0": _power_law_phi0},
}


def solve_equilibrium(Hessian, fixed_idx, fixed_vals, N_Nodes):
    """Direct linear solve for the node-voltage equilibrium given a set of clamped node values. Used
    directly for the "linear" resistor model (fast path, no iteration needed), and internally as the
    initial guess for the nonlinear solver below."""
    fixed_idx = np.asarray(fixed_idx, dtype=int)
    fixed_vals = np.asarray(fixed_vals, dtype=float)
    free_idx = np.setdiff1d(np.arange(N_Nodes), fixed_idx)
    H_ff = Hessian[np.ix_(free_idx, free_idx)]
    H_fc = Hessian[np.ix_(free_idx, fixed_idx)]
    V = np.zeros(N_Nodes)
    V[fixed_idx] = fixed_vals
    V[free_idx] = np.linalg.solve(H_ff, -H_fc @ fixed_vals)
    return V


def solve_equilibrium_general(wVec, Delta_Learning, fixed_idx, fixed_vals, N_Nodes,
                               resistor_model=RESISTOR_MODEL, model_params=None,
                               max_iters=NEWTON_MAX_ITERS, tol=NEWTON_TOL):
    """
    Equilibrium node voltages under whichever resistor model is selected. "linear" is a single direct
    linear solve (see solve_equilibrium). The three nonlinear models are solved as a convex minimization
    of the network's total energy over the free-node voltages, via scipy's trust-region Newton-CG, using
    the model's exact gradient (net current at each free node) and exact Hessian (local effective-
    conductance weighted Laplacian).

    A hand-rolled plain-Newton-with-backtracking version of this was tried first (for the saturating model)
    and diverged or stalled on harder cases: once several edges saturate, the local Hessian becomes
    severely ill-conditioned, which fixed damping/backtracking handles poorly. A trust-region method adapts
    both step size AND direction based on how reliable the local quadratic model currently is, which is
    exactly what's needed here -- so this uses scipy's mature implementation rather than re-deriving that
    robustness by hand for every nonlinearity.
    """
    model_params = model_params if model_params is not None else RESISTOR_MODEL_PARAMS[resistor_model]

    fixed_idx = np.asarray(fixed_idx, dtype=int)
    fixed_vals = np.asarray(fixed_vals, dtype=float)
    free_idx = np.setdiff1d(np.arange(N_Nodes), fixed_idx)

    if resistor_model == "linear":
        Hessian = Delta_Learning.T @ np.diag(wVec) @ Delta_Learning
        return solve_equilibrium(Hessian, fixed_idx, fixed_vals, N_Nodes)

    g_fn = RESISTOR_MODELS[resistor_model]["g"]
    g_prime_fn = RESISTOR_MODELS[resistor_model]["g_prime"]
    phi0_fn = RESISTOR_MODELS[resistor_model]["phi0"]

    Delta_free = Delta_Learning[:, free_idx]
    baseline = Delta_Learning[:, fixed_idx] @ fixed_vals  # each edge's deltaV contribution from fixed nodes

    def energy(V_free):
        deltaV = Delta_free @ V_free + baseline
        return float(np.sum(wVec * phi0_fn(deltaV, model_params)))

    def gradient(V_free):
        deltaV = Delta_free @ V_free + baseline
        currents = wVec * g_fn(deltaV, model_params)
        return Delta_free.T @ currents

    def hessian(V_free):
        deltaV = Delta_free @ V_free + baseline
        w_eff = wVec * np.maximum(g_prime_fn(deltaV, model_params), 1e-9)  # keep conditioning safe universally
        return Delta_free.T @ (w_eff[:, None] * Delta_free)

    # Initial guess: plain linear solve using the raw weights (ignores the nonlinearity, but is a
    # reasonable universal starting point regardless of which model is selected, and free since it reuses
    # the direct linear solver).
    Hessian_lin = Delta_Learning.T @ np.diag(wVec) @ Delta_Learning
    V0 = solve_equilibrium(Hessian_lin, fixed_idx, fixed_vals, N_Nodes)

    result = minimize(energy, V0[free_idx], jac=gradient, hess=hessian, method="trust-ncg",
                       options={"gtol": tol, "maxiter": max_iters})

    V = np.zeros(N_Nodes)
    V[fixed_idx] = fixed_vals
    V[free_idx] = result.x
    return V


# ===============================================================================================================================
# SECTION 4: SINGLE-PAIR CONTRASTIVE LEARNING (2 inputs, 2 outputs, one output fixed and one swept)
# ===============================================================================================================================

def train_single_pair(Delta_Learning, N_Nodes, N_edges, input_idx, output_idx, rng,
                       resistor_model=RESISTOR_MODEL, model_params=None,
                       steps_per_pair=STEPS_PER_PAIR, eta=ETA, alpha=ALPHA,
                       conv_threshold=CONV_THRESHOLD, max_total_steps=None):
    """
    Continuous contrastive-learning training on a single fixed input/output node assignment (this script
    only ever has one pair -- the K-pairs round-robin machinery from Multi_Pair_Capacity_vs_Learning.py
    would be equivalent but is unneeded complexity here). Training proceeds in blocks of steps_per_pair
    weight updates, followed by an evaluation sweep (measuring error with NO further training that step) --
    kept identical in structure to the multi-pair script's convergence check, for comparability, even
    though there's nothing to interfere with this pair here.

    The weight-update rule is derived from the resistor model's own energy potential (see
    RESISTOR_MODELS/phi0), so it's the correct nonlinear generalization of the original contrastive rule,
    not a rule swapped in by analogy -- and is provably identical to the original rule when
    resistor_model="linear".
    """
    model_params = model_params if model_params is not None else RESISTOR_MODEL_PARAMS[resistor_model]
    phi0_fn = RESISTOR_MODELS[resistor_model]["phi0"]

    if max_total_steps is None:
        max_total_steps = int(MAX_TOTAL_STEPS_FACTOR * LEARNING_ITERATIONS)

    n_in, n_out = len(input_idx), len(output_idx)
    target_matrix = rng.random((n_out, n_in))
    target_matrix /= target_matrix.sum(axis=1, keepdims=True)
    # Nonnegative rows summing to 1 -- a convex combination, guaranteeing every output lands within
    # [min(inputs), max(inputs)] for ANY inputs, so the target can never ask for a physically unreachable
    # output regardless of which resistor model is in play (see the design discussion this was fixed from).
    assert np.all(target_matrix >= 0) and np.allclose(target_matrix.sum(axis=1), 1.0)

    def sample_task():
        inputs = rng.random(n_in) * 10.0
        inputs[0] = 0.0  # first input pinned near 0 (ground-like), matching prior scripts' convention
        outputs = target_matrix @ inputs
        return inputs, outputs

    def evaluate():
        inputs, outputs = sample_task()
        NodeValsFree = solve_equilibrium_general(wVec, Delta_Learning, input_idx, inputs, N_Nodes,
                                                   resistor_model, model_params)
        return float(np.sqrt(np.mean((NodeValsFree[output_idx] - outputs) ** 2)))

    wVec = rng.random(N_edges) + 0.5
    step_count = 0
    round_num = 0

    while step_count < max_total_steps:
        round_num += 1
        for _ in range(steps_per_pair):
            inputs, outputs = sample_task()
            NodeValsFree = solve_equilibrium_general(wVec, Delta_Learning, input_idx, inputs, N_Nodes,
                                                       resistor_model, model_params)

            clamped_targets = eta * outputs + (1 - eta) * NodeValsFree[output_idx]
            clamped_idx = np.concatenate([input_idx, output_idx])
            clamped_vals = np.concatenate([inputs, clamped_targets])
            NodeValsClamped = solve_equilibrium_general(wVec, Delta_Learning, clamped_idx, clamped_vals, N_Nodes,
                                                          resistor_model, model_params)

            dPhiF = phi0_fn(Delta_Learning @ NodeValsFree, model_params)
            dPhiC = phi0_fn(Delta_Learning @ NodeValsClamped, model_params)
            wVec = wVec - (alpha / eta) * (dPhiC - dPhiF)
            wVec = np.maximum(wVec, W_MIN)  # keep conductances positive
            step_count += 1

            if not np.all(np.isfinite(wVec)):
                return {"T_conv_total": np.nan, "converged": False, "n_rounds": round_num, "blew_up": True}

            if step_count >= max_total_steps:
                break

        if evaluate() < conv_threshold:
            return {"T_conv_total": step_count, "converged": True, "n_rounds": round_num, "blew_up": False}

    return {"T_conv_total": np.nan, "converged": False, "n_rounds": round_num, "blew_up": False}


# ===============================================================================================================================
# SECTION 5: OUTPUT-NODE CONVERGENCE-TIME SENSITIVITY MAP
# ===============================================================================================================================

def _evaluate_candidate_output_node(Delta_Learning, N_Nodes, N_edges, fixed_input_idx, fixed_output_idx,
                                     candidate_node, n_repeats, resistor_model, model_params):
    """One candidate output-node's worth of work: run n_repeats independent training runs with output
    nodes [*fixed_output_idx, candidate_node] (fixed_output_idx may be empty -- see N_OUTPUTS_TOTAL), and
    aggregate. Designed to be called independently so candidates can be parallelized across processes."""
    rng = np.random.default_rng()
    input_idx = np.asarray(fixed_input_idx)
    output_idx = np.concatenate([np.asarray(fixed_output_idx, dtype=int), [candidate_node]])

    run_results = [
        train_single_pair(Delta_Learning, N_Nodes, N_edges, input_idx, output_idx, rng,
                           resistor_model=resistor_model, model_params=model_params)
        for _ in range(n_repeats)
    ]
    converged_runs = [r for r in run_results if r["converged"]]
    T_conv_vals = np.array([r["T_conv_total"] for r in converged_runs], dtype=float)

    return {
        "candidate_node": candidate_node,
        "T_conv_mean": float(np.mean(T_conv_vals)) if len(T_conv_vals) else np.nan,
        "T_conv_std": float(np.std(T_conv_vals)) if len(T_conv_vals) > 1 else 0.0,
        "convergence_rate": len(converged_runs) / n_repeats,
    }


def plot_node_sensitivity_map(G, fixed_input_idx, fixed_output_idx, node_results, pos, ax=None, title=None,
                               cmap="viridis_r"):
    """
    Draw the graph with every candidate node colored by its mean time-to-convergence as the swept output
    (cmap="viridis_r" so fast/short times are bright). Candidates that NEVER converged within budget in any
    repeat are drawn as black X markers rather than folded into the color scale -- "did not converge" and
    "converged slowly" are qualitatively different outcomes a single color gradient would conflate. The
    fixed inputs (blue circles) and any fixed outputs (red squares -- there can be zero of these, see
    N_OUTPUTS_TOTAL) are drawn distinctly on top.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7.5, 7.5))

    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="lightgray", width=1.0, alpha=0.6)

    candidates = list(node_results.keys())
    T_conv = np.array([node_results[n]["T_conv_mean"] for n in candidates])
    valid = ~np.isnan(T_conv)

    if valid.any():
        valid_nodes = [n for n, ok in zip(candidates, valid) if ok]
        sc = ax.scatter([pos[n][0] for n in valid_nodes], [pos[n][1] for n in valid_nodes],
                         c=T_conv[valid], cmap=cmap, s=180, edgecolors="black", linewidths=0.8, zorder=3)
        cbar = plt.colorbar(sc, ax=ax, shrink=0.8)
        cbar.set_label("Mean steps to convergence")

    failed_nodes = [n for n, ok in zip(candidates, valid) if not ok]
    if failed_nodes:
        ax.scatter([pos[n][0] for n in failed_nodes], [pos[n][1] for n in failed_nodes],
                    c="black", marker="x", s=110, linewidths=2.2, zorder=3, label="Never converged")

    ax.scatter([pos[n][0] for n in fixed_input_idx], [pos[n][1] for n in fixed_input_idx],
               c="blue", marker="o", s=260, edgecolors="black", linewidths=1.5, zorder=4, label="Input (fixed)")
    fixed_output_idx = np.asarray(fixed_output_idx, dtype=int)
    if len(fixed_output_idx) > 0:
        ax.scatter([pos[n][0] for n in fixed_output_idx], [pos[n][1] for n in fixed_output_idx],
                   c="red", marker="s", s=260, edgecolors="black", linewidths=1.5, zorder=4, label="Output (fixed)")

    ax.set_title(title or "Convergence-time sensitivity to swept output node's location")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.set_axis_off()
    return ax


def _sweep_candidate_outputs(G, Delta_Learning, N_edges, fixed_input_idx, fixed_output_idx, n_repeats,
                              resistor_model, model_params, desc="Candidate output nodes"):
    """Shared core of the sensitivity-map computation: given a graph and a FIXED (input, output) node
    placement, sweep every remaining node as the candidate swept output, parallelized across processes.
    Used by both run_output_node_sensitivity_map (one map) and show_representative_maps (several)."""
    N_Nodes = G.number_of_nodes()
    excluded = set(fixed_input_idx.tolist()) | set(np.asarray(fixed_output_idx, dtype=int).tolist())
    candidate_nodes = [n for n in G.nodes() if n not in excluded]

    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(_evaluate_candidate_output_node, Delta_Learning, N_Nodes, N_edges,
                             fixed_input_idx, fixed_output_idx, cand, n_repeats, resistor_model, model_params)
            for cand in candidate_nodes
        ]
        raw = [f.result() for f in tqdm(as_completed(futures), total=len(futures), desc=desc, unit="node")]

    return {r["candidate_node"]: r for r in raw}


def run_output_node_sensitivity_map(graph_type=GRAPH_TYPE, params=None, N=SENSITIVITY_MAP_N,
                                     n_repeats=SENSITIVITY_MAP_N_REPEATS, n_inputs=N_INPUTS,
                                     n_outputs_total=N_OUTPUTS_TOTAL, resistor_model=RESISTOR_MODEL,
                                     model_params=None, seed=None):
    """
    Full pipeline for ONE map: sample one graph, fix n_inputs input nodes + (n_outputs_total - 1) output
    nodes (distant, clustered placement; n_outputs_total=1 means zero fixed outputs), sweep every
    remaining node as the candidate swept output, and plot the resulting sensitivity map. Returns
    (G, fixed_input_idx, fixed_output_idx, node_results) in case you want to inspect the raw numbers (also
    printed, sorted best-to-worst). See show_representative_maps for a multi-map panel instead of just one.
    """
    assert n_inputs >= 2, (
        f"n_inputs must be >= 2 (got {n_inputs}): the first input is always pinned to ground (see "
        f"train_single_pair's sample_task), so with only 1 input every training example would be "
        f"identical and the task would degenerate to learning a constant zero output."
    )
    params = params if params is not None else GRAPH_PARAMS[graph_type]
    model_params = model_params if model_params is not None else RESISTOR_MODEL_PARAMS[resistor_model]
    rng = np.random.default_rng(seed)

    G = sample_graph_once(graph_type, N, rng, params)
    fixed_input_idx, fixed_output_idx = select_distant_clustered_io(G, n_inputs=n_inputs,
                                                                      n_outputs=n_outputs_total - 1, rng=rng)
    Delta_Learning = nx.incidence_matrix(G, oriented=True).toarray().T
    N_edges = G.number_of_edges()

    print(f"Resistor model: {resistor_model}  (params: {model_params})")
    print(f"Sensitivity map: fixed inputs={list(fixed_input_idx)}, fixed outputs={list(fixed_output_idx)}, "
          f"sweeping {G.number_of_nodes() - n_inputs - len(fixed_output_idx)} candidate output nodes...")

    node_results = _sweep_candidate_outputs(G, Delta_Learning, N_edges, fixed_input_idx, fixed_output_idx,
                                             n_repeats, resistor_model, model_params)

    ranked = sorted(node_results.items(), key=lambda kv: (np.isnan(kv[1]["T_conv_mean"]), kv[1]["T_conv_mean"]))
    print("\nCandidate output nodes, fastest to slowest (or never-converged, listed last):")
    for node, r in ranked:
        t = "never converged" if np.isnan(r["T_conv_mean"]) else f"{r['T_conv_mean']:.1f} steps"
        print(f"  node {node}: {t}  (convergence rate {r['convergence_rate']:.0%})")

    pos = _graph_layout(G, rng)
    plot_node_sensitivity_map(G, fixed_input_idx, fixed_output_idx, node_results, pos,
                               title=f"Sensitivity map ({graph_type}, N={N}, {resistor_model} resistors)")

    return G, fixed_input_idx, fixed_output_idx, node_results


# ===============================================================================================================================
# SECTION 5b: REPRESENTATIVE MAPS PANEL
# ===============================================================================================================================
# One graph, several maps side by side, each with a FRESH random input/output node placement -- shows how much
# the sensitivity map's shape depends on where the fixed nodes happen to sit. (An earlier version of this also
# showed repeated maps for the SAME fixed placement, to check whether n_repeats was large enough for a stable
# map -- removed as redundant with just raising n_repeats directly: it answers "is there noise" without doing
# anything to reduce it, whereas increasing n_repeats does both more directly.)

def show_representative_maps(graph_type=GRAPH_TYPE, params=None, N=SENSITIVITY_MAP_N,
                              n_repeats=SENSITIVITY_MAP_N_REPEATS, n_inputs=N_INPUTS,
                              n_outputs_total=N_OUTPUTS_TOTAL, resistor_model=RESISTOR_MODEL, model_params=None,
                              n_maps=N_MAPS, seed=None):
    """Produce a panel of n_maps sensitivity maps, all on ONE freshly sampled graph, each with its own
    freshly chosen (random) input/output node placement. Returns the list of (fixed_input_idx,
    fixed_output_idx, node_results) tuples, one per map, in the order plotted."""
    assert n_inputs >= 2, (
        f"n_inputs must be >= 2 (got {n_inputs}): the first input is always pinned to ground (see "
        f"train_single_pair's sample_task), so with only 1 input every training example would be "
        f"identical and the task would degenerate to learning a constant zero output."
    )
    params = params if params is not None else GRAPH_PARAMS[graph_type]
    model_params = model_params if model_params is not None else RESISTOR_MODEL_PARAMS[resistor_model]
    rng = np.random.default_rng(seed)

    G = sample_graph_once(graph_type, N, rng, params)
    Delta_Learning = nx.incidence_matrix(G, oriented=True).toarray().T
    N_edges = G.number_of_edges()
    pos = _graph_layout(G, rng)

    print(f"Resistor model: {resistor_model}  (params: {model_params})")

    fig, axes = plt.subplots(1, n_maps, figsize=(6.5 * n_maps, 6.5), squeeze=False)
    axes = axes[0]

    results = []
    for j in range(n_maps):
        fixed_input_idx, fixed_output_idx = select_distant_clustered_io(G, n_inputs=n_inputs,
                                                                          n_outputs=n_outputs_total - 1, rng=rng)
        print(f"\n[Map {j + 1}/{n_maps}] inputs={list(fixed_input_idx)}, outputs={list(fixed_output_idx)}")
        node_results = _sweep_candidate_outputs(G, Delta_Learning, N_edges, fixed_input_idx, fixed_output_idx,
                                                  n_repeats, resistor_model, model_params,
                                                  desc=f"Map {j + 1}/{n_maps}")
        plot_node_sensitivity_map(G, fixed_input_idx, fixed_output_idx, node_results, pos,
                                   ax=axes[j], title=f"Map {j + 1}/{n_maps}")
        results.append((fixed_input_idx, fixed_output_idx, node_results))

    fig.suptitle(f"Representative sensitivity maps ({graph_type}, N={N}, {resistor_model} resistors)", fontsize=14)
    plt.tight_layout()

    return results


# ===============================================================================================================================
# SECTION 5c: CONVERGENCE TIME VS. DISTANCE TO INPUTS
# ===============================================================================================================================
# Aggregates every candidate node's (distance, mean convergence time) across N_DISTANCE_ANALYSIS_GRAPHS
# independently sampled planar graphs into one scatter plot, to see whether nodes farther from the inputs
# take longer to converge as the swept output.
#
# DISTANCE METRIC (as specified): for a candidate node, distance = (minimum hop count to whichever input
# node is closer) + (hop count between the two input nodes themselves). This is hop count, not effective
# resistance -- unlike effective resistance, it ignores parallel-path redundancy entirely, so it's a cruder
# but more direct "how many relay steps away" measure. Only well-defined for exactly 2 input nodes ("either
# one of the input nodes" doesn't generalize unambiguously to more), so n_inputs is fixed at 2 here.

def hop_distance_to_inputs(G, fixed_input_idx, candidate_node):
    """distance = min hop count from candidate_node to either input, plus the hop count between the two
    inputs themselves. Requires exactly 2 input nodes."""
    i1, i2 = int(fixed_input_idx[0]), int(fixed_input_idx[1])
    d_to_nearest = min(nx.shortest_path_length(G, candidate_node, i1),
                        nx.shortest_path_length(G, candidate_node, i2))
    d_between_inputs = nx.shortest_path_length(G, i1, i2)
    return d_to_nearest + d_between_inputs


def run_distance_vs_convergence_analysis(graph_type=GRAPH_TYPE, params=None, N=SENSITIVITY_MAP_N,
                                          n_repeats=SENSITIVITY_MAP_N_REPEATS, n_outputs_total=N_OUTPUTS_TOTAL,
                                          resistor_model=RESISTOR_MODEL, model_params=None,
                                          n_graphs=N_DISTANCE_ANALYSIS_GRAPHS, seed=None):
    """
    Sample n_graphs independent planar graphs; for each, fix 2 input nodes (+ n_outputs_total-1 output
    nodes), sweep every remaining candidate node as before, and record (hop_distance_to_inputs,
    T_conv_mean) for every candidate across every graph. Plots the combined scatter with a binned
    mean +/- SEM trend line overlaid (raw scatter alone is hard to read since integer hop distances repeat
    heavily), and reports the Pearson correlation. Returns (distances, T_conv_means) arrays (converged
    candidates only -- see printed count of excluded never-converged candidates).
    """
    params = params if params is not None else GRAPH_PARAMS[graph_type]
    model_params = model_params if model_params is not None else RESISTOR_MODEL_PARAMS[resistor_model]
    rng = np.random.default_rng(seed)

    print(f"Resistor model: {resistor_model}  (params: {model_params})")

    distances, T_convs = [], []
    n_excluded = 0

    for g in range(n_graphs):
        G = sample_graph_once(graph_type, N, rng, params)
        fixed_input_idx, fixed_output_idx = select_distant_clustered_io(G, n_inputs=2,
                                                                          n_outputs=n_outputs_total - 1, rng=rng)
        Delta_Learning = nx.incidence_matrix(G, oriented=True).toarray().T
        N_edges = G.number_of_edges()

        node_results = _sweep_candidate_outputs(G, Delta_Learning, N_edges, fixed_input_idx, fixed_output_idx,
                                                  n_repeats, resistor_model, model_params,
                                                  desc=f"Graph {g + 1}/{n_graphs}")

        for candidate, r in node_results.items():
            dist = hop_distance_to_inputs(G, fixed_input_idx, candidate)
            if np.isnan(r["T_conv_mean"]):
                n_excluded += 1
                continue
            distances.append(dist)
            T_convs.append(r["T_conv_mean"])

    distances = np.array(distances, dtype=float)
    T_convs = np.array(T_convs, dtype=float)

    print(f"\n{len(distances)} converged candidate points across {n_graphs} graphs "
          f"({n_excluded} never-converged candidates excluded).")

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    ax.scatter(distances, T_convs, alpha=0.35, edgecolors="black", linewidths=0.3, label="Individual candidate nodes")

    # Binned mean +/- SEM trend line: distances are integer hop counts that repeat heavily, so a raw
    # scatter alone is hard to read as a trend -- this makes the actual pattern visible on top of it.
    unique_d = np.unique(distances)
    bin_means = np.array([T_convs[distances == d].mean() for d in unique_d])
    bin_sems = np.array([
        T_convs[distances == d].std() / np.sqrt((distances == d).sum()) if (distances == d).sum() > 1 else 0.0
        for d in unique_d
    ])
    ax.errorbar(unique_d, bin_means, yerr=bin_sems, marker="o", color="red", capsize=3, linewidth=2,
                label="Mean +/- SEM per distance")

    if len(distances) > 2 and np.std(distances) > 0:
        corr = float(np.corrcoef(distances, T_convs)[0, 1])
        slope, intercept = np.polyfit(distances, T_convs, 1)
        xs = np.linspace(distances.min(), distances.max(), 50)
        ax.plot(xs, slope * xs + intercept, "k--", linewidth=1.5, label=f"Linear fit (Pearson r={corr:.3f})")
        print(f"Pearson correlation (distance vs. mean convergence time): {corr:.3f}")

    ax.set_xlabel("Distance to inputs (hops to nearer input + hops between inputs)")
    ax.set_ylabel("Mean steps to convergence")
    ax.set_title(f"Convergence time vs. distance to inputs\n({graph_type}, N={N}, {n_graphs} graphs, "
                 f"{resistor_model} resistors)")
    ax.legend(fontsize=8)
    plt.tight_layout()

    return distances, T_convs


# ===============================================================================================================================
# SECTION 6: RUN
# ===============================================================================================================================

if __name__ == "__main__":
    show_representative_maps()
    run_distance_vs_convergence_analysis()
    plt.show()

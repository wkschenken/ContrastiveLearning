"""
COMPARE FVS METHODS: exact ILP solver vs. the original heuristic (multi-spanning-tree) search.
===============================================================================================================================
For a batch of randomly sampled graphs, this script computes the Feedback Vertex Set (FVS) two ways:

  1. EXACT  -- the cutting-plane ILP solver (min_fvs_exact), which provably finds the true minimum FVS
              (or falls back honestly if it can't certify optimality within its time budget).
  2. HEURISTIC -- your original method: repeatedly (a) build a random spanning tree, (b) find the "chord"
              edges (the ones not in the tree -- removing all of them makes the graph acyclic), and
              (c) iteratively delete the highest-chord-degree vertex until no chord edges remain. Repeat
              over many random spanning trees and keep the smallest FVS found. (This is algorithmically
              identical to what your original script did -- rewritten here with an adjacency-list
              representation + incremental degree tracking for speed, but the procedure itself, and
              therefore what it's capable of finding, is unchanged.)

It then reports, per graph and in aggregate: how often the heuristic actually finds the true optimum, how
large the gap is when it doesn't, and how the two methods' runtimes compare.
===============================================================================================================================
"""

import time
import numpy as np
import networkx as nx
import pandas as pd
import pulp
import matplotlib.pyplot as plt


# ===============================================================================================================================
# CONFIGURATION
# ===============================================================================================================================

N_NODES = 100             # nodes per test graph (matches the main learning-vs-graph-properties script)
P_EDGE = 0.55             # Erdos-Renyi edge probability
N_TEST_GRAPHS = 30        # number of graphs to compare methods on
                          # NOTE: the exact ILP solver can take anywhere from ~1s to ~60s per graph, so
                          # this script can take several minutes to run at N_TEST_GRAPHS=30. Lower this
                          # for a quicker check, or raise it (and consider parallelizing per-graph, as in
                          # the main script) for a more statistically solid comparison.

HEURISTIC_NUM_TREES = 1000     # number of random spanning trees tried per graph (same as the original script)

ILP_TIME_LIMIT_PER_SOLVE = 5
ILP_MAX_CUTS = 60
ILP_MAX_TOTAL_TIME = 60


# ===============================================================================================================================
# METHOD 1: EXACT FVS VIA CUTTING-PLANE ILP
# ===============================================================================================================================

def min_fvs_exact(G, time_limit_per_solve=ILP_TIME_LIMIT_PER_SOLVE, max_cuts=ILP_MAX_CUTS,
                   max_total_time=ILP_MAX_TOTAL_TIME):
    """
    Exact minimum FVS via branch-and-cut: minimize sum(x_v) subject to "every cycle has at least one
    x_v = 1", with cycle constraints added lazily (one per currently-uncovered cycle, every round) as
    they're discovered, rather than listed exhaustively up front. See the parent conversation / main
    script for the full explanation. Returns (fvs_set, exact_flag).
    """
    nodes = list(G.nodes())
    prob = pulp.LpProblem("MinFVS", pulp.LpMinimize)
    x = {v: pulp.LpVariable(f"x_{v}", cat="Binary") for v in nodes}
    prob += pulp.lpSum(x.values())

    t_start = time.time()
    removed = set(nodes)
    for _ in range(max_cuts):
        if time.time() - t_start > max_total_time:
            return removed, False
        prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=time_limit_per_solve))
        status_optimal = pulp.LpStatus[prob.status] == "Optimal"
        removed = {v for v in nodes if x[v].value() is not None and x[v].value() > 0.5}
        remaining = [v for v in nodes if v not in removed]
        cycles = nx.cycle_basis(G.subgraph(remaining))
        if not cycles:
            return removed, status_optimal
        for cyc in cycles:
            prob += pulp.lpSum(x[v] for v in cyc) >= 1
    return removed, False


# ===============================================================================================================================
# METHOD 2: HEURISTIC FVS VIA RANDOM SPANNING TREES (your original method)
# ===============================================================================================================================

def heuristic_fvs_size(G, num_trees=HEURISTIC_NUM_TREES, rng=None):
    """
    Approximate FVS: for each of `num_trees` random spanning trees, find the FVS implied by that tree
    (delete the highest chord-degree vertex repeatedly until no chord edges remain), and keep the
    smallest FVS found across all trials. This is the same algorithm as your original script's
    NumTrees-loop; only the bookkeeping (adjacency lists + incremental degree updates instead of
    repeatedly recomputing an incidence-matrix product) has changed, for speed.
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


# ===============================================================================================================================
# RUN THE COMPARISON
# ===============================================================================================================================

if __name__ == "__main__":

    rng = np.random.default_rng()
    rows = []

    for i in range(N_TEST_GRAPHS):
        while True:
            G = nx.gnp_random_graph(N_NODES, P_EDGE, seed=int(rng.integers(1 << 31)))
            if nx.is_connected(G):
                break

        n_edges = G.number_of_edges()
        cycle_size = len(nx.cycle_basis(G))

        t0 = time.time()
        exact_removed, exact_flag = min_fvs_exact(G)
        exact_time = time.time() - t0
        exact_size = len(exact_removed)

        t0 = time.time()
        heuristic_size = heuristic_fvs_size(G, rng=rng)
        heuristic_time = time.time() - t0

        gap = heuristic_size - exact_size
        print(f"graph {i+1}/{N_TEST_GRAPHS}: edges={n_edges}  "
              f"exact={exact_size} (verified={exact_flag}, {exact_time:.2f}s)  "
              f"heuristic={heuristic_size} ({heuristic_time:.2f}s)  gap={gap}")

        rows.append({
            "graph_index": i,
            "n_edges": n_edges,
            "cycle_size": cycle_size,
            "fvs_exact": exact_size,
            "exact_verified": exact_flag,
            "exact_time_sec": exact_time,
            "fvs_heuristic": heuristic_size,
            "heuristic_time_sec": heuristic_time,
            "gap": gap,
        })

    df = pd.DataFrame(rows)
    df.to_csv("fvs_method_comparison.csv", index=False)

    # ---------------------------------------------------------------------------------------------------------------------
    # SUMMARY STATISTICS
    # ---------------------------------------------------------------------------------------------------------------------
    n_verified = int(df["exact_verified"].sum())
    n_exact_match = int((df["gap"] == 0).sum())

    print("\n" + "=" * 100)
    print(f"Graphs tested: {len(df)}   (exact solver certified optimal on {n_verified}/{len(df)})")
    print(f"Heuristic matched the exact optimum on {n_exact_match}/{len(df)} graphs "
          f"({100 * n_exact_match / len(df):.1f}%)")
    print(f"Mean gap (heuristic - exact): {df['gap'].mean():.2f}  "
          f"(median {df['gap'].median():.1f}, max {df['gap'].max()})")
    print(f"Mean exact-solver time:     {df['exact_time_sec'].mean():.2f}s per graph")
    print(f"Mean heuristic time:        {df['heuristic_time_sec'].mean():.3f}s per graph")
    print(f"Exact solver is on average {df['exact_time_sec'].mean() / df['heuristic_time_sec'].mean():.0f}x "
          f"slower than the heuristic, in exchange for a guaranteed-optimal answer.")
    print("=" * 100)

    # ---------------------------------------------------------------------------------------------------------------------
    # PLOTS
    # ---------------------------------------------------------------------------------------------------------------------

    # 1. Heuristic vs. exact FVS size, with y=x reference line (points above the line = heuristic overestimated)
    plt.figure()
    max_val = max(df["fvs_exact"].max(), df["fvs_heuristic"].max()) + 1
    plt.plot([0, max_val], [0, max_val], 'k--', linewidth=1, label="y = x (perfect agreement)")
    plt.scatter(df["fvs_exact"], df["fvs_heuristic"], alpha=0.7)
    plt.xlabel("Exact FVS size")
    plt.ylabel("Heuristic FVS size")
    plt.title("Heuristic vs. exact FVS size (points above the line = heuristic overestimated)")
    plt.legend()

    # 2. Histogram of the gap (heuristic - exact)
    plt.figure()
    plt.hist(df["gap"], bins=range(int(df["gap"].min()), int(df["gap"].max()) + 2), edgecolor='black')
    plt.xlabel("Gap (heuristic FVS size - exact FVS size)")
    plt.ylabel("Number of graphs")
    plt.title("Distribution of heuristic overestimation")

    # 3. Runtime comparison (log scale, since the exact solver is typically orders of magnitude slower)
    plt.figure()
    plt.boxplot([df["exact_time_sec"], df["heuristic_time_sec"]], tick_labels=["Exact (ILP)", "Heuristic"])
    plt.yscale("log")
    plt.ylabel("Time per graph (s, log scale)")
    plt.title("Runtime: exact solver vs. heuristic search")

    plt.show()

    print("\nFull per-graph comparison saved to fvs_method_comparison.csv")

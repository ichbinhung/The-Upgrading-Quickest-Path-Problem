"""
Upgrading Quickest Path Problem (UQPP) — Computational Experiments.

This module implements and benchmarks the decomposition algorithms introduced in:

    "The Upgrading Quickest Path Problem on Networks under Linear and
     Weighted Sum Hamming Costs"

The UQPP seeks an optimal capacity-upgrade strategy x for a network G=(V,E)
so that the quickest path transmission time T(P) = L(P) + sigma / C(P) is
minimised subject to a budget B. Two cost structures are studied:

    - UQPP_1  (linear cost)    : C_1(x) = sum_e  w(e) * x_e  (Section 2.2)
    - UQPP_H  (Hamming cost)   : C_H(x) = sum_e  w(e) * H(x_e)  (Section 2.3)
    - UQPP_HC (cardinality)    : special case of UQPP_H with w(e)=1, budget k

Function-to-paper mapping
--------------------------
generate_graph                  : random graph instances (Section 4, Exp. Setup)
generate_graph_controlled_K     : instances with controlled |K| (Section 4)
get_k_size                      : computes |K|, the critical capacity set size
fast_larac_l1                   : LARAC oracle O_RSPP for UQPP_1 (Section 2.2)
fast_larac_hw                   : LARAC oracle O_RSPP for UQPP_H (Section 2.3)
solve_linear_decomposition      : DA-L1 — Algorithm 1 (Section 2.2)
solve_weighted_hamming_decomp.  : DA-HW — Algorithm 2 (Section 2.3, Case 1)
solve_cardinality_layered       : DA-HC — Algorithm 3 (Section 2.3, Case 2)
solve_mip_miqcp_l1              : Gurobi MIQCP benchmark for UQPP_1 (Appendix)
solve_mip_milp_hamming          : Gurobi MILP benchmark for UQPP_H (Appendix)
run_part1_1 / run_part1_2       : Table 1 experiments — UQPP_1 scaling
run_part2                       : Table 2 experiments — UQPP_H scaling
run_part3                       : Table 3 experiments — cardinality UQPP
run_part4                       : Table 4 experiments — sensitivity to sigma

Usage
-----
    python updgrading_quickest_path.py
"""

import networkx as nx
import random
import time
import pandas as pd
import numpy as np
import mip
import heapq
import matplotlib.pyplot as plt
import os

# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================
try:
    SAVE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SAVE_DIR = os.getcwd()

plt.rcParams.update({'font.size': 11, 'font.family': 'serif'})


# =============================================================================
# SECTION 1. DATA GENERATION
# =============================================================================

def generate_graph(topology, n):
    """Generate a random connected graph with edge parameters for UQPP instances.

    Creates Erdős-Rényi (ER) or Barabási-Albert (BA) graphs (Section 4,
    Experimental Setup) and assigns per-edge parameters:
      l(e)  ~ U[10, 100]  — lead time (latency)
      c(e)  ~ U[10,  50]  — initial capacity
      c̄(e) =  c(e) * U[1.5, 3.0] — maximum upgradeable capacity
      w(e)  ~ U[ 1,  10]  — upgrade cost coefficient

    Args:
        topology: Graph model, either 'ER' (Erdős-Rényi) or 'BA' (Barabási-Albert).
        n: Number of nodes |V|.

    Returns:
        tuple: (G, s, t) where G is the NetworkX graph, s is the source node,
            and t is the sink node.
    """
    if topology == 'ER':
        p = 6.0 / n if n > 6 else 1.0  # target average degree ≈ 6
        G = nx.erdos_renyi_graph(n, p)
    else:
        G = nx.barabasi_albert_graph(n, 3)  # attachment parameter m=3

    # Ensure connectivity by bridging disconnected components
    if not nx.is_connected(G):
        components = list(nx.connected_components(G))
        for i in range(len(components) - 1):
            G.add_edge(
                random.choice(list(components[i])),
                random.choice(list(components[i + 1]))
            )

    for u, v in G.edges():
        G[u][v]['latency'] = random.uniform(10, 100)    # l(e)
        c = random.uniform(10, 50)                       # c(e): initial capacity
        c_max = c * random.uniform(1.5, 3.0)             # c̄(e): max capacity
        G[u][v]['capacity'] = c
        G[u][v]['max_capacity'] = c_max
        G[u][v]['cost'] = random.uniform(1, 10)          # w(e): upgrade cost coefficient

    s, t = random.sample(list(G.nodes()), 2)
    return G, s, t


def generate_graph_controlled_K(topology, n, target_k_size):
    """Generate a random graph with a controlled critical capacity set size |K|.

    Constructs a graph where the cardinality of the critical capacity set
    K = {c(e) | e in E} ∪ {c̄(e) | e in E} is exactly target_k_size.
    This is used in Part 1.1 experiments (Table 1) to isolate the effect
    of |K| on algorithm performance.

    Args:
        topology: Graph model, either 'ER' or 'BA'.
        n: Number of nodes |V|.
        target_k_size: Desired cardinality |K| of the critical capacity set.

    Returns:
        tuple: (G, s, t) where G is the NetworkX graph, s is the source node,
            and t is the sink node.
    """
    if topology == 'ER':
        p = 6.0 / n if n > 6 else 1.0
        G = nx.erdos_renyi_graph(n, p)
    else:
        G = nx.barabasi_albert_graph(n, 3)

    if not nx.is_connected(G):
        components = list(nx.connected_components(G))
        for i in range(len(components) - 1):
            G.add_edge(
                random.choice(list(components[i])),
                random.choice(list(components[i + 1]))
            )

    m = G.number_of_edges()

    # Build a pool of (c, c̄) pairs that collectively produce exactly
    # target_k_size distinct values in K
    vals = set()
    pair_pool = []
    while len(vals) < target_k_size:
        c = random.uniform(10, 50)
        c_max = c * random.uniform(1.5, 3.0)
        vals.add(c)
        vals.add(c_max)
        pair_pool.append((c, c_max))

    # Assign pairs to edges, repeating from the pool as necessary
    assigned_pairs = list(pair_pool)
    while len(assigned_pairs) < m:
        assigned_pairs.append(random.choice(pair_pool))

    assigned_pairs = assigned_pairs[:m]
    random.shuffle(assigned_pairs)

    for i, (u, v) in enumerate(G.edges()):
        G[u][v]['latency'] = random.uniform(10, 100)
        G[u][v]['capacity'] = assigned_pairs[i][0]       # c(e)
        G[u][v]['max_capacity'] = assigned_pairs[i][1]   # c̄(e)
        G[u][v]['cost'] = random.uniform(1, 10)           # w(e)

    s, t = random.sample(list(G.nodes()), 2)
    return G, s, t


def get_k_size(G):
    """Compute the cardinality of the critical capacity set K.

    K is defined as K = {c(e) | e in E} ∪ {c̄(e) | e in E}, i.e., the
    sorted set of all initial and maximum capacities (Section 2.2).
    The size |K| controls the number of RSPP oracle calls in Algorithms 1–3.

    Args:
        G: NetworkX graph with 'capacity' and 'max_capacity' edge attributes.

    Returns:
        int: Cardinality |K| of the critical capacity set.
    """
    K = set()
    for u, v, d in G.edges(data=True):
        K.add(d['capacity'])      # c(e)
        K.add(d['max_capacity'])  # c̄(e)
    return len(K)


# =============================================================================
# SECTION 2. LARAC HEURISTICS FOR THE RSPP ORACLE
# =============================================================================
# Both functions implement the Lagrangian Relaxation-based Aggregated Cost
# (LARAC) heuristic used as the RSPP oracle O_RSPP(G, l, omega, B).
#
# Given a fixed target bottleneck capacity C, the RSPP sub-problem is:
#   min  sum_{e in P} l(e)   subject to   sum_{e in P} Omega_C(e) <= B
#
# LARAC solves this via three phases:
#   Phase 1: Find P_lat = argmin L(P)  (min-latency path, ignoring cost)
#   Phase 2: Find P_feas = argmin W(P) (min-cost path, unconstrained latency)
#   Phase 3: Lagrangian iterations — update lambda and search Pareto-optimal paths
# =============================================================================

def fast_larac_l1(adj, s, t, budget, C, n):
    """LARAC oracle O_RSPP for the linear cost model UQPP_1 (Section 2.2).

    Implements the LARAC heuristic to approximately solve the RSPP
    sub-problem arising from the interval-based parametric search
    (Algorithm 1). The linear upgrade cost function is:

        Omega_C(e) = w(e) * max(0, C - c(e))    if C <= c̄(e), else inf

    which is the direct implementation of Eq. (2) in the paper.

    The inner Dijkstra supports three weight modes:
      w_type=0 → w = l(e)                    (Phase 1: min-latency path P_lat)
      w_type=1 → w = Omega_C(e)              (Phase 2: min-cost path P_feas)
      w_type=2 → w = l(e) + lambda*Omega_C(e) (Phase 3: Lagrangian composite)

    Args:
        adj: Adjacency list where adj[u] = [(v, l(e), c(e), c̄(e), w(e)), ...].
        s: Source node index.
        t: Sink (target) node index.
        budget: Available upgrade budget B.
        C: Fixed target bottleneck capacity for this RSPP call.
        n: Total number of nodes |V|.

    Returns:
        tuple: (path, L_P, W_P) — the heuristic path, its total latency L(P),
            and its total upgrade cost W(P, C) = sum_{e in P} Omega_C(e).
            Returns ([], inf, inf) if no feasible path exists.
    """
    def custom_dijkstra(w_type, lambda_val=0.0):
        """Run Dijkstra with composite weight for the given phase.

        Args:
            w_type: Weight mode (0=latency, 1=cost, 2=Lagrangian composite).
            lambda_val: Lagrangian multiplier lambda >= 0 (used only in Phase 3).

        Returns:
            tuple: (path, total_latency, total_upgrade_cost)
        """
        pq = [(0.0, s)]
        dist = [float('inf')] * n
        dist[s] = 0.0
        parent = [None] * n

        while pq:
            d_curr, u = heapq.heappop(pq)
            if d_curr > dist[u]:
                continue
            if u == t:
                break
            for v, lat, cap, max_cap, ecost in adj[u]:
                # Skip edges that cannot support capacity C even after full upgrade
                if max_cap < C:
                    continue

                # Omega_C(e): linear upgrade cost (Eq. 2, Section 2.2)
                # cst = w(e) * max(0, C - c(e))
                cst = (C - cap) * ecost if C > cap else 0.0

                if w_type == 0:
                    w = lat                         # Phase 1: min L(P)
                elif w_type == 1:
                    w = cst                         # Phase 2: min W(P,C)
                else:
                    w = lat + lambda_val * cst      # Phase 3: Lagrangian

                if d_curr + w < dist[v]:
                    dist[v] = d_curr + w
                    parent[v] = (u, lat, cst)
                    heapq.heappush(pq, (dist[v], v))

        if parent[t] is None:
            return [], float('inf'), float('inf')

        # Reconstruct path and accumulate L(P), W(P,C)
        path = []
        total_lat, total_cost = 0.0, 0.0
        curr = t
        while curr != s:
            path.append(curr)
            p_node, l, c = parent[curr]
            total_lat += l
            total_cost += c
            curr = p_node
        path.append(s)
        path.reverse()
        return path, total_lat, total_cost

    # --- Phase 1: min-latency path P_lat ---
    # pc = P_lat: minimum latency path (cost may violate budget B)
    # lc = L(P_lat), cc = W(P_lat, C)
    pc, lc, cc = custom_dijkstra(0)
    if cc <= budget:
        return pc, lc, cc       # P_lat is already budget-feasible
    if not pc:
        return [], float('inf'), float('inf')

    # --- Phase 2: min-cost path P_feas ---
    # pw = P_feas: minimum upgrade-cost path (guaranteed W(P) <= B if feasible)
    # lw = L(P_feas), cw = W(P_feas, C)
    pw, lw, cw = custom_dijkstra(1)
    if cw > budget or not pw:
        return [], float('inf'), float('inf')

    # --- Phase 3: Lagrangian iterations ---
    # Update lambda = (L(P_lat) - L(P_feas)) / (W(P_feas) - W(P_lat))
    # and search for a Pareto-improving path on the latency-cost convex hull
    while True:
        if cw == cc:
            break
        lambda_val = (lc - lw) / (cw - cc)     # Lagrangian multiplier update
        p, l, c = custom_dijkstra(2, lambda_val)
        if not p:
            break
        # Convergence check: no improvement on the Lagrangian objective
        if l + lambda_val * c >= lc + lambda_val * cc - 1e-6:
            break
        if c <= budget:
            pw, lw, cw = p, l, c    # new path is budget-feasible → update P_feas
        else:
            pc, lc, cc = p, l, c    # new path exceeds budget → update P_lat

    return pw, lw, cw


def fast_larac_hw(adj, s, t, budget, C, n):
    """LARAC oracle O_RSPP for the Hamming cost model UQPP_H (Section 2.3).

    Implements the LARAC heuristic to approximately solve the RSPP
    sub-problem arising from Algorithm 2 (UQPP_H decomposition). The
    discrete Hamming upgrade cost function is:

        Omega_C(e) = 0     if c(e) >= C       (no upgrade needed)
                   = w(e)  if c(e) < C <= c̄(e) (binary upgrade decision)
                   = inf   if C > c̄(e)         (edge cannot support C)

    which is the direct implementation of Eq. (4) in the paper.

    The inner Dijkstra supports the same three weight modes as fast_larac_l1.

    Args:
        adj: Adjacency list where adj[u] = [(v, l(e), c(e), c̄(e), w(e)), ...].
        s: Source node index.
        t: Sink (target) node index.
        budget: Available upgrade budget B.
        C: Fixed target bottleneck capacity for this RSPP call.
        n: Total number of nodes |V|.

    Returns:
        tuple: (path, L_P, W_P) — the heuristic path, its total latency L(P),
            and its total Hamming upgrade cost W(P, C) = sum_{e in P} Omega_C(e).
            Returns ([], inf, inf) if no feasible path exists.
    """
    def custom_dijkstra(w_type, lambda_val=0.0):
        """Run Dijkstra with composite weight for the given phase.

        Args:
            w_type: Weight mode (0=latency, 1=cost, 2=Lagrangian composite).
            lambda_val: Lagrangian multiplier lambda >= 0 (used only in Phase 3).

        Returns:
            tuple: (path, total_latency, total_hamming_cost)
        """
        pq = [(0.0, s)]
        dist = [float('inf')] * n
        dist[s] = 0.0
        parent = [None] * n

        while pq:
            d_curr, u = heapq.heappop(pq)
            if d_curr > dist[u]:
                continue
            if u == t:
                break
            for v, lat, cap, max_cap, ecost in adj[u]:
                # Skip edges that cannot support capacity C even after full upgrade
                if max_cap < C:
                    continue

                # Omega_C(e): Hamming upgrade cost (Eq. 4, Section 2.3)
                # cst = 0 if c(e) >= C (no upgrade), else w(e) (binary upgrade)
                cst = 0.0 if cap >= C else ecost

                if w_type == 0:
                    w = lat                         # Phase 1: min L(P)
                elif w_type == 1:
                    w = cst                         # Phase 2: min W(P,C)
                else:
                    w = lat + lambda_val * cst      # Phase 3: Lagrangian

                if d_curr + w < dist[v]:
                    dist[v] = d_curr + w
                    parent[v] = (u, lat, cst)
                    heapq.heappush(pq, (dist[v], v))

        if parent[t] is None:
            return [], float('inf'), float('inf')

        # Reconstruct path and accumulate L(P), W(P,C)
        path = []
        total_lat, total_cost = 0.0, 0.0
        curr = t
        while curr != s:
            path.append(curr)
            p_node, l, c = parent[curr]
            total_lat += l
            total_cost += c
            curr = p_node
        path.append(s)
        path.reverse()
        return path, total_lat, total_cost

    # --- Phase 1: min-latency path P_lat ---
    # pc = P_lat, lc = L(P_lat), cc = W(P_lat, C)
    pc, lc, cc = custom_dijkstra(0)
    if cc <= budget:
        return pc, lc, cc
    if not pc:
        return [], float('inf'), float('inf')

    # --- Phase 2: min-cost path P_feas ---
    # pw = P_feas, lw = L(P_feas), cw = W(P_feas, C)
    pw, lw, cw = custom_dijkstra(1)
    if cw > budget or not pw:
        return [], float('inf'), float('inf')

    # --- Phase 3: Lagrangian iterations ---
    while True:
        if cw == cc:
            break
        lambda_val = (lc - lw) / (cw - cc)     # Lagrangian multiplier update
        p, l, c = custom_dijkstra(2, lambda_val)
        if not p:
            break
        if l + lambda_val * c >= lc + lambda_val * cc - 1e-6:
            break
        if c <= budget:
            pw, lw, cw = p, l, c    # update P_feas
        else:
            pc, lc, cc = p, l, c    # update P_lat

    return pw, lw, cw


# =============================================================================
# SECTION 3. DECOMPOSITION ALGORITHMS (DA-L1, DA-HW, DA-HC)
# =============================================================================

def solve_linear_decomposition(G, s, t, sigma, budget):
    """DA-L1: decomposition algorithm for UQPP_1 (Algorithm 1, Section 2.2).

    Implements the interval-based parametric search that reduces UQPP_1 to
    repeated calls to the LARAC oracle O_RSPP. The algorithm iterates over
    the sorted critical capacity set K = {c(e)} ∪ {c̄(e)} and, within each
    interval [K_i, K_{i+1}], advances a search pointer C_curr to the binding
    capacity C_bind = C_curr + (B - W_P(C_curr)) / beta where
    beta = sum_{e in P: c(e) < K_{i+1}} w(e) is the cost slope.

    The quickest distance Q(C) = Z(C) + sigma/C is evaluated at both C_curr
    and C_bind for each discovered path P, as proven in Theorem 1.

    Branch-and-bound pruning: if L_min + sigma/C_end >= global_min, the
    remaining intervals cannot improve the objective and the search terminates.

    Args:
        G: NetworkX graph with edge attributes 'latency', 'capacity',
            'max_capacity', 'cost' corresponding to l(e), c(e), c̄(e), w(e).
        s: Source node.
        t: Sink node.
        sigma: Data transmission size sigma > 0.
        budget: Upgrade budget B.

    Returns:
        float: Optimal (approximate) quickest distance Q* = min_C Q(C).
            Returns inf if no feasible upgrade path exists.
    """
    nodes = list(G.nodes())
    n = len(nodes)
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    s_idx, t_idx = node_to_idx[s], node_to_idx[t]

    # Build compact adjacency list and collect critical capacity set K
    adj = [[] for _ in range(n)]
    edge_info = {}
    K_set = set()

    for u, v, d in G.edges(data=True):
        ui, vi = node_to_idx[u], node_to_idx[v]
        lat, cap, max_cap, ecost = (
            d['latency'], d['capacity'], d['max_capacity'], d['cost']
        )
        adj[ui].append((vi, lat, cap, max_cap, ecost))
        adj[vi].append((ui, lat, cap, max_cap, ecost))
        edge_info[(ui, vi)] = (cap, ecost)  # (c(e), w(e))
        edge_info[(vi, ui)] = (cap, ecost)
        K_set.add(cap)      # c(e) ∈ K
        K_set.add(max_cap)  # c̄(e) ∈ K

    # K_desc: K sorted in descending order for outer loop
    K_desc = sorted(list(K_set), reverse=True)

    # --- Step 1: Compute L_min = min_{P} L(P) for branch-and-bound pruning ---
    # L_min is a lower bound on Z(C) for all C, enabling early termination
    pq = [(0.0, s_idx)]
    dist = [float('inf')] * n
    dist[s_idx] = 0.0
    while pq:
        d_curr, u = heapq.heappop(pq)
        if d_curr > dist[u]:
            continue
        if u == t_idx:
            break
        for v, lat, _, _, _ in adj[u]:
            if d_curr + lat < dist[v]:
                dist[v] = d_curr + lat
                heapq.heappush(pq, (dist[v], v))

    L_min = dist[t_idx]
    if L_min == float('inf'):
        return float('inf')

    # --- Step 2: Warm-start with the smallest critical capacity K_min ---
    global_min = float('inf')
    path, lp, wp = fast_larac_l1(adj, s_idx, t_idx, budget, K_desc[-1], n)
    if path:
        global_min = lp + sigma / K_desc[-1]

    # --- Step 3: Interval-based parametric search (Algorithm 1) ---
    # Outer loop over consecutive pairs (K_{i+1}, K_i) in descending order.
    # Within each interval [C_curr, C_end = K_i], advance C_curr to C_bind.
    for i in range(len(K_desc) - 1):
        C_end = K_desc[i]       # upper boundary of current interval
        C_curr = K_desc[i + 1]  # lower boundary = next critical capacity

        # Branch-and-bound pruning (Theorem 1):
        # if the best possible Q at C_end cannot beat global_min, stop
        if L_min + sigma / C_end >= global_min:
            break

        c_search = C_curr
        while c_search < C_end:
            if L_min + sigma / C_end >= global_min:
                break

            # Call LARAC oracle O_RSPP at current search point C_curr
            path, L_P, W_P = fast_larac_l1(adj, s_idx, t_idx, budget, c_search, n)
            if not path or W_P > budget:
                break

            # Evaluate Q(C_curr) = L(P) + sigma / C_curr
            global_min = min(global_min, L_P + sigma / c_search)

            # Compute cost slope beta on [c_search, C_end]:
            # beta = sum_{e in P: c(e) < C_end} w(e)
            beta = sum(
                edge_info[(path[j], path[j + 1])][1]
                for j in range(len(path) - 1)
                if edge_info[(path[j], path[j + 1])][0] < C_end
            )

            # Compute C_bind = C_curr + (B - W_P(C_curr)) / beta
            if beta <= 1e-6:
                c_bind = C_end  # cost is constant; path valid until interval end
            else:
                c_bind = min(C_end, c_search + (budget - W_P) / beta)

            # Evaluate Q(C_bind) = L(P) + sigma / C_bind (local minimum for P)
            global_min = min(global_min, L_P + sigma / c_bind)

            if c_bind >= C_end:
                break  # path P remains feasible for the whole interval

            # Advance past C_bind (epsilon shift to force new path discovery)
            c_search = c_bind + 1e-5

    return global_min


def solve_weighted_hamming_decomposition(G, s, t, sigma, budget):
    """DA-HW: decomposition algorithm for UQPP_H (Algorithm 2, Section 2.3).

    Implements the UQPP_H decomposition for the general weighted Hamming
    cost model (Case 1, Section 2.3). By Proposition 2, the optimal
    bottleneck capacity C* lies in K, so the algorithm iterates C over
    K = {c(e)} ∪ {c̄(e)} and calls the LARAC oracle at each value.

    The total complexity is O(|K| * T_RSPP) = O(|E| * T_RSPP) (Theorem 2).

    Branch-and-bound pruning: as in DA-L1, the search terminates early when
    L_min + sigma/C >= global_min, since Q(C) cannot improve further.

    Args:
        G: NetworkX graph with edge attributes 'latency', 'capacity',
            'max_capacity', 'cost' corresponding to l(e), c(e), c̄(e), w(e).
        s: Source node.
        t: Sink node.
        sigma: Data transmission size sigma > 0.
        budget: Upgrade budget B (weighted Hamming budget).

    Returns:
        float: Approximate optimal quickest distance Q* = min_{C in K} Q(C).
            Returns inf if no feasible upgrade path exists.
    """
    nodes = list(G.nodes())
    n = len(nodes)
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    s_idx, t_idx = node_to_idx[s], node_to_idx[t]

    adj = [[] for _ in range(n)]
    K_set = set()

    for u, v, d in G.edges(data=True):
        ui, vi = node_to_idx[u], node_to_idx[v]
        lat, cap, max_cap, ecost = (
            d['latency'], d['capacity'], d['max_capacity'], d['cost']
        )
        adj[ui].append((vi, lat, cap, max_cap, ecost))
        adj[vi].append((ui, lat, cap, max_cap, ecost))
        K_set.add(cap)      # c(e) ∈ K
        K_set.add(max_cap)  # c̄(e) ∈ K

    K_desc = sorted(list(K_set), reverse=True)

    # Compute L_min for branch-and-bound pruning
    pq = [(0.0, s_idx)]
    dist = [float('inf')] * n
    dist[s_idx] = 0.0
    while pq:
        d_curr, u = heapq.heappop(pq)
        if d_curr > dist[u]:
            continue
        if u == t_idx:
            break
        for v, lat, _, _, _ in adj[u]:
            if d_curr + lat < dist[v]:
                dist[v] = d_curr + lat
                heapq.heappush(pq, (dist[v], v))

    L_min = dist[t_idx]
    if L_min == float('inf'):
        return float('inf')

    # Warm-start with the smallest critical capacity
    global_min = float('inf')
    path, lp, wp = fast_larac_hw(adj, s_idx, t_idx, budget, K_desc[-1], n)
    if path:
        global_min = lp + sigma / K_desc[-1]

    # Main loop: iterate C over K in descending order (Proposition 2 / Alg. 2)
    for C in K_desc:
        # Branch-and-bound pruning: Q(C) >= L_min + sigma/C >= global_min
        if L_min + sigma / C >= global_min:
            break

        path, lat, cost = fast_larac_hw(adj, s_idx, t_idx, budget, C, n)
        if path and cost <= budget:
            global_min = min(global_min, lat + sigma / C)

    return global_min


def solve_cardinality_layered(G, s, t, sigma, budget):
    """DA-HC: exact layered-graph algorithm for cardinality-constrained UQPP.

    Implements Algorithm 3 (Section 2.3, Case 2) for the unit Hamming model
    where w(e) = 1 for all e and the budget B = k is a non-negative integer
    bounding the number of edge upgrades. This variant is solved exactly in
    polynomial time O(m * k * (m + n log n)) (Theorem 3).

    For each C ∈ K, the algorithm constructs the layered graph G_L = (V_L, E_L)
    with V_L = {(v, i) | v ∈ V, 0 <= i <= k} where a node (v, i) represents
    arriving at v having consumed exactly i upgrades. Edges are classified as:
      - Level edges  ((u,i),(v,i)): c(e) >= C, no upgrade consumed
      - Upgrade edges ((u,i),(v,i+1)): c(e) < C <= c̄(e), 1 upgrade consumed
    (Proposition 3 and Example 1 in Section 2.3.)

    An A* search with an admissible heuristic h(v) = shortest latency from v
    to t (reverse Dijkstra) is used instead of plain Dijkstra for efficiency,
    with Pareto domination pruning on the (latency, upgrades) state space.

    Args:
        G: NetworkX graph with edge attributes 'latency', 'capacity',
            'max_capacity' corresponding to l(e), c(e), c̄(e). Edge costs
            w(e) are unused here (unit Hamming model).
        s: Source node.
        t: Sink node.
        sigma: Data transmission size sigma > 0.
        budget: Integer cardinality budget k (max number of upgrades).

    Returns:
        float: Exact optimal quickest distance Q* = min_{C in K} Q(C).
            Returns inf if no feasible upgrade path exists.
    """
    nodes = list(G.nodes())
    n = len(nodes)
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    s_idx, t_idx = node_to_idx[s], node_to_idx[t]
    k = min(budget, n - 1)  # cardinality budget k, capped at n-1

    adj = [[] for _ in range(n)]
    K_set = set()
    for u, v, d in G.edges(data=True):
        ui, vi = node_to_idx[u], node_to_idx[v]
        lat, cap, max_cap = d['latency'], d['capacity'], d['max_capacity']
        adj[ui].append((vi, lat, cap, max_cap))
        adj[vi].append((ui, lat, cap, max_cap))
        K_set.add(cap)      # c(e) ∈ K
        K_set.add(max_cap)  # c̄(e) ∈ K

    K_desc = sorted(list(K_set), reverse=True)

    # Admissible heuristic h(v) = min latency from v to t (reverse Dijkstra).
    # h(v) <= true shortest latency v→t, so A* remains admissible.
    h = [float('inf')] * n
    h[t_idx] = 0
    pq_h = [(0, t_idx)]
    while pq_h:
        d_curr, u = heapq.heappop(pq_h)
        if d_curr > h[u]:
            continue
        for v, lat, _, _ in adj[u]:
            if d_curr + lat < h[v]:
                h[v] = d_curr + lat
                heapq.heappush(pq_h, (h[v], v))

    L_min = h[s_idx]  # unconstrained min latency, used for B&B pruning
    if L_min == float('inf'):
        return float('inf')

    global_min = float('inf')

    # Main loop: iterate C over K in descending order (Algorithm 3)
    for C in K_desc:
        # B&B pruning: Q(C) >= L_min + sigma/C
        if L_min + sigma / C >= global_min:
            break

        # A* on implicit layered graph G_L for fixed C.
        # State: (f = g + h(v), g = path latency, v = node, upgrades used)
        # dist[v][i] = best known latency to reach v having used i upgrades
        dist = [[float('inf')] * (k + 1) for _ in range(n)]
        dist[s_idx][0] = 0
        min_upgrades_extracted = [k + 1] * n  # Pareto domination tracker
        pq = [(h[s_idx], 0, s_idx, 0)]        # (f, g, node, upgrades)

        while pq:
            f, g, u, upgrades = heapq.heappop(pq)

            # Pareto domination: skip if a state with fewer upgrades was already expanded
            if upgrades >= min_upgrades_extracted[u]:
                continue
            min_upgrades_extracted[u] = upgrades

            # Pruning: this state cannot beat the current best solution
            if f + sigma / C >= global_min:
                continue

            if u == t_idx:
                global_min = min(global_min, g + sigma / C)
                break

            for v, edge_lat, cap, max_cap in adj[u]:
                # Determine upgrade cost for this edge at target capacity C
                if cap >= C:
                    nxt_up = upgrades          # level edge: no upgrade needed
                elif max_cap >= C and upgrades < k:
                    nxt_up = upgrades + 1      # upgrade edge: consume 1 budget unit
                else:
                    continue                   # infeasible edge (c̄(e) < C or k exhausted)

                nxt_g = g + edge_lat
                if nxt_g < dist[v][nxt_up]:
                    dist[v][nxt_up] = nxt_g
                    nxt_f = nxt_g + h[v]       # A* f-value: g + h(v)
                    if nxt_f + sigma / C < global_min:
                        heapq.heappush(pq, (nxt_f, nxt_g, v, nxt_up))

    return global_min


# =============================================================================
# SECTION 4. BENCHMARK MIP SOLVERS (GUROBI REFERENCE IMPLEMENTATIONS)
# =============================================================================

def solve_mip_miqcp_l1(G, s, t, sigma, budget, time_limit=60):
    """Gurobi MIQCP exact solver for UQPP_1 (Appendix, linear cost model).

    Formulates UQPP_1 as a Mixed-Integer Quadratically Constrained Program
    (MIQCP) using a piecewise-linear outer approximation of the term sigma/C
    evaluated at 50 linearisation points. Serves as the exact reference
    benchmark for Table 1 (Section 4).

    Decision variables:
      z[u,v] ∈ {0,1}: directed flow on edge (u,v) (path selection)
      x[u,v] ∈ [0, c̄(e)-c(e)]: continuous capacity upgrade x_e
      C ∈ R+: bottleneck capacity variable
      T_trans ∈ R+: piecewise-linear approximation of sigma/C

    Args:
        G: NetworkX graph with edge attributes 'latency', 'capacity',
            'max_capacity', 'cost'.
        s: Source node.
        t: Sink node.
        sigma: Data transmission size sigma > 0.
        budget: Upgrade budget B for linear cost constraint sum_e w(e)*x_e <= B.
        time_limit: Maximum solver time in seconds (default 60).

    Returns:
        tuple: (objective_value, solve_time). Returns (inf, time_limit) on
            timeout or infeasibility.
    """
    m = mip.Model("UQPP_L1", sense=mip.MINIMIZE, solver_name=mip.GUROBI)
    m.verbose = 0
    m.max_seconds = time_limit
    m.threads = 1
    edges = list(G.edges(data=True))
    M_val = max(d['max_capacity'] for u, v, d in edges) * 2  # big-M constant
    z, x = {}, {}
    for u, v, d in edges:
        z[u, v] = m.add_var(var_type=mip.BINARY)
        z[v, u] = m.add_var(var_type=mip.BINARY)
        x[u, v] = m.add_var(lb=0, ub=d['max_capacity'] - d['capacity'])  # x_e bounds

    C = m.add_var(lb=0.001)     # bottleneck capacity C
    T_trans = m.add_var(lb=0)   # piecewise-linear approx. of sigma/C

    # Flow conservation constraints for s-t path
    for node in G.nodes():
        out_f = mip.xsum(z[node, j] for j in G.neighbors(node))
        in_f = mip.xsum(z[j, node] for j in G.neighbors(node))
        if node == s:
            m += out_f - in_f == 1
        elif node == t:
            m += out_f - in_f == -1
        else:
            m += out_f - in_f == 0

    # Budget constraint: sum_e w(e) * x_e <= B
    m += mip.xsum(d['cost'] * x[u, v] for u, v, d in edges) <= budget

    # Bottleneck capacity constraints: C <= c(e) + x_e for all e on path
    for u, v, d in edges:
        m += C <= d['capacity'] + x[u, v] + M_val * (1 - z[u, v] - z[v, u])

    # Piecewise-linear outer approximation of sigma/C (50 tangent cuts)
    c_min = min(d['capacity'] for u, v, d in edges)
    c_max = max(d['max_capacity'] for u, v, d in edges)
    if c_max > c_min:
        for P_i in np.linspace(max(0.001, c_min), c_max, 50):
            m += T_trans >= round(2 * sigma / P_i, 6) - round(sigma / (P_i ** 2), 6) * C

    m.objective = (
        mip.xsum(d['latency'] * (z[u, v] + z[v, u]) for u, v, d in edges) + T_trans
    )
    start_time = time.time()
    status = m.optimize()
    if status in [mip.OptimizationStatus.OPTIMAL, mip.OptimizationStatus.FEASIBLE]:
        return m.objective_value, time.time() - start_time
    return float('inf'), time_limit


def solve_mip_milp_hamming(G, s, t, sigma, budget, unit_cost=False, time_limit=60):
    """Gurobi MILP exact solver for UQPP_H and cardinality UQPP (Appendix).

    Formulates UQPP_H as a Mixed-Integer Linear Program. The bottleneck
    capacity is linearised by introducing a binary selector mu[idx] over
    the sorted critical capacity set K. Serves as the exact reference
    benchmark for Table 2 (UQPP_H) and Table 3 (cardinality UQPP).

    Decision variables:
      z[u,v] ∈ {0,1}: directed flow on edge (u,v) (path selection)
      y[u,v] ∈ {0,1}: upgrade decision H(x_e) for edge (u,v)
      mu[idx] ∈ {0,1}: selector for C = K[idx] (exactly one is 1)

    Args:
        G: NetworkX graph with edge attributes 'latency', 'capacity',
            'max_capacity', 'cost'.
        s: Source node.
        t: Sink node.
        sigma: Data transmission size sigma > 0.
        budget: Budget B. Interpreted as cardinality k if unit_cost=True,
            or as weighted Hamming budget sum_e w(e)*y_e <= B otherwise.
        unit_cost: If True, use unit Hamming cost (cardinality constraint,
            UQPP_HC). If False, use weighted Hamming cost (UQPP_H).
        time_limit: Maximum solver time in seconds (default 60).

    Returns:
        tuple: (objective_value, solve_time). Returns (inf, time_limit) on
            timeout or infeasibility.
    """
    m = mip.Model("UQPP_H", sense=mip.MINIMIZE, solver_name=mip.GUROBI)
    m.verbose = 0
    m.max_seconds = time_limit
    m.threads = 1
    edges = list(G.edges(data=True))
    M_val = max(d['max_capacity'] for u, v, d in edges) * 2

    # Critical capacity set K (sorted)
    K = sorted(
        list(
            set(d['capacity'] for u, v, d in edges) |
            set(d['max_capacity'] for u, v, d in edges)
        )
    )
    z, y = {}, {}
    for u, v, d in edges:
        z[u, v] = m.add_var(var_type=mip.BINARY)
        z[v, u] = m.add_var(var_type=mip.BINARY)
        y[u, v] = m.add_var(var_type=mip.BINARY)  # y_e: binary upgrade decision

    # mu[idx]: selects C = K[idx] as the bottleneck capacity
    mu = [m.add_var(var_type=mip.BINARY) for _ in range(len(K))]
    m += mip.xsum(mu) == 1  # exactly one critical capacity is selected

    # Flow conservation
    for node in G.nodes():
        out_f = mip.xsum(z[node, j] for j in G.neighbors(node))
        in_f = mip.xsum(z[j, node] for j in G.neighbors(node))
        if node == s:
            m += out_f - in_f == 1
        elif node == t:
            m += out_f - in_f == -1
        else:
            m += out_f - in_f == 0

    # Budget constraint: sum_e Omega(e)*y_e <= B
    # unit_cost=True → sum_e y_e <= k (cardinality); False → sum_e w(e)*y_e <= B
    m += mip.xsum(
        y[u, v] if unit_cost else d['cost'] * y[u, v]
        for u, v, d in edges
    ) <= budget

    # Bottleneck capacity constraints using mu selector
    for u, v, d in edges:
        m += (
            mip.xsum(K[idx] * mu[idx] for idx in range(len(K)))
            <= d['capacity'] + (d['max_capacity'] - d['capacity']) * y[u, v]
            + M_val * (1 - z[u, v] - z[v, u])
        )

    # Objective: L(P) + sigma/C, with sigma/C linearised via mu selector
    m.objective = (
        mip.xsum(d['latency'] * (z[u, v] + z[v, u]) for u, v, d in edges) +
        mip.xsum(round(sigma / K[idx], 6) * mu[idx] for idx in range(len(K)))
    )
    start = time.time()
    status = m.optimize()
    if status in [mip.OptimizationStatus.OPTIMAL, mip.OptimizationStatus.FEASIBLE]:
        return m.objective_value, time.time() - start
    return float('inf'), time_limit


# =============================================================================
# SECTION 5. HELPER UTILITIES
# =============================================================================

def compute_metrics(sum_da, sum_mip, sum_gap, timeout_cnt, trials, tl):
    """Format benchmark metrics for a single configuration into display strings.

    Computes average solve times and optimality gap for the decomposition
    algorithm (DA) vs. Gurobi (MIP) across N_TRIALS instances.

    The gap is computed only over non-timeout trials as:
        gap(%) = |Q_DA - Q_Gurobi| / Q_Gurobi * 100

    Args:
        sum_da: Cumulative DA solve time over all trials (seconds).
        sum_mip: Cumulative Gurobi solve time over all trials (seconds).
        sum_gap: Cumulative optimality gap (%) over non-timeout trials.
        timeout_cnt: Number of trials where Gurobi hit the time limit.
        trials: Total number of trials N_TRIALS.
        tl: Gurobi time limit in seconds (used for speedup denominator).

    Returns:
        tuple: (mip_str, da_str, speedup_str, gap_str) — formatted strings
            ready for table output.
    """
    avg_da = sum_da / trials
    avg_mip = sum_mip / trials
    valid_trials = trials - timeout_cnt
    avg_gap = (sum_gap / valid_trials) if valid_trials > 0 else 0.0

    if timeout_cnt == trials:
        return "TL", f"{avg_da:.4f}", f">{tl / max(avg_da, 1e-5):.0f}x", "-"
    elif timeout_cnt > 0:
        return (f"{avg_mip:.2f}*", f"{avg_da:.4f}",
                f"{avg_mip / max(avg_da, 1e-5):.0f}x", f"{avg_gap:.2f}")
    else:
        return (f"{avg_mip:.2f}", f"{avg_da:.4f}",
                f"{avg_mip / max(avg_da, 1e-5):.0f}x", f"{avg_gap:.2f}")


def warmup_benchmark(warmup_trials=3, warmup_nodes=50, sigma=1000):
    """Warm-up phase to avoid cold-start bias in runtime measurements.

    Runs all three decomposition algorithms (DA-L1, DA-HW, DA-HC) on small
    graphs before the actual benchmark to allow JIT compilation, OS caching,
    and NumPy/NetworkX internal initialisation to complete. This ensures
    that the first benchmark trial is not disproportionately slow.

    Args:
        warmup_trials: Number of warm-up iterations (default: 3).
        warmup_nodes: Number of nodes in warm-up graphs (default: 50,
            much smaller than benchmark sizes of 100–5000).
        sigma: Data size sigma used in warm-up runs (default: 1000).
    """
    print("\n" + "-" * 85)
    print("WARM-UP PHASE: Running 3 small trials to initialize runtime...")
    print("-" * 85)

    try:
        for i in range(warmup_trials):
            G_er, s_er, t_er = generate_graph('ER', warmup_nodes)
            G_ba, s_ba, t_ba = generate_graph('BA', warmup_nodes)

            budget_er = sum(d['cost'] for u, v, d in G_er.edges(data=True)) * 0.20
            budget_ba = sum(d['cost'] for u, v, d in G_ba.edges(data=True)) * 0.20

            try:
                _ = solve_linear_decomposition(G_er, s_er, t_er, sigma, budget_er)
                _ = solve_weighted_hamming_decomposition(G_er, s_er, t_er, sigma, budget_er)
                _ = solve_cardinality_layered(
                    G_er, s_er, t_er, sigma,
                    int(G_er.number_of_edges() * 0.20)
                )
            except Exception:
                pass  # Errors during warm-up are non-critical

            print(f"  Warm-up iteration {i + 1}/{warmup_trials} completed")

    except Exception as e:
        print(f"  Warning: Warm-up encountered error (non-critical): {e}")

    print("✓ Warm-up phase complete. Starting actual benchmark...\n")


def safe_solve(solve_func, *args, **kwargs):
    """Call a solver function with exception handling.

    Wraps any decomposition solver (DA-L1, DA-HW, DA-HC) in a try-except
    block so that a single failing instance does not abort the full benchmark.

    Args:
        solve_func: Callable solver (e.g., solve_linear_decomposition).
        *args: Positional arguments forwarded to solve_func.
        **kwargs: Keyword arguments forwarded to solve_func.

    Returns:
        float: Solver result, or inf if an exception is raised.
    """
    try:
        return solve_func(*args, **kwargs)
    except Exception as e:
        print(f"    Warning: Solver failed: {e}")
        return float('inf')


def graph_generator(topology, n_nodes, count):
    """Yield random graph instances on demand (memory-efficient generator).

    Args:
        topology: Graph model, either 'ER' or 'BA'.
        n_nodes: Number of nodes |V| per instance.
        count: Number of instances to generate.

    Yields:
        tuple: (G, s, t) for each generated instance.
    """
    for _ in range(count):
        yield generate_graph(topology, n_nodes)


def _run_single_algorithm(G, s, t, sig, solver_gurobi, solver_da, budget,
                          solver_kwargs, time_limit):
    """Run one (Gurobi, DA) pair and return timing and gap metrics.

    Executes the Gurobi reference solver and the corresponding decomposition
    algorithm on a single graph instance, then computes the optimality gap:
        gap(%) = |Q_DA - Q_Gurobi| / Q_Gurobi * 100

    Args:
        G: NetworkX graph instance.
        s: Source node.
        t: Sink node.
        sig: Data size sigma.
        solver_gurobi: Callable Gurobi solver (e.g., solve_mip_miqcp_l1).
        solver_da: Callable decomposition algorithm (e.g., solve_linear_decomposition).
        budget: Budget B for this trial (cost budget for L1/HW, cardinality k for HC).
        solver_kwargs: Extra keyword arguments for solver_gurobi (e.g., unit_cost).
        time_limit: Gurobi time limit in seconds.

    Returns:
        tuple: (t_da, t_gurobi, gap_percent, is_timeout) where t_da and
            t_gurobi are wall-clock times in seconds, gap_percent is the
            relative optimality gap, and is_timeout indicates whether Gurobi
            hit the time limit.
    """
    try:
        obj_gurobi, t_mip = solver_gurobi(G, s, t, sig, budget, **solver_kwargs)
    except Exception as e:
        print(f"    Gurobi error: {e}")
        obj_gurobi, t_mip = float('inf'), time_limit

    st = time.time()
    obj_da = safe_solve(solver_da, G, s, t, sig, budget)
    t_da = time.time() - st

    is_timeout = t_mip >= time_limit or obj_gurobi == float('inf')
    gap_percent = 0.0
    if not is_timeout and obj_gurobi > 1e-6:
        gap_percent = abs(obj_da - obj_gurobi) / obj_gurobi * 100

    t_gurobi_final = time_limit if is_timeout else t_mip
    return t_da, t_gurobi_final, gap_percent, is_timeout


# =============================================================================
# SECTION 6. EXPERIMENTAL RUNNERS (TABLES 1–4, SECTION 4)
# =============================================================================

def run_part1_1(N_TRIALS, TIME_LIMIT, SIGMA):
    """Table 1 (Part A): UQPP_1 scalability with fixed |K| = 100.

    Benchmarks DA-L1 vs. Gurobi MIQCP across graph sizes n ∈ {100, …, 5000}
    with the critical capacity set size |K| controlled to 100 (Table 1,
    Section 4). Budget B = 20% of total upgrade cost for full capacity.

    Args:
        N_TRIALS: Number of random instances per configuration.
        TIME_LIMIT: Gurobi time limit in seconds.
        SIGMA: Data size sigma for quickest path evaluation.
    """
    print("\n" + "=" * 85)
    print("=== PART 1.1: UQPP_1 vs Gurobi (Fixed |K| = 100, Varying Nodes) ===")
    print("=" * 85)

    warmup_benchmark(warmup_trials=3, warmup_nodes=50, sigma=SIGMA)

    n_ranges = [100, 200, 500, 1000, 2000, 5000]
    results = []
    da_times = {'ER': [], 'BA': []}
    gurobi_times = {'ER': [], 'BA': []}

    for topo in ['ER', 'BA']:
        for n_val in n_ranges:
            s_da, s_mip, s_gap, to_cnt = 0, 0, 0, 0
            actual_k_mean = 0

            for _ in range(N_TRIALS):
                G, s, t = generate_graph_controlled_K(topo, n_val, 100)
                budget = sum(d['cost'] for u, v, d in G.edges(data=True)) * 0.20
                actual_k_mean += get_k_size(G)

                obj_gurobi, t_mip = solve_mip_miqcp_l1(G, s, t, SIGMA, budget, TIME_LIMIT)
                st = time.time()
                obj_da = solve_linear_decomposition(G, s, t, SIGMA, budget)
                t_da = time.time() - st

                s_da += t_da
                if t_mip >= TIME_LIMIT or obj_gurobi == float('inf'):
                    s_mip += TIME_LIMIT
                    to_cnt += 1
                else:
                    s_mip += t_mip
                    if obj_gurobi > 1e-6:
                        s_gap += abs(obj_da - obj_gurobi) / obj_gurobi * 100

            avg_k = int(actual_k_mean / N_TRIALS)
            da_times[topo].append(s_da / N_TRIALS)
            gurobi_times[topo].append(s_mip / N_TRIALS)

            m_str, d_str, sp, gap_str = compute_metrics(
                s_da, s_mip, s_gap, to_cnt, N_TRIALS, TIME_LIMIT
            )
            results.append({
                "Topology": topo, "Nodes": n_val, "|K|": avg_k,
                "Gurobi (s)": m_str, "DA-L1 (s)": d_str,
                "Speedup": sp, "Gap (%)": gap_str
            })
            print(
                f"[{topo}] Nodes: {n_val:4d} | |K|: {avg_k:3d} | "
                f"DA: {s_da / N_TRIALS:.3f}s | Gurobi: {m_str}s | Gap: {gap_str}%"
            )

    df = pd.DataFrame(results)
    print("\nTable 1.2: UQPP_1 Algorithm vs. Gurobi (Varying Nodes)")
    print(df.to_string(index=False))

    csv_file = os.path.join(SAVE_DIR, 'Table_1_1_UQPP1_Fixed_K.csv')
    df.to_csv(csv_file, index=False)
    print(f"✓ Table exported to {csv_file}")

    plt.figure(figsize=(10, 6))
    plt.plot(n_ranges, da_times['ER'], 'bo-', label='DA-L1 (ER)')
    plt.plot(n_ranges, gurobi_times['ER'], 'r^--', label='Gurobi (ER)')
    plt.plot(n_ranges, da_times['BA'], 'gs-', label='DA-L1 (BA)')
    plt.plot(n_ranges, gurobi_times['BA'], 'mD--', label='Gurobi (BA)')
    plt.title('Computation Time vs. Number of Nodes for UQPP_1 (|K| = 100)')
    plt.xlabel('Number of Nodes')
    plt.xscale('log')
    plt.ylabel('Computation Time (s)')
    plt.yscale('log')
    plt.xticks(n_ranges)
    plt.grid(True, which="both", linestyle=':')
    plt.legend()
    plt.savefig(
        os.path.join(SAVE_DIR, 'Fig_1_1_UQPP1_vs_Nodes.png'),
        dpi=300, bbox_inches='tight'
    )
    plt.close()


def run_part1_2(N_TRIALS, TIME_LIMIT, SIGMA):
    """Table 1 (Part B): UQPP_1 scalability with unconstrained |K|.

    Benchmarks DA-L1 vs. Gurobi MIQCP across graph sizes n ∈ {100, …, 5000}
    without controlling |K| (standard random generation). This measures
    algorithm performance as both n and |K| grow naturally (Table 1,
    Section 4). Budget B = 20% of total upgrade cost.

    Args:
        N_TRIALS: Number of random instances per configuration.
        TIME_LIMIT: Gurobi time limit in seconds.
        SIGMA: Data size sigma for quickest path evaluation.
    """
    print("\n" + "=" * 85)
    print("=== PART 1.2: UQPP_1 vs Gurobi (Unconstrained |K|, Varying Nodes) ===")
    print("=" * 85)

    warmup_benchmark(warmup_trials=3, warmup_nodes=50, sigma=SIGMA)

    n_ranges = [100, 200, 500, 1000, 2000, 5000]
    results = []
    da_times = {'ER': [], 'BA': []}
    gurobi_times = {'ER': [], 'BA': []}

    for topo in ['ER', 'BA']:
        for n_val in n_ranges:
            s_da, s_mip, s_gap, to_cnt = 0, 0, 0, 0
            actual_k_mean = 0

            for _ in range(N_TRIALS):
                G, s, t = generate_graph(topo, n_val)  # uncontrolled |K|
                budget = sum(d['cost'] for u, v, d in G.edges(data=True)) * 0.20
                actual_k_mean += get_k_size(G)

                obj_gurobi, t_mip = solve_mip_miqcp_l1(G, s, t, SIGMA, budget, TIME_LIMIT)
                st = time.time()
                obj_da = solve_linear_decomposition(G, s, t, SIGMA, budget)
                t_da = time.time() - st

                s_da += t_da
                if t_mip >= TIME_LIMIT or obj_gurobi == float('inf'):
                    s_mip += TIME_LIMIT
                    to_cnt += 1
                else:
                    s_mip += t_mip
                    if obj_gurobi > 1e-6:
                        s_gap += abs(obj_da - obj_gurobi) / obj_gurobi * 100

            avg_k = int(actual_k_mean / N_TRIALS)
            da_times[topo].append(s_da / N_TRIALS)
            gurobi_times[topo].append(s_mip / N_TRIALS)

            m_str, d_str, sp, gap_str = compute_metrics(
                s_da, s_mip, s_gap, to_cnt, N_TRIALS, TIME_LIMIT
            )
            results.append({
                "Topology": topo, "Nodes": n_val, "|K|": avg_k,
                "Gurobi (s)": m_str, "DA-L1 (s)": d_str,
                "Speedup": sp, "Gap (%)": gap_str
            })
            print(
                f"  [{topo}] Nodes: {n_val:4d} | |K|: {avg_k:4d} | "
                f"DA: {s_da / N_TRIALS:.3f}s | Gurobi: {m_str}s | Gap: {gap_str}%"
            )

    df = pd.DataFrame(results)
    print("\nTable 1.3: UQPP_1 Algorithm vs. Gurobi (Unconstrained |K|, Varying Nodes)")
    print(df.to_string(index=False))

    csv_file = os.path.join(SAVE_DIR, 'Table_1_2_UQPP1_Unconstrained_K.csv')
    df.to_csv(csv_file, index=False)
    print(f"✓ Table exported to {csv_file}")

    plt.figure(figsize=(10, 6))
    plt.plot(n_ranges, da_times['ER'], 'bo-', label='DA-L1 (ER)')
    plt.plot(n_ranges, gurobi_times['ER'], 'r^--', label='Gurobi (ER)')
    plt.plot(n_ranges, da_times['BA'], 'gs-', label='DA-L1 (BA)')
    plt.plot(n_ranges, gurobi_times['BA'], 'mD--', label='Gurobi (BA)')
    plt.title('Computation Time vs. Number of Nodes for UQPP_1 (Unconstrained |K|)')
    plt.xlabel('Number of Nodes')
    plt.xscale('log')
    plt.ylabel('Computation Time (s)')
    plt.yscale('log')
    plt.xticks(n_ranges)
    plt.grid(True, which="both", linestyle=':')
    plt.legend()
    plt.savefig(
        os.path.join(SAVE_DIR, 'Fig_1_2_UQPP1_vs_Nodes_Unconstrained_K.png'),
        dpi=300, bbox_inches='tight'
    )
    plt.close()


def run_part2(N_TRIALS, TIME_LIMIT, SIGMA):
    """Table 2: UQPP_H scalability — DA-HW vs. Gurobi MILP.

    Benchmarks DA-HW (Algorithm 2) vs. the Gurobi MILP formulation for
    the weighted Hamming cost model across graph sizes n ∈ {100, …, 500}
    and two topologies (ER, BA). Budget B = 20% of total weighted Hamming
    upgrade cost (Table 2, Section 4).

    Args:
        N_TRIALS: Number of random instances per configuration.
        TIME_LIMIT: Gurobi time limit in seconds.
        SIGMA: Data size sigma for quickest path evaluation.
    """
    print("\n" + "=" * 85)
    print("=== PART 2: UQPP_H vs Gurobi (Varying Nodes) ===")
    print("=" * 85)

    warmup_benchmark(warmup_trials=3, warmup_nodes=50, sigma=SIGMA)

    n_ranges = [100, 200, 300, 400, 500]
    results = []
    da_times = {'ER': [], 'BA': []}
    gurobi_times = {'ER': [], 'BA': []}

    for topo in ['ER', 'BA']:
        for n_val in n_ranges:
            s_da, s_mip, s_gap, to_cnt = 0, 0, 0, 0
            actual_k_mean = 0

            for _ in range(N_TRIALS):
                G, s, t = generate_graph(topo, n_val)
                budget = sum(d['cost'] for u, v, d in G.edges(data=True)) * 0.20
                actual_k_mean += get_k_size(G)

                obj_gurobi, t_mip = solve_mip_milp_hamming(
                    G, s, t, SIGMA, budget, unit_cost=False, time_limit=TIME_LIMIT
                )
                st = time.time()
                obj_da = solve_weighted_hamming_decomposition(G, s, t, SIGMA, budget)
                t_da = time.time() - st

                s_da += t_da
                if t_mip >= TIME_LIMIT or obj_gurobi == float('inf'):
                    s_mip += TIME_LIMIT
                    to_cnt += 1
                else:
                    s_mip += t_mip
                    if obj_gurobi > 1e-6:
                        s_gap += abs(obj_da - obj_gurobi) / obj_gurobi * 100

            avg_k = int(actual_k_mean / N_TRIALS)
            da_times[topo].append(s_da / N_TRIALS)
            gurobi_times[topo].append(s_mip / N_TRIALS)

            m_str, d_str, sp, gap_str = compute_metrics(
                s_da, s_mip, s_gap, to_cnt, N_TRIALS, TIME_LIMIT
            )
            results.append({
                "Topology": topo, "Nodes": n_val, "|K|": avg_k,
                "Gurobi (s)": m_str, "DA-HW (s)": d_str,
                "Speedup": sp, "Gap (%)": gap_str
            })
            print(
                f"[{topo}] Nodes: {n_val:4d} | |K|: {avg_k:4d} | "
                f"DA: {s_da / N_TRIALS:.3f}s | Gurobi: {m_str}s | Gap: {gap_str}%"
            )

    df = pd.DataFrame(results)
    print("\nTable 2: UQPP_H Algorithm vs. Gurobi")
    print(df.to_string(index=False))

    csv_file = os.path.join(SAVE_DIR, 'Table_2_UQPP_H.csv')
    df.to_csv(csv_file, index=False)
    print(f"✓ Table exported to {csv_file}")

    plt.figure(figsize=(10, 6))
    plt.plot(n_ranges, da_times['ER'], 'bo-', label='DA-HW (ER)')
    plt.plot(n_ranges, gurobi_times['ER'], 'r^--', label='Gurobi (ER)')
    plt.plot(n_ranges, da_times['BA'], 'gs-', label='DA-HW (BA)')
    plt.plot(n_ranges, gurobi_times['BA'], 'mD--', label='Gurobi (BA)')
    plt.title('Computation Time vs. Number of Nodes for UQPP_H')
    plt.xlabel('Number of Nodes')
    plt.ylabel('Computation Time (s)')
    plt.yscale('log')
    plt.xticks(n_ranges)
    plt.grid(True, which="both", linestyle=':')
    plt.legend()
    plt.savefig(
        os.path.join(SAVE_DIR, 'Fig_2_UQPPH_vs_Nodes.png'),
        dpi=300, bbox_inches='tight'
    )
    plt.close()


def run_part3(N_TRIALS, TIME_LIMIT, SIGMA):
    """Table 3: Cardinality-constrained UQPP — DA-HC vs. Gurobi MILP.

    Benchmarks DA-HC (Algorithm 3, exact polynomial-time) vs. the Gurobi
    MILP formulation for the unit Hamming (cardinality) model across graph
    sizes n ∈ {100, …, 500} and three budget levels k ∈ {25%, 50%, 75%}
    of |E| (Table 3, Section 4). ER topology only.

    Args:
        N_TRIALS: Number of random instances per configuration.
        TIME_LIMIT: Gurobi time limit in seconds.
        SIGMA: Data size sigma for quickest path evaluation.
    """
    print("\n" + "=" * 85)
    print("=== PART 3: Cardinality Constrained UQPP vs Gurobi ===")
    print("=" * 85)

    warmup_benchmark(warmup_trials=3, warmup_nodes=50, sigma=SIGMA)

    n_ranges = [100, 200, 300, 400, 500]
    b_ratios = [0.25, 0.50, 0.75]  # cardinality budget k = ratio * |E|
    results = []

    plot_data_da = {ratio: [] for ratio in b_ratios}
    plot_data_gurobi = {ratio: [] for ratio in b_ratios}

    for ratio in b_ratios:
        print(f"\n--- Testing Budget = {int(ratio * 100)}% of edges ---")
        for n_val in n_ranges:
            s_da, s_mip, s_gap, to_cnt = 0, 0, 0, 0
            actual_k_mean = 0

            for _ in range(N_TRIALS):
                G, s, t = generate_graph('ER', n_val)
                m = G.number_of_edges()
                k_budget = max(1, int(m * ratio))  # cardinality budget k
                actual_k_mean += get_k_size(G)

                obj_gurobi, t_mip = solve_mip_milp_hamming(
                    G, s, t, SIGMA, k_budget, unit_cost=True, time_limit=TIME_LIMIT
                )
                st = time.time()
                obj_da = solve_cardinality_layered(G, s, t, SIGMA, k_budget)
                t_da = time.time() - st

                s_da += t_da
                if t_mip >= TIME_LIMIT or obj_gurobi == float('inf'):
                    s_mip += TIME_LIMIT
                    to_cnt += 1
                else:
                    s_mip += t_mip
                    if obj_gurobi > 1e-6:
                        s_gap += abs(obj_da - obj_gurobi) / obj_gurobi * 100

            avg_k = int(actual_k_mean / N_TRIALS)
            plot_data_da[ratio].append(s_da / N_TRIALS)
            plot_data_gurobi[ratio].append(s_mip / N_TRIALS)

            m_str, d_str, sp, gap_str = compute_metrics(
                s_da, s_mip, s_gap, to_cnt, N_TRIALS, TIME_LIMIT
            )
            results.append({
                "Topology": "ER", "Nodes": n_val, "|K|": avg_k,
                "Budget": f"{int(ratio * 100)}%",
                "Gurobi (s)": m_str, "DA-HC (s)": d_str,
                "Speedup": sp, "Gap (%)": gap_str
            })
            print(
                f"  Nodes: {n_val:4d} | Budget: {int(ratio * 100)}% | "
                f"DA: {s_da / N_TRIALS:.3f}s | Gurobi: {m_str}s | Gap: {gap_str}%"
            )

    df = pd.DataFrame(results)
    print("\nTable 3: Cardinality Constrained UQPP vs. Gurobi (ER Topology)")
    print(df.to_string(index=False))

    csv_file = os.path.join(SAVE_DIR, 'Table_3_Cardinality_Constrained_UQPP.csv')
    df.to_csv(csv_file, index=False)
    print(f"✓ Table exported to {csv_file}")

    plt.figure(figsize=(10, 6))
    colors = {0.25: 'blue', 0.50: 'green', 0.75: 'red'}
    for ratio in b_ratios:
        plt.plot(
            n_ranges, plot_data_da[ratio],
            marker='o', linestyle='-', color=colors[ratio],
            label=f'DA-HC ({int(ratio * 100)}% Budget)'
        )
        plt.plot(
            n_ranges, plot_data_gurobi[ratio],
            marker='^', linestyle='--', color=colors[ratio],
            label=f'Gurobi ({int(ratio * 100)}% Budget)'
        )

    plt.title('Computation Time vs. Nodes for Cardinality Constrained UQPP')
    plt.xlabel('Number of Nodes')
    plt.ylabel('Computation Time (s)')
    plt.yscale('log')
    plt.xticks(n_ranges)
    plt.grid(True, which="both", linestyle=':')
    plt.legend()
    plt.savefig(
        os.path.join(SAVE_DIR, 'Fig_3_Cardinality_vs_Nodes.png'),
        dpi=300, bbox_inches='tight'
    )
    plt.close()


def run_part4(N_TRIALS: int, TIME_LIMIT: int,
              sigma_values: list = None, n_nodes: int = 300,
              topologies: list = None, budget_ratio: float = 0.50) -> None:
    """Table 4: Sensitivity analysis — algorithm performance vs. data size sigma.

    Evaluates how the data size sigma affects the running time and solution
    quality of DA-L1, DA-HW, and DA-HC relative to their Gurobi benchmarks
    (Table 4, Section 4). Graphs are fixed at n_nodes nodes, and sigma is
    varied across sigma_values. The transmission time T(P) = L(P) + sigma/C(P)
    is directly affected by sigma, making it the primary sensitivity parameter.

    Args:
        N_TRIALS: Number of random graph instances per sigma value (must be > 0).
        TIME_LIMIT: Gurobi time limit in seconds (must be > 0).
        sigma_values: List of sigma values to sweep. Default: [1000, …, 10000].
        n_nodes: Number of nodes |V| in generated graphs. Default: 300.
        topologies: List of topologies to test ('ER', 'BA'). Default: ['ER'].
        budget_ratio: Budget B as a fraction of total edge cost. Default: 0.50.

    Raises:
        ValueError: If N_TRIALS or TIME_LIMIT are not positive, if budget_ratio
            is outside (0, 1], or if n_nodes < 10.
    """
    # --- Input validation ---
    if N_TRIALS <= 0 or TIME_LIMIT <= 0:
        raise ValueError(
            f"N_TRIALS and TIME_LIMIT must be positive, "
            f"got N_TRIALS={N_TRIALS}, TIME_LIMIT={TIME_LIMIT}"
        )
    if TIME_LIMIT < 10:
        print(f"Warning: TIME_LIMIT={TIME_LIMIT}s is very short, Gurobi may not find solutions")
    if budget_ratio <= 0 or budget_ratio > 1:
        raise ValueError(f"budget_ratio must be in (0, 1], got {budget_ratio}")
    if n_nodes < 10:
        raise ValueError(f"n_nodes must be >= 10, got {n_nodes}")

    if sigma_values is None:
        sigma_values = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
    if topologies is None:
        topologies = ['ER']

    warmup_benchmark(
        warmup_trials=3, warmup_nodes=50,
        sigma=sigma_values[0] if sigma_values else 1000
    )

    print("\n" + "=" * 85)
    print("=== PART 4: SENSITIVITY TO DATA SIZE (Sigma) ===")
    print("=" * 85)

    results = []
    times_da = {'L1': [], 'HW': [], 'HC': []}
    gurobi_times_by_alg = {'L1': [], 'HW': [], 'HC': []}

    # Algorithm configuration: (gurobi_solver, da_solver, gurobi_kwargs)
    alg_config = {
        'L1': (solve_mip_miqcp_l1,    solve_linear_decomposition,         {}),
        'HW': (solve_mip_milp_hamming, solve_weighted_hamming_decomposition, {'unit_cost': False}),
        'HC': (solve_mip_milp_hamming, solve_cardinality_layered,           {'unit_cost': True}),
    }

    print(f"Evaluating stability across {len(sigma_values)} Sigma values...")
    print(f"  Topology: {topologies}, Nodes: {n_nodes}, Budget Ratio: {budget_ratio * 100:.0f}%")
    print(
        f"  Running {N_TRIALS} trials × {len(sigma_values)} sigmas × 3 algorithms "
        f"= {N_TRIALS * len(sigma_values) * 3} solves\n"
    )

    for sig in sigma_values:
        print(f"Testing Sigma: {sig:5d}...")

        # Per-sigma accumulators
        s_da = {'L1': 0, 'HW': 0, 'HC': 0}
        s_gurobi = {'L1': 0, 'HW': 0, 'HC': 0}
        s_gap = {'L1': 0, 'HW': 0, 'HC': 0}
        timeout_cnt = {'L1': 0, 'HW': 0, 'HC': 0}
        actual_k_mean = 0

        for G, s, t in graph_generator(topologies[0], n_nodes, N_TRIALS):
            m = G.number_of_edges()
            budget_cost = sum(d['cost'] for u, v, d in G.edges(data=True)) * budget_ratio
            k_budget = max(1, int(m * budget_ratio))  # cardinality budget for HC
            actual_k_mean += get_k_size(G)

            for alg_name in ['L1', 'HW', 'HC']:
                print(f"  Running {alg_name} variant...")
                gurobi_solver, da_solver, gurobi_kwargs = alg_config[alg_name]

                # B is cost budget for L1/HW; cardinality k for HC
                budget = k_budget if alg_name == 'HC' else budget_cost

                t_da, t_gurobi, gap, is_timeout = _run_single_algorithm(
                    G, s, t, sig, gurobi_solver, da_solver, budget,
                    gurobi_kwargs, TIME_LIMIT
                )

                s_da[alg_name] += t_da
                s_gurobi[alg_name] += t_gurobi
                s_gap[alg_name] += gap
                if is_timeout:
                    timeout_cnt[alg_name] += 1

        avg_k = int(actual_k_mean / N_TRIALS)

        # Store per-sigma average times for plotting
        for alg in ['L1', 'HW', 'HC']:
            times_da[alg].append(s_da[alg] / N_TRIALS)
            gurobi_times_by_alg[alg].append(s_gurobi[alg] / N_TRIALS)

        # Compute formatted metrics for all three algorithm variants
        metrics = {}
        for alg in ['L1', 'HW', 'HC']:
            m_str, d_str, sp, g = compute_metrics(
                s_da[alg], s_gurobi[alg], s_gap[alg],
                timeout_cnt[alg], N_TRIALS, TIME_LIMIT
            )
            metrics[alg] = (m_str, d_str, g)

        results.append({
            "Topology": topologies[0], "Nodes": n_nodes, "|K|": avg_k, "Sigma": sig,
            "Gurobi L1(s)": metrics['L1'][0], "DA-L1(s)": metrics['L1'][1], "Gap L1(%)": metrics['L1'][2],
            "Gurobi HW(s)": metrics['HW'][0], "DA-HW(s)": metrics['HW'][1], "Gap HW(%)": metrics['HW'][2],
            "Gurobi HC(s)": metrics['HC'][0], "DA-HC(s)": metrics['HC'][1], "Gap HC(%)": metrics['HC'][2],
        })
        print(
            f"  Sigma: {sig:5d} | DA-L1: {metrics['L1'][1]}s (Gap {metrics['L1'][2]}%) | "
            f"DA-HW: {metrics['HW'][1]}s (Gap {metrics['HW'][2]}%) | "
            f"DA-HC: {metrics['HC'][1]}s (Gap {metrics['HC'][2]}%)"
        )

    df = pd.DataFrame(results)
    print("\nTable 4: Sensitivity to Data Size (Sigma)")
    print(df.to_string(index=False))

    csv_file = os.path.join(SAVE_DIR, 'Table_4_Sensitivity_Sigma.csv')
    df.to_csv(csv_file, index=False)
    print(f"✓ Table exported to {csv_file}")

    # --- Plot: computation time vs. sigma for all six solver/algorithm curves ---
    plt.figure(figsize=(10, 6))

    if times_da['L1']:
        plt.plot(sigma_values, times_da['L1'], marker='o', linestyle='-', color='blue', label='DA-L1')
    if times_da['HW']:
        plt.plot(sigma_values, times_da['HW'], marker='s', linestyle='-', color='green', label='DA-HW')
    if times_da['HC']:
        plt.plot(sigma_values, times_da['HC'], marker='D', linestyle='-', color='purple', label='DA-HC')

    if gurobi_times_by_alg['L1']:
        plt.plot(sigma_values, gurobi_times_by_alg['L1'], marker='o', linestyle='--', color='cyan', label='Gurobi L1')
    if gurobi_times_by_alg['HW']:
        plt.plot(sigma_values, gurobi_times_by_alg['HW'], marker='s', linestyle='--', color='lime', label='Gurobi HW')
    if gurobi_times_by_alg['HC']:
        plt.plot(sigma_values, gurobi_times_by_alg['HC'], marker='D', linestyle='--', color='magenta', label='Gurobi HC')

    all_times = []
    for alg in ['L1', 'HW', 'HC']:
        all_times.extend(times_da[alg])
        all_times.extend(gurobi_times_by_alg[alg])
    if all_times:
        plt.ylim(0, max(all_times) * 1.5)

    plt.title('Sensitivity of Proposed Algorithms and Gurobi to Data Size (Sigma)')
    plt.xlabel('Data Size (Sigma)')
    plt.ylabel('Computation Time (s)')
    plt.xticks(sigma_values)
    plt.grid(True, linestyle=':')
    plt.legend()
    plt.savefig(
        os.path.join(SAVE_DIR, 'Fig_4_Sensitivity_Sigma.png'),
        dpi=300, bbox_inches='tight'
    )
    plt.close()

    print(f"\n✓ Plot saved to {os.path.join(SAVE_DIR, 'Fig_4_Sensitivity_Sigma.png')}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    N_TRIALS = 1        # number of random instances per configuration
    TIME_LIMIT = 600     # Gurobi time limit in seconds
    SIGMA_DEFAULT = 1000 # default data size sigma for Tables 1–3

    print("=" * 85)
    print("STARTING BENCHMARK SUITE")
    print(f"Number of Trials: {N_TRIALS}")
    print(f"Plots will be saved to: {SAVE_DIR}")
    print("=" * 85)

    run_part1_1(N_TRIALS, TIME_LIMIT, SIGMA_DEFAULT)
    run_part1_2(N_TRIALS, TIME_LIMIT, SIGMA_DEFAULT)
    run_part2(N_TRIALS, TIME_LIMIT, SIGMA_DEFAULT)
    run_part3(N_TRIALS, TIME_LIMIT, SIGMA_DEFAULT)
    run_part4(N_TRIALS, TIME_LIMIT)

    print("\n" + "=" * 85)
    print("BENCHMARK SUITE COMPLETED SUCCESSFULLY.")
    print("All tables printed above. All figures saved as PNG files in the working directory.")
    print("=" * 85)

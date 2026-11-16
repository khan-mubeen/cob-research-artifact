"""the freight-corner assignment: which sourced candidate each freight agent prices with

a pure, deterministic search over every bijective assignment. its only inputs are
  demand_counts      calibration-sample orders per (city, weight piece)
  candidate_domains  each candidate's weight domain: min_kg, max_kg, max_inclusive
  service_regions    each freight agent's existing v4 service regions
it reads no price, F, alpha, damage probability, specialty text or diagnostic output,
and imports nothing that could supply them

weight pieces: the candidates' domain boundaries cut the weight axis into pieces, each
either a single boundary weight (exactly 500 kg lies in three domains) or the open
stretch between two boundaries. a piece lies wholly inside or wholly outside every
domain, so each order is counted once, in the piece its exact weight falls in

demand per (city, candidate) counts an order toward every domain that contains its
weight, and the 1% meaningful threshold is applied to that demand. the objective counts
orders, so an order inside two overlapping domains still counts once:
  1. orders lying in at least one meaningful cell of their city that have at least one
     eligible provider
  2. among ties, such orders with at least two eligible providers
  3. among ties, the lexicographically smallest candidate sequence, agents sorted by id
an eligible provider is a freight agent whose service regions include the order's city
and whose assigned candidate's domain contains the order's weight piece
"""

from itertools import permutations

MEANINGFUL_PERCENT = 1


def boundaries(candidate_domains):
    return sorted({d["min_kg"] for d in candidate_domains.values()}
                  | {d["max_kg"] for d in candidate_domains.values()})


def piece_of(weight_kg, candidate_domains):
    """(w, w) when a weight sits exactly on a boundary, else the open (low, high) around it"""
    points = boundaries(candidate_domains)
    if weight_kg < points[0]:
        raise ValueError(f"{weight_kg} kg lies below every candidate boundary")
    for low, high in zip(points, points[1:] + [float("inf")]):
        if weight_kg == low:
            return (low, low)
        if low < weight_kg < high:
            return (low, high)
    raise ValueError(f"{weight_kg} kg fits no weight piece")


def contains(domain, piece):
    low, high = piece
    if low < domain["min_kg"]:
        return False
    if low == high:
        return low < domain["max_kg"] or (low == domain["max_kg"] and domain["max_inclusive"])
    return high <= domain["max_kg"]


def piece_label(piece):
    low, high = piece
    if low == high:
        return f"exactly {low:g} kg"
    return f"({low:g}, {'inf' if high == float('inf') else format(high, 'g')}) kg"


def city_orders(demand_counts):
    totals = {}
    for (city, _), count in demand_counts.items():
        totals[city] = totals.get(city, 0) + count
    return totals


def domain_demand(demand_counts, candidate_domains):
    """orders per (city, candidate), an order counting toward every domain that contains it"""
    demand = {}
    for (city, piece), count in demand_counts.items():
        for candidate, domain in candidate_domains.items():
            if contains(domain, piece):
                demand[(city, candidate)] = demand.get((city, candidate), 0) + count
    return demand


def meaningful_cells(demand_counts, candidate_domains):
    """(city, candidate) cells whose demand is at least 1% of the city's orders"""
    totals = city_orders(demand_counts)
    return {cell for cell, count in domain_demand(demand_counts, candidate_domains).items()
            if count * 100 >= MEANINGFUL_PERCENT * totals[cell[0]]}


def providers(assignment, regions, city, inside):
    return sum(1 for agent, candidate in assignment.items() if candidate in inside and city in regions[agent])


def solve(demand_counts, candidate_domains, service_regions):
    """every bijective assignment of candidates to agents, scored and ranked best first"""
    agents, candidates = sorted(service_regions), sorted(candidate_domains)
    if len(agents) != len(candidates):
        raise ValueError(f"a bijection needs as many candidates ({len(candidates)}) as agents ({len(agents)})")
    regions = {agent: frozenset(cities) for agent, cities in service_regions.items()}
    meaningful = meaningful_cells(demand_counts, candidate_domains)
    groups = []
    for (city, piece), count in sorted(demand_counts.items()):
        inside = frozenset(c for c in candidates if contains(candidate_domains[c], piece))
        cells = sum(1 for c in inside if (city, c) in meaningful)
        if cells:
            groups.append((city, count, inside, cells))
    rows = []
    for sequence in permutations(candidates):
        assignment = dict(zip(agents, sequence))
        once = twice = cell_once = cell_twice = 0
        for city, count, inside, cells in groups:
            found = providers(assignment, regions, city, inside)
            if found >= 1:
                once, cell_once = once + count, cell_once + count * cells
            if found >= 2:
                twice, cell_twice = twice + count, cell_twice + count * cells
        rows.append({"sequence": sequence, "assignment": assignment, "covered_once": once, "covered_twice": twice,
                     "cell_weighted_once": cell_once, "cell_weighted_twice": cell_twice})
    rows.sort(key=lambda r: (-r["covered_once"], -r["covered_twice"], r["sequence"]))
    return rows


def cell_coverage(assignment, demand_counts, candidate_domains, service_regions, cells):
    """per (city, candidate) cell: its orders, and how many have one or more, and two or more,
    eligible providers under an assignment"""
    regions = {agent: frozenset(cities) for agent, cities in service_regions.items()}
    coverage = {cell: {"orders": 0, "with_one_or_more": 0, "with_two_or_more": 0} for cell in cells}
    for (city, piece), count in demand_counts.items():
        inside = frozenset(c for c, d in candidate_domains.items() if contains(d, piece))
        found = providers(assignment, regions, city, inside)
        for candidate in inside:
            cell = coverage.get((city, candidate))
            if cell is None:
                continue
            cell["orders"] += count
            cell["with_one_or_more"] += count if found >= 1 else 0
            cell["with_two_or_more"] += count if found >= 2 else 0
    return coverage

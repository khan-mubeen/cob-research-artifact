"""unit tests for the blind bijective freight assignment.
runnable directly (python ...) or under pytest.

the algorithm is checked on hand-made inputs whose right answer is known, on the seven
real candidate domains, and for blindness: its signature, its imports, the fields it
actually touches, and invariance to every price or damage value it could be handed
"""

import ast
import inspect

from src.data_synthesis.v5 import freight_assignment as fa
from src.data_synthesis.v5 import freight_candidates

# the seven sourced domains exactly as listed: number -> (min kg, max kg, max inclusive)
LITERAL_DOMAINS = {1: (0.0, 1000.0, False), 2: (1000.0, 1500.0, False), 3: (1500.0, 2000.0, False),
                   4: (2000.0, 2500.0, False), 5: (2500.0, 3000.0, False), 6: (35.0, 500.0, True),
                   7: (500.0, 1000.0, True)}


def domains(literal=LITERAL_DOMAINS):
    return {n: {"min_kg": lo, "max_kg": hi, "max_inclusive": inc} for n, (lo, hi, inc) in literal.items()}


def inside(weight):
    piece = fa.piece_of(weight, domains())
    return {n for n, d in domains().items() if fa.contains(d, piece)}


def test_the_literal_domains_are_the_reused_candidates():
    assert freight_candidates.domains(freight_candidates.load_candidates()) == domains()


def test_pieces_respect_every_boundary_and_its_inclusivity():
    assert inside(20.0) == {1}
    assert inside(35.0) == {1, 6} and inside(34.999) == {1}
    assert inside(300.0) == {1, 6}
    assert inside(500.0) == {1, 6, 7}
    assert inside(500.001) == {1, 7} and inside(999.99) == {1, 7}
    assert inside(1000.0) == {2, 7}
    assert inside(1499.99) == {2} and inside(1500.0) == {3}
    assert inside(2999.9) == {5} and inside(3000.0) == set()
    assert fa.piece_of(500.0, domains()) == (500.0, 500.0)
    assert fa.piece_of(600.0, domains()) == (500.0, 1000.0)
    assert fa.piece_label((500.0, 1000.0)) == "(500, 1000) kg" and fa.piece_label((1000.0, 1000.0)) == "exactly 1000 kg"


def test_domain_demand_counts_an_order_in_every_domain_that_contains_it():
    demand = {("A", fa.piece_of(600.0, domains())): 10, ("A", fa.piece_of(20.0, domains())): 5}
    by_cell = fa.domain_demand(demand, domains())
    assert by_cell == {("A", 1): 15, ("A", 7): 10}
    assert fa.city_orders(demand) == {"A": 15}


def test_the_meaningful_threshold_is_one_percent_of_the_city_inclusive():
    d = domains()
    demand = {("A", fa.piece_of(20.0, d)): 990, ("A", fa.piece_of(1200.0, d)): 10,
              ("B", fa.piece_of(20.0, d)): 991, ("B", fa.piece_of(1200.0, d)): 9}
    meaningful = fa.meaningful_cells(demand, d)
    assert ("A", 2) in meaningful and ("B", 2) not in meaningful
    assert {("A", 1), ("B", 1)} <= meaningful


TOY_DOMAINS = {1: {"min_kg": 0.0, "max_kg": 100.0, "max_inclusive": False},
               2: {"min_kg": 50.0, "max_kg": 100.0, "max_inclusive": True},
               3: {"min_kg": 100.0, "max_kg": 200.0, "max_inclusive": False}}


def toy_demand():
    return {("A", fa.piece_of(20.0, TOY_DOMAINS)): 40, ("A", fa.piece_of(70.0, TOY_DOMAINS)): 30,
            ("A", fa.piece_of(150.0, TOY_DOMAINS)): 30, ("B", fa.piece_of(150.0, TOY_DOMAINS)): 100}


def test_solve_evaluates_every_bijection_exactly_once():
    regions = {"X": ("A",), "Y": ("A", "B"), "Z": ("B",)}
    rows = fa.solve(toy_demand(), TOY_DOMAINS, regions)
    assert len(rows) == 6 and len({r["sequence"] for r in rows}) == 6
    for r in rows:
        assert sorted(r["sequence"]) == [1, 2, 3]
        assert sorted(r["assignment"]) == ["X", "Y", "Z"] and sorted(r["assignment"].values()) == [1, 2, 3]
    raises_value_error(lambda: fa.solve(toy_demand(), TOY_DOMAINS, {"X": ("A",), "Y": ("A",)}))


def raises_value_error(fn):
    try:
        fn()
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_the_objective_is_lexicographic_and_counts_overlapping_orders_once():
    regions = {"X": ("A",), "Y": ("A", "B"), "Z": ("B",)}
    rows = {r["sequence"]: r for r in fa.solve(toy_demand(), TOY_DOMAINS, regions)}
    # X=1 covers 20 and 70 in A, Y=3 covers 150 in A and B: every order once, the 70 kg order not twice
    best = rows[(1, 3, 2)]
    assert (best["covered_once"], best["covered_twice"]) == (200, 0)
    # X=1 and Y=2 both carry the 70 kg order: counted once for objective 1, and once for objective 2
    overlap = rows[(1, 2, 3)]
    assert (overlap["covered_once"], overlap["covered_twice"]) == (170, 30)
    ranked = fa.solve(toy_demand(), TOY_DOMAINS, regions)
    assert ranked[0]["sequence"] == (1, 3, 2)
    assert [r["covered_once"] for r in ranked] == sorted((r["covered_once"] for r in ranked), reverse=True)


def test_the_tie_break_is_the_smallest_candidate_sequence():
    regions = {"X": ("A",), "Y": ("A",)}
    doms = {1: {"min_kg": 0.0, "max_kg": 10.0, "max_inclusive": False},
            2: {"min_kg": 0.0, "max_kg": 10.0, "max_inclusive": False}}
    demand = {("A", fa.piece_of(5.0, doms)): 100}
    rows = fa.solve(demand, doms, regions)
    assert [r["sequence"] for r in rows] == [(1, 2), (2, 1)]
    assert rows[0]["covered_once"] == rows[1]["covered_once"] and rows[0]["covered_twice"] == rows[1]["covered_twice"]


def test_rare_cells_do_not_count_toward_the_objective():
    d = domains()
    demand = {("A", fa.piece_of(20.0, d)): 995, ("A", fa.piece_of(2200.0, d)): 5}
    regions = {f"L{i}": ("A",) for i in range(7)}
    best = fa.solve(demand, d, regions)[0]
    assert best["covered_once"] == 995


def test_cell_coverage_reports_orders_with_one_and_two_providers():
    regions = {"X": ("A",), "Y": ("A", "B"), "Z": ("B",)}
    cells = {("A", 1), ("A", 2), ("B", 3)}
    coverage = fa.cell_coverage({"X": 1, "Y": 2, "Z": 3}, toy_demand(), TOY_DOMAINS, regions, cells)
    assert coverage[("A", 1)] == {"orders": 70, "with_one_or_more": 70, "with_two_or_more": 30}
    assert coverage[("A", 2)] == {"orders": 30, "with_one_or_more": 30, "with_two_or_more": 30}
    assert coverage[("B", 3)] == {"orders": 100, "with_one_or_more": 100, "with_two_or_more": 0}


# ── blindness ───────────────────────────────────────────────────────────────

FORBIDDEN = ("price", "fixed_charge", "alpha", "beta", "damage", "specialty", "win_share", "wins", "diagnostic",
             "fuel", "budget")


def code_tokens(tree):
    """identifiers, attributes, arguments and non-docstring strings in a module"""
    docstrings = {id(node.body[0].value) for node in ast.walk(tree)
                  if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef)) and node.body
                  and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant)}
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            yield node.id
        elif isinstance(node, ast.Attribute):
            yield node.attr
        elif isinstance(node, ast.arg):
            yield node.arg
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstrings:
            yield node.value


def test_solve_takes_exactly_the_three_permitted_inputs():
    assert list(inspect.signature(fa.solve).parameters) == ["demand_counts", "candidate_domains", "service_regions"]


def test_the_assignment_module_imports_and_names_nothing_that_could_carry_price_or_damage():
    tree = ast.parse(inspect.getsource(fa))
    imports = [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
    imports += [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
    assert imports == ["itertools"], imports
    tokens = [t.lower() for t in code_tokens(tree)]
    for word in FORBIDDEN:
        assert not any(word in t for t in tokens), (word, [t for t in tokens if word in t])


class RecordingDomain(dict):
    """a domain that records every field the algorithm reads"""

    def __init__(self, values, seen):
        super().__init__(values)
        self.seen = seen

    def __getitem__(self, key):
        self.seen.add(key)
        return super().__getitem__(key)

    def get(self, key, default=None):
        self.seen.add(key)
        return super().get(key, default)


def test_solve_reads_only_domain_bounds_and_is_invariant_to_price_and_damage():
    base = domains()
    demand = {("A", fa.piece_of(w, base)): n for w, n in ((20.0, 50), (300.0, 80), (700.0, 60), (1200.0, 30),
                                                          (1700.0, 20), (2200.0, 10), (2700.0, 10))}
    demand.update({("B", fa.piece_of(w, base)): n for w, n in ((400.0, 90), (1100.0, 40))})
    regions = {"L01": ("A", "B"), "L02": ("A",), "L04": ("B",), "L05": ("A",), "L08": ("B",), "L09": ("A", "B"),
               "L11": ("A",)}
    results = []
    for poison in ({"fixed_charge_eur": 1.0, "alpha_eur_per_kg": 0.1, "damage_probability": 0.01, "specialty": "bulk"},
                   {"fixed_charge_eur": 999.0, "alpha_eur_per_kg": 9.9, "damage_probability": 0.99, "specialty": "small"}):
        seen = set()
        poisoned = {n: RecordingDomain({**d, **poison}, seen) for n, d in base.items()}
        results.append([(r["sequence"], r["covered_once"], r["covered_twice"]) for r in fa.solve(demand, poisoned, regions)])
        assert seen <= {"min_kg", "max_kg", "max_inclusive"}, seen
    assert results[0] == results[1]

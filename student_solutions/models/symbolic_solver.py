"""Symbolic rule solver for RAVEN-I (works on XML ground truth and CNN predictions)."""

import numpy as np

RULE_PRIORITY = ["Constant", "Progression", "Arithmetic", "Distribute_Three"]


def _scalar_scores(vals, allow_arithmetic=True, offset=0):
    """Score the 4 row-wise rules on 8 context ints [row1(0..2), row2(3..5), row3(6..7)]."""
    v = [int(x) for x in vals]
    s1, s2, r3 = set(v[0:3]), set(v[3:6]), set(v[6:8])

    prog = max(
        (sum([v[1]-v[0]==d, v[2]-v[1]==d, v[4]-v[3]==d, v[5]-v[4]==d, v[7]-v[6]==d]) / 5.0, v[7] + d)
        for d in (-2, -1, 1, 2)
    )
    add_sc = sum([v[2] == v[0] + v[1] + offset, v[5] == v[3] + v[4] + offset]) / 2.0
    sub_sc = sum([v[2] == v[0] - v[1] - offset, v[5] == v[3] - v[4] - offset]) / 2.0
    arith = (add_sc, v[6] + v[7] + offset) if add_sc >= sub_sc else (sub_sc, v[6] - v[7] - offset)

    rem = (s1 if len(s1) == 3 else s2) - r3
    dist_sc = sum([len(s1) == 3, len(s2) == 3, s1 == s2, v[6] != v[7] and r3.issubset(s1 | s2)]) / 4.0

    scores = {
        "Constant": sum([v[0]==v[1], v[1]==v[2], v[3]==v[4], v[4]==v[5], v[6]==v[7]]) / 5.0,
        "Progression": prog[0],
        "Arithmetic": arith[0] if allow_arithmetic else 0.0,
        "Distribute_Three": dist_sc,
    }
    preds = {
        "Constant": v[7],
        "Progression": prog[1],
        "Arithmetic": arith[1] if allow_arithmetic else None,
        "Distribute_Three": next(iter(rem)) if len(rem) == 1 else None,
    }
    return scores, preds


def infer_scalar_rule(vals, allow_arithmetic=True, offset=0, uniform_panels=None):
    """Infer (rule_name, expected_val, scores) for Type, Size, Color, or Number."""
    if uniform_panels is not None and not all(uniform_panels):
        return "Constant", None, {"Constant": 1.0, "Progression": 0.0, "Arithmetic": 0.0, "Distribute_Three": 0.0}
    scores, preds = _scalar_scores(vals, allow_arithmetic, offset)
    best = max(RULE_PRIORITY, key=lambda r: (scores[r], -RULE_PRIORITY.index(r)))
    return best, preds[best], scores


def infer_num_pos_rule(numbers, pos_sets, num_slots):
    """Infer (rule_name, attr_name, expected_val) for Number/Position across context."""
    if num_slots == 1:
        return "Constant", "Number/Position", pos_sets[7]

    p = pos_sets
    shift = lambda s, d: frozenset((i + d) % num_slots for i in s)
    cands = [("Constant", "Number/Position", sum([p[0]==p[1], p[1]==p[2], p[3]==p[4], p[4]==p[5], p[6]==p[7]]) / 5.0, p[7])]

    pos_prog = max(
        (sum([shift(p[0],d)==p[1], shift(p[1],d)==p[2], shift(p[3],d)==p[4], shift(p[4],d)==p[5], shift(p[6],d)==p[7]]) / 5.0, shift(p[7], d))
        for d in (-2, -1, 1, 2)
    )
    cands.append(("Progression", "Position", pos_prog[0], pos_prog[1]))

    u_sc = sum([p[2] == (p[0] | p[1]), p[5] == (p[3] | p[4])]) / 2.0
    d_sc = sum([p[2] == (p[0] - p[1]), p[5] == (p[3] - p[4])]) / 2.0
    cands.append(("Arithmetic", "Position", max(u_sc, d_sc), (p[6] | p[7]) if u_sc >= d_sc else (p[6] - p[7])))

    s1, s2 = {p[0], p[1], p[2]}, {p[3], p[4], p[5]}
    rem = (s1 if len(s1) == 3 else s2) - {p[6], p[7]}
    dist_sc = sum([len(s1) == 3, len(s2) == 3, s1 == s2, p[6] != p[7] and {p[6], p[7]}.issubset(s1 | s2)]) / 4.0
    cands.append(("Distribute_Three", "Position", dist_sc, next(iter(rem)) if len(rem) == 1 else None))

    num_scores, num_preds = _scalar_scores(numbers, allow_arithmetic=True, offset=1)
    for r in ("Progression", "Arithmetic", "Distribute_Three"):
        cands.append((r, "Number", num_scores[r], num_preds[r]))

    rank = {"Number/Position": 0, "Position": 1, "Number": 2}
    best = max(cands, key=lambda x: (x[2], -rank[x[1]], -RULE_PRIORITY.index(x[0])))
    return best[0], best[1], best[3]


def _match_scalar(cand_cp, p7_cp, attr, rule, expected, regen):
    """Soft match score in [0, 1] for one scalar attribute of a candidate panel."""
    vals = [e[attr] for e in cand_cp["entities"]]
    uniform = len(set(vals)) == 1
    if rule == "Constant":
        if regen:
            return 1.0  # I-RAVEN resamples non-uniform attributes when Number changes
        if expected is not None:
            return 1.0 if (uniform and vals[0] == expected) else 0.5 / (1.0 + abs(vals[0] - expected))
        return 1.0 if vals == [e[attr] for e in p7_cp["entities"]] else 0.2
    if not uniform:
        return 0.0
    if expected is None:
        return 0.5
    return 1.0 if vals[0] == expected else 0.5 / (1.0 + abs(vals[0] - expected))


def _match_num_pos(cand_cp, attr, expected):
    """Soft match score in [0, 1] for Number/Position of a candidate panel."""
    if expected is None:
        return 0.5
    if attr in ("Number/Position", "Position"):
        cpos = frozenset(cand_cp["positions"])
        if cpos == expected:
            return 1.0
        u = len(cpos | expected)
        return 0.5 * len(cpos & expected) / u if u else 0.0
    return 1.0 if cand_cp["number"] == expected else 0.5 / (1.0 + abs(cand_cp["number"] - expected))


def solve_problem(panels):
    """Infer component rules from context panels 0..7 and select best candidate 0..7."""
    rules, scores = [], np.zeros(8, dtype=np.float64)
    for c in range(len(panels[0])):
        np_rule, np_attr, np_exp = infer_num_pos_rule(
            [panels[p][c]["number"] for p in range(8)],
            [frozenset(panels[p][c]["positions"]) for p in range(8)],
            panels[0][c]["num_slots"],
        )
        regen = (np_attr == "Number") or (np_rule == "Arithmetic")
        comp_rules, specs = [(np_rule, np_attr)], {}

        for attr, allow_a, off in [("type", False, 0), ("size", True, 1), ("color", True, 0)]:
            uni = [len({e[attr] for e in panels[p][c]["entities"]}) == 1 for p in range(8)]
            r_name, exp, _ = infer_scalar_rule([panels[p][c][attr] for p in range(8)], allow_a, off, uni)
            comp_rules.append((r_name, attr.capitalize()))
            specs[attr] = (r_name, exp)
        rules.append(comp_rules)

        for k in range(8):
            scores[k] += _match_num_pos(panels[8 + k][c], np_attr, np_exp)
            for attr, (r_name, exp) in specs.items():
                scores[k] += _match_scalar(panels[8 + k][c], panels[7][c], attr, r_name, exp, regen)

    return {"rules": rules, "scores": scores, "answer": int(np.argmax(scores))}


def solve_from_arrays(type_arr, size_arr, color_arr, number_arr=None, pos_mask_arr=None, num_components=1, num_slots=1):
    """Run solve_problem() directly on (16, 2) integer arrays (type shifted 0..4 -> 1..5)."""
    panels = []
    for p in range(16):
        comps = []
        for c in range(num_components):
            pos = [int(i) for i in np.where(pos_mask_arr[p, c, :num_slots] > 0.5)[0]] if pos_mask_arr is not None else [0]
            ent = {"type": int(type_arr[p, c]) + 1, "size": int(size_arr[p, c]), "color": int(color_arr[p, c])}
            comps.append({
                "number": int(number_arr[p, c]) if number_arr is not None else 0,
                "num_slots": num_slots,
                "positions": pos,
                "entities": [ent],
                **ent,
            })
        panels.append(comps)
    return solve_problem(panels)
